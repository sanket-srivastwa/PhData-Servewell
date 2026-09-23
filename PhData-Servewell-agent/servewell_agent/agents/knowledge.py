"""Agent 2 - Knowledge / Retrieval.

Turns an enriched ticket into (a) a ranked set of KB documents and (b) the specific chunks that
will be allowed as evidence. Retrieval is *scoped by structured facts first* (domain, effective
system version, subcategory) and only then by text similarity - the query is built from the
de-emotionalised issue core + error codes, not the raw rant.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .. import config
from ..intel_platform.kb import Chunk, KnowledgeBase
from ..models import Enrichment, Ticket
from ..textutil import content_tokens

SOP_ESCALATION = "sop/escalation-procedure.md"


@dataclass
class Retrieval:
    query: str
    docs: list = field(default_factory=list)          # [(doc, score, why)]
    primary: Optional[str] = None                     # primary runbook path
    primary_score: float = 0.0
    secondary: Optional[str] = None
    chunks: list = field(default_factory=list)        # [(Chunk, score)]
    spec_doc: Optional[str] = None
    sufficient: bool = False
    gap: bool = False                                 # no runbook is named for this subcategory (KB backlog signal)

    def doc_paths(self) -> list:
        return [d for d, _, _ in self.docs]


class Retriever:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def build_query(self, t: Ticket, e: Enrichment) -> str:
        core = e.issue_core or t.description
        ver = (e.get("effective_system").evidence.get("system_version") if e.get("effective_system") else "") or ""
        return " ".join([t.subject, core, t.subcategory, " ".join(e.error_codes), ver])

    def retrieve(self, t: Ticket, e: Enrichment, escalating: bool = False) -> Retrieval:
        ver = e.get("effective_system").evidence.get("system_version") if e.get("effective_system") else None
        q = self.build_query(t, e)
        ranked = self.kb.rank_docs(q, domain=e.domain, subcategory=t.subcategory, system_version=ver, top=10)
        r = Retrieval(query=q)
        runbooks = [(d, s, w) for d, s, w in ranked if self.kb.docs[d]["type"] == "runbook"]
        if runbooks:
            r.primary, r.primary_score = runbooks[0][0], runbooks[0][1]
            if len(runbooks) > 1 and runbooks[1][1] >= 0.60 * runbooks[0][1] and runbooks[1][1] >= 0.5:
                r.secondary = runbooks[1][0]
        docs = ranked[: config.TOP_DOCS]
        extra = []
        spec = self.kb.spec_doc_for(ver)
        r.spec_doc = spec
        if spec and spec not in [d for d, _, _ in docs]:
            extra.append((spec, 0.0, "system-spec (forced by effective system)"))
        if escalating and SOP_ESCALATION not in [d for d, _, _ in docs]:
            extra.append((SOP_ESCALATION, 0.0, "escalation SOP (route = L2)"))
        r.docs = docs + extra
        r.gap = bool(t.subcategory) and not any('subcat-name' in w for _, _, w in runbooks[:3])
        r.sufficient = bool(r.primary) and r.primary_score >= config.MIN_PRIMARY_RUNBOOK_SCORE and not e.has("unknown_system")
        allowed = {d for d, _, _ in r.docs}
        r.chunks = self.kb.search_chunks(q, k=12, doc_filter=allowed)
        return r

    # ---- runbook helpers -------------------------------------------------------------------
    def runbook_chunks(self, doc: str, section_types: set) -> list:
        return [c for c in self.kb.chunks if c.doc == doc and c.section_type in section_types]

    def escalation_criteria(self, doc: Optional[str]) -> list:
        """Bullet lines from the runbook's own 'When to Escalate to L2' section."""
        if not doc:
            return []
        out = []
        for c in self.runbook_chunks(doc, {"escalate"}):
            for line in c.text.splitlines():
                m = re.match(r"^\s*[-*]\s+(.*)", line)
                if m and len(m.group(1)) > 25:
                    out.append((m.group(1).strip(), c.id))
        return out

    def escalation_hits(self, doc: Optional[str], t: Ticket, e: Enrichment) -> list:
        """Which of the runbook's own escalation criteria does this ticket already satisfy?
        Explainable, runbook-grounded signal: overlap of criterion tokens with what the ticket says
        (+ what has already been tried)."""
        hay = content_tokens(t.text + " " + t.history_text + " " + " ".join(e.tried_steps).replace("_", " "))
        hits = []
        for text, cid in self.escalation_criteria(doc):
            clean = re.sub(r"\*\*", "", text)
            if clean.rstrip().endswith(":"):                      # section header bullets, not criteria
                continue
            crit = content_tokens(clean)
            shared = crit & hay
            if len(crit) < 3:
                continue
            ov = len(shared) / len(crit)
            if (ov >= 0.6 and len(shared) >= 3) or (e.error_codes and any(c.lower() in text.lower() for c in e.error_codes)):
                hits.append({"criterion": clean[:160], "chunk": cid, "overlap": round(ov, 2)})
        return hits
