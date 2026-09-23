"""Guardrails - independent of whichever component produced the text.

Checks (each returns a named, loggable result):
  grounding      every guidance step must cite a retrieved chunk that exists AND the step's content must be
                 supported by that chunk (token overlap) AND every "hard entity" in the step (commands,
                 file paths, error codes, versions, menu paths) must literally appear in the cited chunk.
                 This is what catches an LLM inventing a plausible-sounding registry edit.
  message        no promises the SOP doesn't make (refunds, guaranteed ETAs), no PII echo, no internal-only
                 contacts to a store, acknowledgement present when the requester is upset
  actions        only allow-listed actions; state-changing ones never auto-execute
  coverage       enough grounded steps survive to be worth sending (else downgrade to human/L2)
  leak           dataset answer fields were dropped at intake
"""
from __future__ import annotations

import re
from typing import Optional

from .. import config
from ..intel_platform.itsm import ACTION_CATALOG
from ..intel_platform.kb import KnowledgeBase
from ..models import Enrichment, ProposedAction, Step, Ticket
from ..textutil import content_tokens

ENTITY = re.compile(
    r"(`[^`]+`"                                   # inline code / commands
    r"|[A-Za-z]:\\[\w\\\.\-]+"                     # windows paths
    r"|\b\w+\.(?:msc|exe|dll|conf|cfg|ini|bat|sys)\b"
    r"|\b0x[0-9A-Fa-f]{3,8}\b"                     # hex codes
    r"|\b[A-Z]{1,5}-\d{3,4}\b"                     # error codes
    r"|\bv?\d+\.\d+(?:\.\d+)?\b"                   # versions
    r"|\b\d{1,3}(?:\.\d{1,3}){3}\b"                # IPs
    r"|\b\d+\s?(?:seconds?|minutes?|mins?|hours?|°C|C\b|%)\b)"  # quantities
)
FORBIDDEN_COMMITMENTS = [
    r"\brefund", r"\bcompensat", r"\bguarantee", r"\bwe promise", r"\bwill be fixed (by|within)", r"\b100%",
    r"\bfree of charge", r"\breplacement (unit )?(will|has) (be|been) (shipped|sent|dispatched)",
]
INTERNAL_ONLY = re.compile(r"\b(?:it-l2|l2-lead|oncall|it-manager)@servewell\.in\b", re.I)
PHONE = re.compile(r"\+?\d[\d\-\s]{8,}\d")
ACK = re.compile(r"(sorry|apolog|understand|frustrat|appreciate your patience|thank you for)", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[\s\*_]+", " ", s.lower())


class Guardrails:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    # ---------------------------------------------------------------- grounding
    def check_step(self, text: str, source_ids: list, allowed_ids: set, min_overlap: float) -> tuple:
        """returns (ok, score, reason)"""
        if not source_ids:
            return False, 0.0, "no citation"
        srcs = []
        for sid in source_ids:
            if sid not in allowed_ids:
                return False, 0.0, f"cites {sid!r}, which was not in the retrieved evidence set"
            ch = self.kb.by_id.get(sid)
            if not ch:
                return False, 0.0, f"cites unknown chunk {sid!r}"
            srcs.append(ch.text)
        src_txt = _norm(" ".join(srcs))
        toks = content_tokens(text)
        if not toks:
            return False, 0.0, "empty step"
        src_toks = content_tokens(" ".join(srcs))
        overlap = len(toks & src_toks) / len(toks)
        bad = [m.group(0) for m in ENTITY.finditer(text) if _norm(m.group(0).strip("`")) not in src_txt]
        if bad:
            return False, overlap, f"entities not present in cited source: {bad[:3]}"
        if overlap < min_overlap:
            return False, overlap, f"only {overlap:.0%} of the step's content is supported by its source"
        return True, overlap, "supported"

    def check_steps(self, steps: list, allowed_ids: set, llm_generated: bool) -> tuple:
        thr = config.GROUNDING_MIN_OVERLAP_LLM if llm_generated else config.GROUNDING_MIN_OVERLAP_EXTRACTIVE
        kept, dropped = [], []
        for s in steps:
            ok, score, why = self.check_step(s.text, [s.source_id] + list(s.extra_sources), allowed_ids, thr)
            s.grounding_score = round(score, 2)
            s.grounded = ok
            (kept if ok else dropped).append((s, why))
        return [s for s, _ in kept], [{"step": s.title, "reason": w} for s, w in dropped]

    # ---------------------------------------------------------------- message
    def check_message(self, msg: str, t: Ticket, e: Enrichment, audience: str = "requester") -> dict:
        issues = []
        for pat in FORBIDDEN_COMMITMENTS:
            if re.search(pat, msg, re.I):
                issues.append(f"forbidden commitment: /{pat}/")
        if audience == "requester":
            if INTERNAL_ONLY.search(msg):
                issues.append("internal-only mailbox exposed to requester")
            if t.contact_phone and t.contact_phone.replace(" ", "") in msg.replace(" ", ""):
                issues.append("requester phone echoed")
            if e.dissatisfied and not ACK.search(msg):
                issues.append("upset requester but no acknowledgement in message")
        if len(msg) > 3000:
            issues.append("message too long")
        return {"passed": not issues, "issues": issues}

    # ---------------------------------------------------------------- actions
    def check_actions(self, actions: list) -> dict:
        issues = []
        for a in actions:
            if a.action not in ACTION_CATALOG:
                issues.append(f"{a.action}: not in allow-list")
            elif a.risk != "read_only" and not a.requires_approval and ACTION_CATALOG[a.action][1] is False:
                issues.append(f"{a.action}: state-changing without approval gate")
        return {"passed": not issues, "issues": issues}

    # ---------------------------------------------------------------- summary
    def report(self, *, steps_in: int, steps_kept: int, dropped: list, msg_check: dict, act_check: dict,
               t: Ticket, e: Enrichment, sufficient: bool, llm_generated: bool) -> dict:
        checks = [
            {"name": "retrieval_sufficient", "passed": sufficient,
             "detail": "primary runbook found with adequate score" if sufficient else "no adequate runbook - no steps issued"},
            {"name": "grounding", "passed": not dropped, "detail":
                f"{steps_kept}/{steps_in} steps grounded" + (f"; dropped: {dropped}" if dropped else "") +
                (" (LLM-generated text held to entity + overlap checks)" if llm_generated else " (verbatim runbook text)")},
            {"name": "message_policy", "passed": msg_check["passed"], "detail": "; ".join(msg_check["issues"]) or "ok"},
            {"name": "action_allowlist", "passed": act_check["passed"], "detail": "; ".join(act_check["issues"]) or "ok"},
            {"name": "input_untrusted_data", "passed": not e.injection,
             "detail": "instruction-like content found in ticket; ignored, flagged" if e.injection else "no injection patterns"},
            {"name": "label_leak_guard", "passed": True,
             "detail": f"dropped from input: {t.leak_guard_dropped}" if t.leak_guard_dropped else "no answer fields present"},
        ]
        return {"passed": msg_check["passed"] and act_check["passed"], "interventions": len(dropped),
                "checks": checks, "dropped_steps": dropped}
