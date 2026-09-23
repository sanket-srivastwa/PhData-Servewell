"""Knowledge retrieval service (simulated "Intelligence Platform" RAG building block).

Design
------
* Section-aware chunking of markdown (runbooks / FAQ / SOP / specs). A chunk is one step-group or
  one Q&A and keeps its heading path, so every chunk is citeable
  ("runbooks/printer-offline.md > L1 Resolution > For USB Printers").
* Lexical BM25 over chunks. Error codes such as 0x8004, ERR-4521 or E04 need exact-match behaviour,
  which is where dense-only retrieval is weakest; the corpus is small (~400 KB) so BM25 is fast,
  explainable and needs no embedding service. A dense re-ranker is the documented next step.
* Metadata-aware document ranking: domain, system/version -> spec sheet, subcategory-name affinity,
  and a subcategory -> document prior learned from the labelled training tickets.
* The index is rebuilt from the files on start-up and stamped with a content hash, so newly dropped
  runbooks are picked up without code changes and every answer can cite the KB version it used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..textutil import tokenize

DOMAIN_BY_PREFIX = {
    "pos": "POS", "printer": "Printers", "printers": "Printers", "network": "Wi-Fi / Network",
    "online-orders": "Online Orders", "soft-serve": "Soft Serve", "kiosk": "Kiosks", "kiosks": "Kiosks",
}
SPEC_SYSTEMS = {
    "foodtech-pos": ["FoodTech POS"],
    "orbitpos": ["OrbitPOS"],
    "epson-printer": ["EpsonTM-T88VI"],
    "star-printer": ["StarMC Print v2"],
    "netlink-router": ["NetLink NL-3000"],
    "creamtech-soft-serve": ["CreamTech SC-300", "CreamTech SC-500"],
    "frostypro-soft-serve": ["FrostyPro 800"],
}
SECTION_TYPES = [
    ("immediate", "immediate"), ("l1 diagnosis", "diagnosis"), ("l1 resolution", "resolution"),
    ("when to escalate", "escalate"), ("symptom", "symptoms"), ("affected", "affected"),
    ("overview", "overview"), ("related", "related"), ("revision", "revision"),
    ("verification", "verification"),
]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


@dataclass
class Chunk:
    id: str
    doc: str
    doc_title: str
    doc_type: str            # runbook | faq | sop | spec | doc
    domain: str
    heading_path: list
    section_type: str
    text: str
    tokens: list = field(default_factory=list, repr=False)

    @property
    def label(self) -> str:
        hp = " > ".join(h for h in self.heading_path if h)
        return f"{self.doc} > {hp}" if hp else self.doc


def _section_type(h2: Optional[str]) -> str:
    if not h2:
        return "other"
    low = h2.lower()
    for key, typ in SECTION_TYPES:
        if low.startswith(key) or key in low:
            return typ
    return "reference"


def _split_long(body: str, limit: int = 1500, target: int = 950) -> list:
    if len(body) <= limit:
        return [body]
    parts, cur = [], []
    for line in body.splitlines():
        if (re.match(r"^\d+\.\s", line) or re.match(r"^\*\*[^*]+\*\*\s*$", line.strip())) and cur:
            parts.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        parts.append("\n".join(cur))
    out, acc = [], ""
    for p in parts:
        if acc and len(acc) + len(p) > target:
            out.append(acc)
            acc = p
        else:
            acc = (acc + "\n" + p) if acc else p
    if acc:
        out.append(acc)
    return out


def chunk_markdown(doc_rel: str, text: str, doc_type: str, domain: str) -> list:
    lines = text.splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), doc_rel)
    chunks, heads, buf = [], {2: None, 3: None, 4: None}, []
    counter = Counter()

    def flush():
        body = "\n".join(buf).strip()
        buf.clear()
        if not body or len(body) < 25:
            return
        path = [heads[2], heads[3], heads[4]]
        for piece in _split_long(body):
            key = "/".join(_slug(p) for p in path if p) or "intro"
            counter[key] += 1
            cid = f"{doc_rel}#{key}" + (f"-{counter[key]}" if counter[key] > 1 else "")
            chunks.append(Chunk(cid, doc_rel, title, doc_type, domain, [p for p in path if p],
                                _section_type(heads[2]), piece))

    for line in lines:
        m = re.match(r"^(#{2,4})\s+(.*)", line)
        if m:
            flush()
            lvl = len(m.group(1))
            heads[lvl] = m.group(2).strip()
            for l in range(lvl + 1, 5):
                heads[l] = None
            continue
        if doc_type == "faq" and re.match(r"^\*\*Q:", line.strip()):
            flush()
        if line.strip() in ("---", "***"):
            continue
        buf.append(line)
    flush()
    return chunks


class BM25:
    def __init__(self, docs_tokens: list, k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / max(1, self.N)
        self.tf = [Counter(d) for d in docs_tokens]
        self.dl = [len(d) for d in docs_tokens]
        df = Counter()
        for d in docs_tokens:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        # inverted index for speed
        self.inv = defaultdict(list)
        for i, tf in enumerate(self.tf):
            for t, f in tf.items():
                self.inv[t].append((i, f))

    def scores(self, q_tokens: list) -> list:
        out = [0.0] * self.N
        for t, qf in Counter(q_tokens).items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.inv[t]:
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                out[i] += idf * (f * (self.k1 + 1) / denom) * (1 + math.log(qf))
        return out


class KnowledgeBase:
    def __init__(self, kb_dir, prior_path: Optional[Path] = None):
        self.kb_dir = Path(kb_dir)
        self.chunks: list = []
        self.docs: dict = {}                 # rel path -> {title, type, domain, text}
        self.version_rows: list = []
        self.prior: dict = {}                # subcategory -> {doc: weight}
        self._load()
        if prior_path and Path(prior_path).exists():
            self.prior = json.loads(Path(prior_path).read_text()).get("subcat_doc_prior", {})
        self.bm25 = BM25([c.tokens for c in self.chunks])
        self.kb_hash = self._hash()

    # ---------------- loading -----------------
    def _load(self):
        for p in sorted(self.kb_dir.rglob("*.md")):
            rel = p.relative_to(self.kb_dir).as_posix()
            typ = {"runbooks": "runbook", "faq": "faq", "sop": "sop", "system-specs": "spec"}.get(rel.split("/")[0], "doc")
            name = p.stem
            parts = name.split("-")
            prefix = "-".join(parts[:2]) if name.startswith(("online-orders", "soft-serve")) else parts[0]
            domain = DOMAIN_BY_PREFIX.get(prefix, "General")
            if typ == "faq":
                domain = DOMAIN_BY_PREFIX.get(name.replace("-faq", ""), domain)
            if typ == "spec":
                domain = "Spec"
            text = p.read_text(encoding="utf-8", errors="replace")
            self.docs[rel] = {"title": next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), rel),
                              "type": typ, "domain": domain, "text": text}
            for ch in chunk_markdown(rel, text, typ, domain):
                head_tokens = tokenize(" ".join(ch.heading_path) + " " + ch.doc_title)
                ch.tokens = tokenize(ch.text) + head_tokens * 2
                self.chunks.append(ch)
        vm = self.kb_dir / "system-specs" / "version-matrix.csv"
        if vm.exists():
            with open(vm, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    r["key"] = (r["system"] + " " + r["version"]).strip()
                    self.version_rows.append(r)
        self.by_id = {c.id: c for c in self.chunks}
        for r in self.version_rows:            # version-matrix rows are citeable evidence too (not BM25-indexed)
            cid = f"system-specs/version-matrix.csv#{r['key']}"
            txt = (f"{r['system']} {r['version']} (released {r['release_date']}). Known bugs: {r['known_bugs']}. "
                   f"Patch available: {r['patch_available']}. Notes: {r['notes'] or 'none'}.")
            self.by_id[cid] = Chunk(cid, "system-specs/version-matrix.csv", "Version matrix", "matrix", "Spec",
                                    [r["key"]], "reference", txt)

    def _hash(self) -> str:
        h = hashlib.sha256()
        for rel in sorted(self.docs):
            h.update(rel.encode())
            h.update(self.docs[rel]["text"].encode())
        return h.hexdigest()[:12]

    # ---------------- catalogue helpers -----------------
    def known_systems(self) -> set:
        s = {r["system"] for r in self.version_rows}
        for v in SPEC_SYSTEMS.values():
            s.update(v)
        return s

    def version_info(self, system_version: Optional[str]) -> Optional[dict]:
        if not system_version:
            return None
        sv = system_version.strip().lower()
        for r in self.version_rows:
            if r["key"].lower() == sv:
                return r
        return None

    def is_known_system(self, system_version: Optional[str]) -> bool:
        if not system_version:
            return False
        if self.version_info(system_version):
            return True
        sv = system_version.lower()
        return any(s.lower() in sv for s in self.known_systems())

    def spec_doc_for(self, system_version: Optional[str]) -> Optional[str]:
        if not system_version:
            return None
        sv = system_version.lower()
        for name, systems in SPEC_SYSTEMS.items():
            if any(s.lower() in sv for s in systems):
                return f"system-specs/{name}.md"
        return None

    # ---------------- retrieval -----------------
    def search_chunks(self, query: str, k: int = 8, doc_filter: Optional[set] = None,
                      section_types: Optional[set] = None) -> list:
        scores = self.bm25.scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        out = []
        for i in ranked:
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            if doc_filter and c.doc not in doc_filter:
                continue
            if section_types and c.section_type not in section_types:
                continue
            out.append((c, scores[i]))
            if len(out) >= k:
                break
        return out

    def rank_docs(self, query: str, domain: str = "", subcategory: str = "",
                  system_version: Optional[str] = None, top: int = 8) -> list:
        """Doc-level ranking = normalised BM25 + domain / subcategory-name affinity + learned prior + spec boost."""
        scores = self.bm25.scores(tokenize(query))
        per_doc = defaultdict(list)
        for c, s in zip(self.chunks, scores):
            per_doc[c.doc].append(s)
        raw = {}
        for d, ss in per_doc.items():
            ss = sorted(ss, reverse=True)
            raw[d] = ss[0] + 0.25 * sum(ss[1:3])
        mx = max(raw.values()) or 1.0
        sub_toks = set(tokenize(subcategory))
        spec = self.spec_doc_for(system_version)
        ranked = []
        for d, r in raw.items():
            info = self.docs[d]
            sc = 0.55 * (r / mx)
            why = [f"bm25={r / mx:.2f}"]
            if domain and info["domain"] == domain:
                sc += 0.12
                why.append("domain")
            if info["type"] == "runbook" and sub_toks:
                name_toks = set(tokenize(Path(d).stem.replace("-", " ")))
                aff = len(sub_toks & name_toks) / max(1, len(sub_toks))
                if aff:
                    sc += 0.30 * aff
                    why.append(f"subcat-name={aff:.2f}")
            pw = self.prior.get(subcategory, {}).get(d, 0.0)
            if pw:
                sc += 0.30 * pw
                why.append(f"prior={pw:.2f}")
            if spec and d == spec:
                sc += 0.15
                why.append("system-spec")
            if info["type"] == "sop" and "escalation" not in d:
                sc -= 0.15
            ranked.append((d, sc, ", ".join(why)))
        ranked.sort(key=lambda x: -x[1])
        return ranked[:top]
