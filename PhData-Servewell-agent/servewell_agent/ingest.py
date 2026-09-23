"""Ticket ingestion: turns any of {ITSM JSON, CSV row, plain text/email} into a `Ticket`.

Two design points worth calling out in the walkthrough:

* **Tolerant, not trusting.** Missing fields don't crash the run; they are recorded in
  `raw_missing_fields` and later surface as "missing mandatory information" findings.
* **Leak guard.** In the training data `resolution_notes` (and terminal statuses) contain the answer.
  A production agent never sees those at intake, so we drop them here and record that we did. This is
  what keeps the offline evaluation honest.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .models import Ticket

LEAKY_FIELDS = ("resolution_notes",)
EXPECTED = ["ticket_id", "store_id", "subject", "description", "priority", "category", "subcategory", "asset_id"]


def _parse_dt(v) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except ValueError:
        try:
            return datetime.strptime(str(v)[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None


def ticket_from_dict(d: dict, strip_status: bool = False) -> Ticket:
    dropped = [k for k in LEAKY_FIELDS if d.get(k)]
    missing = [k for k in EXPECTED if not d.get(k)]
    def _flag(v):
        return v if isinstance(v, bool) else str(v or "").strip().lower() in ("true", "1", "yes", "y")

    def _listy(v):
        if isinstance(v, str):
            return [x.strip() for x in v.replace(";", ",").split(",") if x.strip()]
        return list(v or [])

    hist = d.get("ticket_history") or []
    if isinstance(hist, str):
        hist = [{"actor": "unknown", "note": hist}]
    t = Ticket(
        ticket_id=str(d.get("ticket_id") or d.get("id") or "INC-UNKNOWN"),
        subject=(d.get("subject") or d.get("title") or "").strip(),
        description=(d.get("description") or d.get("body") or "").strip(),
        store_id=d.get("store_id"),
        store_name=d.get("store_name"),
        submitted_by=d.get("submitted_by"),
        contact_phone=d.get("contact_phone"),
        date_opened=_parse_dt(d.get("date_opened")),
        priority=(d.get("priority") or "P3"),
        category=d.get("category") or "",
        subcategory=d.get("subcategory") or "",
        asset_id=d.get("asset_id") or None,
        system_version=d.get("system_version") or None,
        tags=_listy(d.get("tags")),
        escalation_flag=_flag(d.get("escalation_flag")),
        related_tickets=_listy(d.get("related_tickets")),
        ticket_history=list(hist),
        status="Open" if strip_status else (d.get("status") or "Open"),
        leak_guard_dropped=dropped,
        raw_missing_fields=missing,
    )
    return t


def ticket_from_text(text: str, ticket_id: str = "INC-TEXT-0001", store_id: Optional[str] = None) -> Ticket:
    """Free-text / email intake. First non-empty line is the subject."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    subject = lines[0][:200] if lines else ""
    body = " ".join(lines[1:]) if len(lines) > 1 else subject
    m = re.search(r"\b(?:POS|PRN|RTR|SSM|KSK)-\d{4}-[A-Z0-9]{1,3}\b", text)
    s = store_id or (re.search(r"\bSW-\d{4}\b", text).group(0) if re.search(r"\bSW-\d{4}\b", text) else None)
    return ticket_from_dict({"ticket_id": ticket_id, "subject": subject, "description": body,
                             "asset_id": m.group(0) if m else None, "store_id": s})


def load_ticket_file(path: Path, strip_status: bool = True) -> Ticket:
    p = Path(path)
    if p.suffix.lower() == ".json":
        return ticket_from_dict(json.loads(p.read_text(encoding="utf-8")), strip_status)
    return ticket_from_text(p.read_text(encoding="utf-8"), ticket_id=p.stem)


def load_ticket_dir(path: Path, strip_status: bool = True) -> list:
    return [load_ticket_file(p, strip_status) for p in sorted(Path(path).glob("*.json"))]


def load_ticket_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [ticket_from_dict(r, True) for r in csv.DictReader(f)]


def load_any(path: Path) -> list:
    p = Path(path)
    if p.is_dir():
        return load_ticket_dir(p)
    if p.suffix.lower() == ".csv":
        return load_ticket_csv(p)
    if p.suffix.lower() == ".jsonl":
        return [ticket_from_dict(json.loads(l), True) for l in p.read_text().splitlines() if l.strip()]
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        return [ticket_from_dict(x, True) for x in (data if isinstance(data, list) else [data])]
    return [load_ticket_file(p)]


def load_history_corpus(data_root: Path) -> list:
    """Every ticket JSON we can find in the dataset (used only as *history* for recurrence lookups)."""
    out = []
    for sub in ("raw", "train", "test"):
        d = Path(data_root) / "tickets" / sub
        if d.is_dir():
            out += [ticket_from_dict(json.loads(p.read_text(encoding="utf-8")), True) for p in sorted(d.glob("*.json"))]
    seen, uniq = set(), []
    for t in out:
        if t.ticket_id not in seen:
            seen.add(t.ticket_id)
            uniq.append(t)
    return uniq
