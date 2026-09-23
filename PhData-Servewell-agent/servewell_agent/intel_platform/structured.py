"""Structured-data tools (simulated ITSM/CMDB/SLA access on the Intelligence Platform).

The agent *calls these as tools* rather than reasoning about facts from memory: an asset either
exists in the CMDB or it does not, an SLA is a lookup, recurrence is a count. Anything that has a
system of record is a tool call; the LLM (when enabled) is only used for language understanding.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..models import Ticket

DOMAIN_ASSET_TYPE = {
    "POS": "POS Terminal", "Printers": "Receipt Printer", "Wi-Fi / Network": "Network Router",
    "Soft Serve": "Soft Serve Machine", "Kiosks": "Self-Order Kiosk",
}


def _read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class StructuredData:
    def __init__(self, data_dir):
        d = Path(data_dir)
        self.stores = {r["store_id"]: r for r in _read_csv(d / "stores.csv")}
        self.assets = {r["asset_id"]: r for r in _read_csv(d / "assets.csv")}
        self.sla_rows = _read_csv(d / "sla_matrix.csv")
        self.tickets: dict = {}            # ticket store used for history / recurrence lookups

    # ---- registry of known tickets (history) -------------------------------------------------
    def register_tickets(self, tickets: list) -> None:
        for t in tickets:
            self.tickets[t.ticket_id] = t

    # ---- tools -------------------------------------------------------------------------------
    def get_store(self, store_id: Optional[str]) -> Optional[dict]:
        return self.stores.get(store_id) if store_id else None

    def get_asset(self, asset_id: Optional[str]) -> Optional[dict]:
        return self.assets.get(asset_id) if asset_id else None

    def assets_for_store(self, store_id: Optional[str], asset_type: Optional[str] = None) -> list:
        return [a for a in self.assets.values()
                if a["store_id"] == store_id and (asset_type is None or a["asset_type"] == asset_type)]

    def resolve_asset(self, store_id: Optional[str], domain: str) -> Optional[dict]:
        """Best-guess asset for (store, domain) when the ticket's asset is wrong/missing.
        Only returns a value when it is unambiguous (exactly one candidate)."""
        typ = DOMAIN_ASSET_TYPE.get(domain)
        if not typ:
            return None
        cands = self.assets_for_store(store_id, typ)
        return cands[0] if len(cands) == 1 else None

    def sla_for(self, priority: str, category: str) -> Optional[dict]:
        for r in self.sla_rows:
            if r["priority"] == priority and r["category"] == category:
                return {k: (int(v) if v.isdigit() else v) for k, v in r.items()}
        for r in self.sla_rows:                      # fall back to "General"
            if r["priority"] == priority and r["category"] == "General":
                return {k: (int(v) if v.isdigit() else v) for k, v in r.items()}
        return None

    def lookup_ticket(self, ticket_id: str) -> Optional[Ticket]:
        return self.tickets.get(ticket_id)

    def prior_tickets(self, t: Ticket, days: int = 30) -> list:
        """Earlier tickets from the same store (same asset or same subcategory) inside the window."""
        if not t.date_opened:
            return []
        lo = t.date_opened - timedelta(days=days)
        out = []
        for o in self.tickets.values():
            if o.ticket_id == t.ticket_id or not o.date_opened or o.store_id != t.store_id:
                continue
            if lo <= o.date_opened < t.date_opened and (
                    (t.asset_id and o.asset_id == t.asset_id) or (t.subcategory and o.subcategory == t.subcategory)):
                out.append({"ticket_id": o.ticket_id, "date": o.date_opened.isoformat(),
                            "subcategory": o.subcategory, "asset_id": o.asset_id})
        return sorted(out, key=lambda x: x["date"])

    @staticmethod
    def warranty_status(asset: Optional[dict], as_of: Optional[datetime]) -> Optional[str]:
        if not asset or not asset.get("warranty_expiry") or not as_of:
            return None
        try:
            exp = datetime.strptime(asset["warranty_expiry"], "%Y-%m-%d")
        except ValueError:
            return None
        as_of_naive = as_of.replace(tzinfo=None)
        return "in_warranty" if exp >= as_of_naive else "out_of_warranty"
