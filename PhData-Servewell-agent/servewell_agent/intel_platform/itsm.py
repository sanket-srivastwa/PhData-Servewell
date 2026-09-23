"""Simulated ITSM tool endpoints + audit log.

Every state-changing capability the agents can use is declared in ACTION_CATALOG with a risk class.
The executor enforces three rules regardless of what an agent (or LLM) asks for:
  1. only catalogued actions exist (allow-list; unknown action => blocked);
  2. anything not `read_only` needs approval unless the catalog marks it auto-approvable AND the
     run passed its guardrails;
  3. every attempt (allowed or blocked) is written to a hash-chained audit log.

In the POC nothing leaves the process: "execution" mutates an in-memory ticket board and returns a
clearly labelled SIMULATED result. We do NOT fabricate device telemetry: automations that would need
live device data report "no telemetry source in POC".
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..models import ProposedAction

ACTION_CATALOG = {
    # action: (risk, auto_approvable, description)
    "post_internal_note":   ("reversible", True,  "Add an internal work note to the ticket"),
    "post_requester_reply": ("reversible", True,  "Send guided steps / questions to the requester (template-checked)"),
    "request_clarification": ("reversible", True, "Ask requester for missing mandatory information"),
    "redirect_non_it":      ("reversible", True,  "Redirect a non-IT request to the owning function (HR / Billing)"),
    "suggest_priority_change": ("state_changing", False, "Change ticket priority (SOP override rules)"),
    "escalate_to_l2":       ("state_changing", False, "Assign to L2 queue + notify it-l2@ (oncall@ for P1)"),
    "run_diagnostic_ping":  ("read_only", True,   "Remote ping / reachability check of the asset"),
    "restart_print_spooler": ("reversible", False, "Remote restart of the Windows print spooler"),
    "remote_restart_pos_app": ("reversible", False, "Remote restart of the POS application (not the OS)"),
    "clear_pos_cache":      ("reversible", False, "Clear local POS cache via management agent"),
    "remote_router_reboot": ("state_changing", False, "Remote reboot of the store router (store-wide impact)"),
}


class AuditLog:
    """Append-only JSONL with a hash chain (tamper-evident)."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.entries: list = []
        self._prev = "0" * 16
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, ticket_id: str, event: str, payload: dict) -> dict:
        body = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "ticket_id": ticket_id, "event": event, "payload": payload}
        digest = hashlib.sha256((self._prev + json.dumps(body, sort_keys=True, default=str)).encode()).hexdigest()[:16]
        entry = {**body, "prev": self._prev, "hash": digest}
        self._prev = digest
        self.entries.append(entry)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def verify(self) -> bool:
        prev = "0" * 16
        for e in self.entries:
            body = {k: e[k] for k in ("ts", "ticket_id", "event", "payload")}
            if hashlib.sha256((prev + json.dumps(body, sort_keys=True, default=str)).encode()).hexdigest()[:16] != e["hash"]:
                return False
            prev = e["hash"]
        return True


class ITSMSimulator:
    def __init__(self, audit: AuditLog):
        self.audit = audit
        self.board: dict = {}      # ticket_id -> {comments: [], assignment, status, priority}

    def _state(self, tid: str) -> dict:
        return self.board.setdefault(tid, {"comments": [], "assignment": "L1 Queue", "status": "Open", "notifications": []})

    def propose(self, action: str, target: str, params: dict, rationale: str = "") -> ProposedAction:
        if action not in ACTION_CATALOG:
            return ProposedAction(action, target, params, "unknown", True, "blocked", rationale,
                                  "BLOCKED: action not in allow-list")
        risk, auto_ok, _ = ACTION_CATALOG[action]
        return ProposedAction(action, target, params, risk, requires_approval=not auto_ok, rationale=rationale)

    def execute(self, ticket_id: str, act: ProposedAction, approved: bool = False, guardrails_passed: bool = True) -> ProposedAction:
        if act.status == "blocked":
            self.audit.write(ticket_id, "action_blocked", {"action": act.action, "why": act.result})
            return act
        if act.requires_approval and not approved:
            act.status = "proposed"
            act.result = "PENDING HUMAN APPROVAL - not executed"
            self.audit.write(ticket_id, "action_pending_approval", {"action": act.action, "target": act.target, "params": act.params})
            return act
        if not act.requires_approval and not guardrails_passed:
            act.status = "blocked"
            act.result = "BLOCKED: guardrail check failed"
            self.audit.write(ticket_id, "action_blocked", {"action": act.action, "why": "guardrails"})
            return act
        st = self._state(ticket_id)
        act.status = "approved_simulated" if act.requires_approval else "auto_executed_simulated"
        if act.action in ("post_internal_note", "post_requester_reply", "request_clarification", "redirect_non_it"):
            st["comments"].append({"action": act.action, "text": act.params.get("text", "")[:2000]})
            act.result = "SIMULATED: comment posted"
        elif act.action == "escalate_to_l2":
            st["assignment"] = "L2 Support Team"
            st["status"] = "Escalated"
            st["notifications"].append({"to": act.params.get("to", []), "subject": act.params.get("subject", "")})
            act.result = "SIMULATED: assigned to L2 queue; notifications queued to " + ", ".join(act.params.get("to", []))
        elif act.action == "suggest_priority_change":
            st["priority"] = act.params.get("new_priority")
            act.result = f"SIMULATED: priority set to {act.params.get('new_priority')}"
        elif act.action == "run_diagnostic_ping":
            act.result = "SIMULATED: no telemetry source in POC (would ping asset and return latency/loss)"
        else:
            act.result = f"SIMULATED: {act.action} executed on {act.target} (no real device in POC)"
        self.audit.write(ticket_id, "action_executed", {"action": act.action, "target": act.target, "status": act.status})
        return act
