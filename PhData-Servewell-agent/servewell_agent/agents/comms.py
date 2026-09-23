"""Agent 4 (Action & Communication) - turns decisions into words and handoffs.

Templates are deliberate: the wording of anything customer-facing is policy, so the default path is
a reviewed template with grounded steps dropped in. (In LLM mode the model can propose the message;
it must then pass the same message guardrail or the template is used instead.)
"""
from __future__ import annotations

import re
from typing import Optional

from ..models import Decision, Enrichment, Step, Ticket
from .knowledge import Retrieval

# SOP-IT-L1-002: "L2 Response SLAs" (initial response) - used to set customer expectations
L2_INITIAL_RESPONSE = {"P1": "15 minutes", "P2": "1 hour", "P3": "3 hours", "P4": "8 business hours"}
L2_MAILBOX, ONCALL_MAILBOX = "it-l2@servewell.in", "oncall@servewell.in"

DOMAIN_QUESTIONS = {
    "Wi-Fi / Network": "Are the lights on the router on, and can other devices in the store (POS, phones, kitchen display) connect?",
    "POS": "Which terminal is affected (T1, T2, ...) and what exactly is on its screen right now?",
    "Printers": "Is the printer powered on with a steady light, and what does the POS show for it (offline, error, nothing)?",
    "Soft Serve": "Which machine is it, and what does its display show (please read out any code)?",
    "Kiosks": "Which kiosk is affected and what does the screen show?",
    "Online Orders": "Which platform is affected (Zomato, Swiggy, our portal) and are orders missing, duplicated or delayed?",
}


def _first(name: Optional[str]) -> str:
    return (name or "there").split()[0]


def build_questions(t: Ticket, e: Enrichment, d: Decision) -> list:
    q = []
    if "asset_id" in e.missing_mandatory:
        cand = e.resolved_asset
        if t.asset_id and e.has("asset_not_in_cmdb"):
            base = f"The asset ID on the ticket ({t.asset_id}) doesn't exist in our records."
        else:
            base = "The ticket has no asset ID."
        if cand:
            q.append(f"{base} Is the device {cand['asset_id']} ({cand['asset_type']})? If not, please read the asset tag label on the device.")
        else:
            q.append(f"{base} Please read the asset tag label on the device (format like POS-####-T#).")
    if "error_code" in e.missing_mandatory:
        q.append("What exact error message or code is shown on the screen? A photo is perfect.")
    if d.route == "needs_clarification" and e.vague:
        q.append(DOMAIN_QUESTIONS.get(e.domain, "What exactly is happening, and on which device?"))
    if "affected_devices" in e.missing_mandatory:
        q.append("Is it one device or several? If several, which ones?")
    if "steps_already_tried" in e.missing_mandatory and d.route == "needs_clarification":
        q.append("What have you already tried, and what happened?")
    if d.route == "needs_clarification" and not re.search(r"\b(since|around|at)\b.{0,12}\d|morning|today|yesterday", t.text.lower()):
        q.append("When did it start?")
    seen, out = set(), []
    for x in q:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out[:4]


def _pretty(text: str, limit_lines: int) -> str:
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"^(\s*)\d+\.\s", r"\1- ", ln)          # nested runbook numbering -> bullets (avoids "1. Step 2: 1. ...")
        out.append("   " + ln.strip() if not ln.startswith(" ") else "   " + ln)
    if len(out) > limit_lines:
        out = out[:limit_lines] + ["   ..."]
    return "\n".join(out)


def _steps_block(steps: list, limit_lines: int = 5, max_steps: int = 3) -> str:
    out, n = [], 0
    for s in steps:
        if s.kind == "note":
            continue
        n += 1
        if n > max_steps:
            out.append(f"({len([x for x in steps if x.kind != 'note']) - max_steps} more step(s) are on the ticket.)")
            break
        title = re.sub(r"^(Step|Resolution(?: Step)?)\s*[A-Z0-9]+:\s*", "", s.title)
        out.append(f"{n}. {title}\n{_pretty(s.text, limit_lines)}\n   (source: {s.source_label})")
    return "\n\n".join(out)


def requester_message(t: Ticket, e: Enrichment, d: Decision, steps: list, questions: list, r: Retrieval) -> str:
    name = _first(t.submitted_by)
    p = e.suggested_priority
    ack = ""
    if e.dissatisfied:
        ack = ("I'm sorry this has been so disruptive - I understand how frustrating it is when equipment keeps you from serving customers. "
               "You shouldn't have to repeat yourself, so I've pulled together everything from the ticket. ")
    if d.route == "non_it":
        rm = (e.store or {}).get("region_manager")
        who = f"your regional manager ({rm})" if rm else "your regional manager"
        return (f"Hi {name},\n\nThanks for getting in touch. {ack}This looks like a payroll / HR matter, which IT support can't resolve or "
                f"access. Please raise it with {who}, who can point you to the right team. I've noted this on the ticket so nothing is lost.\n\n"
                "If there is also a genuine IT fault (a device or system not working), reply here with the details and we'll pick that up straight away.")
    lines = [f"Hi {name},", ""]
    if d.route == "needs_clarification":
        lines.append(f"Thanks for reporting this. {ack}To get the right help to you quickly, I need a few details:")
        lines += [f"  - {q}" for q in questions]
        if steps:
            lines += ["", "In the meantime, this safe first check is worth doing:", "", _steps_block(steps)]
        lines += ["", "Reply on this ticket with those details and we'll continue straight away."]
        return "\n".join(lines)
    if d.route == "l2_escalation":
        lines.append(f"Thanks for the detail on this. {ack}I'm escalating your ticket to our L2 specialist team with everything collected so far.")
        lines.append(f"You can expect their first response within {L2_INITIAL_RESPONSE.get(p, 'the SOP window')} (priority {p}).")
        if e.has("unknown_system"):
            lines.append("The system involved isn't one our standard runbooks cover, so a specialist needs to look at it.")
        if steps:
            lines += ["", "While you wait, these safe checks may help (skip any you've already done):", "", _steps_block(steps)]
        if questions:
            lines += ["", "If you can, also send:"] + [f"  - {q}" for q in questions]
        if e.non_it == "partial":
            lines += ["", "Separately: the payroll / salary question in your ticket is an HR matter - please raise it with your regional manager."]
        return "\n".join(lines)
    # l1_guided
    lines.append(f"Thanks for reporting this. {ack}Let's get it working again. Please try these steps in order and tell me what happens after each:")
    lines += ["", _steps_block(steps)]
    lines += ["", "If it isn't fixed after these, reply here and I'll escalate to our L2 team with what we've tried so you don't start over."]
    if e.non_it == "partial":
        lines += ["", "Separately: the payroll / salary question in your ticket is an HR matter - please raise it with your regional manager."]
    return "\n".join(lines)


def ticket_note(t: Ticket, e: Enrichment, d: Decision, steps: list, r: Retrieval, guard: dict, mode: str, skipped: list) -> str:
    L = [f"[Agent triage summary - {t.ticket_id}]",
         f"Route: {d.route} (confidence {d.confidence}; autonomy {d.autonomy})",
         "Why: " + " | ".join(d.reasons)]
    if d.human_review:
        L.append("HUMAN REVIEW REQUIRED: " + "; ".join(d.review_reasons))
    L.append(f"Effective domain: {e.domain}" + (f" (filed as {e.claimed_domain})" if e.claimed_domain and e.claimed_domain != e.domain else ""))
    if e.sla:
        L.append(f"Priority: submitted {t.priority} -> suggested {e.suggested_priority}"
                 + (f" [{'; '.join(e.priority_reasons)}]" if e.priority_reasons else "")
                 + f". SLA first response due {e.sla['first_response_due']}, resolution due {e.sla['resolution_due']}.")
    for f in e.findings:
        if f.severity in ("warn", "block") or f.code in ("known_issue", "out_of_warranty", "priority_override"):
            L.append(f"  - [{f.severity}] {f.code}: {f.message}")
    if e.tried_steps:
        L.append("Already tried (per ticket/history): " + ", ".join(e.tried_steps))
    if r.gap:
        L.append(f"KB gap: no runbook is named for '{t.subcategory}' - add one (backlog item).")
    if r.primary:
        L.append(f"Grounded on: {r.primary}" + (f", {r.secondary}" if r.secondary else "") + f" (KB {mode})")
    if skipped:
        L.append(f"Skipped {len(skipped)} runbook step(s) already attempted.")
    L.append("Guardrails: " + "; ".join(f"{c['name']}={'ok' if c['passed'] else 'FAIL'}" for c in guard["checks"]))
    return "\n".join(L)


def l2_handoff(t: Ticket, e: Enrichment, d: Decision, steps: list, r: Retrieval) -> dict:
    """SOP-IT-L1-002 escalation email template, pre-filled; missing mandatory fields are flagged, not invented."""
    p = e.suggested_priority
    asset = e.asset if (e.asset and not e.has("asset_domain_mismatch")) else e.resolved_asset
    tried = e.tried_steps[:]
    for h in t.ticket_history:
        tried.append(f"L1 note: {h.get('note', '')[:140]}")
    verified = {
        "Asset ID": (t.asset_id if (e.asset and not e.has("asset_not_in_cmdb")) else "NOT VERIFIED"),
        "Error Code": (", ".join(e.error_codes) if e.error_codes else "NOT DOCUMENTED"),
        "Steps Tried": ("documented" if tried else "NOT DOCUMENTED"),
        "Store Impact": e.impact,
    }
    to = [L2_MAILBOX] + ([ONCALL_MAILBOX] if p == "P1" else [])
    return {
        "to": to,
        "subject": f"[ESCALATION] Ticket #{t.ticket_id} - {t.store_name or t.store_id} - {p}",
        "priority": p,
        "store": t.store_name or t.store_id,
        "asset": f"{asset['asset_id']} ({asset['asset_type']}, {asset['system_version']})" if asset else "unknown",
        "issue": e.issue_core,
        "error_codes": e.error_codes,
        "steps_already_tried": tried,
        "store_impact": e.impact,
        "mandatory_info_verified": verified,
        "missing_mandatory": e.missing_mandatory,
        "why_escalated": d.hard_triggers or d.reasons[:2],
        "runbook_used": r.primary,
        "runbook_escalation_criteria_met": [],
        "recurrence": {"prior_tickets": e.prior_tickets[-3:], "count": e.recurrence_count},
        "customer_contact": {"name": t.submitted_by, "phone": t.contact_phone, "preferred_method": "not recorded"},
        "manager_route": bool(e.dissatisfied),
        "sla_note": f"L2 initial response target per SOP: {L2_INITIAL_RESPONSE.get(p)}",
    }
