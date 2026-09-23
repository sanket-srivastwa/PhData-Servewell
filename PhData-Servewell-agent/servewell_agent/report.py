"""Human-readable rendering of an AgentResult - written for the live demo: each section says what the
agent did and *why*, in an order a non-technical stakeholder can follow."""
from __future__ import annotations

import json
import os
import sys
import textwrap

from .models import AgentResult, Ticket

_C = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _C else s


def h(title: str) -> str:
    return "\n" + _c("1;36", f"== {title} " + "=" * max(0, 74 - len(title)))


def wrap(s: str, indent: int = 2, width: int = 100) -> str:
    return "\n".join(textwrap.fill(p, width=width, initial_indent=" " * indent, subsequent_indent=" " * indent) if p.strip() else "" for p in s.splitlines())


ROUTE_TEXT = {
    "l1_guided": "L1 GUIDED - agent guides the store through runbook steps",
    "l2_escalation": "L2 ESCALATION - hand to specialists with a complete package",
    "needs_clarification": "NEEDS CLARIFICATION - ask targeted questions before acting",
    "non_it": "NOT AN IT ISSUE - redirect to the owning function",
    "l1_self_service": "L1 SELF-SERVICE",
}


def render(res: AgentResult, t: Ticket, show_trace: bool = True, show_ticket: bool = True) -> str:
    e, d = res.enrichment, res.decision
    out = []
    if show_ticket:
        out.append(h(f"INCOMING TICKET {t.ticket_id}"))
        out.append(f"  {t.store_name or t.store_id} | filed as {t.category}/{t.subcategory} | submitted priority {t.priority} | asset {t.asset_id} | {t.system_version}")
        out.append(f"  Subject: {t.subject}")
        out.append(wrap(t.description[:700] + ("..." if len(t.description) > 700 else ""), 2))
        if t.ticket_history:
            out.append(f"  L1 history: {len(t.ticket_history)} note(s)")
    out.append(h("1. TRIAGE & ENRICHMENT  (triage agent - tool calls to CMDB, SLA, history)"))
    out.append(f"  Effective domain: {e.domain}" + (f"   (ticket was filed under '{e.claimed_domain}')" if e.claimed_domain and e.claimed_domain != e.domain else ""))
    if e.sla:
        out.append(f"  Priority: {t.priority} -> suggested {e.suggested_priority}" + (f"   because: {'; '.join(e.priority_reasons)}" if e.priority_reasons else ""))
        out.append(f"  SLA clock: first response due {e.sla['first_response_due']}, resolution due {e.sla['resolution_due']}")
    if e.asset:
        a = e.asset
        out.append(f"  CMDB asset: {a['asset_id']} - {a['asset_type']} {a['system_version']} (warranty to {a['warranty_expiry']})")
    if e.error_codes:
        out.append(f"  Error codes: {', '.join(e.error_codes)}")
    if e.tried_steps:
        out.append(f"  Already tried: {', '.join(e.tried_steps)}")
    if e.issue_core and e.dissatisfied:
        out.append("  Technical problem (emotion stripped): " + e.issue_core[:240])
    flags = [f for f in e.findings if f.severity in ("warn", "block") or f.code in ("known_issue", "out_of_warranty", "priority_override")]
    for f in flags:
        tag = {"block": _c("1;31", "BLOCK"), "warn": _c("1;33", "WARN "), "info": "info "}[f.severity]
        out.append(f"  [{tag}] {f.code}: {f.message}")
    if not flags:
        out.append("  No data-quality or messy-ticket flags.")

    out.append(h("2. RETRIEVAL  (knowledge agent - scoped by domain + system version, then keyword search)"))
    for row in res.retrieved[:5]:
        out.append(f"  {row['score']:.2f}  {row['doc']:52s} {row['why']}")

    out.append(h("3. DECISION  (policy engine - SOP rules + explainable model; the LLM does not decide)"))
    out.append("  " + _c("1;32", ROUTE_TEXT.get(d.route, d.route)))
    out.append(f"  confidence {d.confidence} | escalation propensity {d.escalation_score} | autonomy {d.autonomy}")
    for r in d.reasons:
        out.append(f"   - {r}")
    if d.human_review:
        out.append("  " + _c("1;33", "HUMAN REVIEW: ") + "; ".join(d.review_reasons))

    out.append(h(f"4. GROUNDED GUIDANCE  ({res.mode})"))
    if res.steps:
        for s in res.steps:
            g = "grounded" if s.grounded else "NOT GROUNDED"
            out.append(f"  {s.n}. [{s.kind}] {s.title}  ({g} {s.grounding_score:.0%})")
            out.append(f"       source: {s.source_label}")
    else:
        out.append("  No runbook steps issued for this ticket (see decision above).")

    out.append(h("5. GUARDRAIL REPORT"))
    for c in res.guardrails["checks"]:
        out.append(f"  [{'PASS' if c['passed'] else _c('1;31', 'FAIL')}] {c['name']:22s} {c['detail'][:110]}")

    out.append(h("6. ACTIONS  (allow-listed; approval gate for anything that changes state)"))
    for a in res.actions:
        out.append(f"  {a.action:24s} risk={a.risk:14s} -> {a.status}" + (f"  | {a.result}" if a.result else ""))

    out.append(h("7. MESSAGE TO REQUESTER"))
    out.append(res.requester_message)
    if res.l2_handoff:
        out.append(h("8. L2 HANDOFF PACKAGE  (SOP-IT-L1-002 template, pre-filled)"))
        out.append(json.dumps({k: v for k, v in res.l2_handoff.items() if k not in ("customer_contact",)}, indent=2, ensure_ascii=False))
    out.append(h("INTERNAL TICKET NOTE"))
    out.append(res.ticket_note)
    if show_trace:
        out.append(h(f"TRACE  ({res.elapsed_ms} ms total)"))
        for tr in res.trace:
            out.append(f"  {tr['ms']:>4} ms  {tr['stage']:20s} {tr['agent']:16s} {tr['detail'][:120]}")
    return "\n".join(out)


def summary_row(res: AgentResult) -> str:
    d = res.decision
    return f"{res.ticket_id:10s} {d.route:20s} conf={d.confidence:<5} P={res.enrichment.suggested_priority} steps={len(res.steps):<2} review={'Y' if d.human_review else 'n'} flags={[f.code for f in res.enrichment.findings if f.severity in ('warn','block')]}"
