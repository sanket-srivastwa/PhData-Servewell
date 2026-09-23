"""Supervisor: a fixed, testable workflow (not a free-roaming agent).

    intake -> triage/enrich -> retrieve -> policy decision -> resolve (extractive | LLM)
           -> guardrails -> communicate -> propose/execute actions (HITL gate) -> audit

Each stage appends to a trace (what ran, on what, how long) - the raw material for observability.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .. import config
from ..ingest import load_history_corpus
from ..intel_platform.itsm import AuditLog, ITSMSimulator
from ..intel_platform.kb import KnowledgeBase
from ..intel_platform.structured import StructuredData
from ..llm import LLM, LLMError, NullLLM
from ..models import AgentResult, Decision, Enrichment, ProposedAction, Step, Ticket
from . import comms
from .guardrails import Guardrails
from .knowledge import Retrieval, Retriever
from .policy import EscalationModel, PolicyEngine
from .reader import TicketReader
from .resolver import Resolver
from .triage import TriageAgent


class SupportAgentSuite:
    def __init__(self, data_root: Optional[Path] = None, llm: Optional[LLM] = None,
                 audit_path: Optional[Path] = None, model_path: Optional[Path] = None,
                 prior_path: Optional[Path] = None, load_history: bool = True):
        self.root = Path(data_root or config.find_data_root())
        self.llm = llm or NullLLM()
        pp = prior_path or (config.MODELS_DIR / "kb_prior.json")
        self.kb = KnowledgeBase(self.root / "kb", pp)
        self.sd = StructuredData(self.root / "data")
        if load_history:
            self.sd.register_tickets(load_history_corpus(self.root))
        self.triage = TriageAgent(self.sd, self.kb)
        self.retriever = Retriever(self.kb)
        self.policy = PolicyEngine(self.retriever, EscalationModel(model_path))
        self.resolver = Resolver(self.kb, self.retriever, self.llm)
        self.reader = TicketReader(self.llm)
        self.guard = Guardrails(self.kb)
        self.audit = AuditLog(audit_path)
        self.itsm = ITSMSimulator(self.audit)

    # ------------------------------------------------------------------------------------------
    def set_llm(self, llm: Optional[LLM]) -> None:
        """Swap the language model at runtime (UI settings). None -> deterministic offline mode."""
        self.llm = llm or NullLLM()
        self.resolver.llm = self.llm
        self.reader.llm = self.llm

    def ingest_history(self, tickets: list) -> None:
        """Newly arrived tickets become history for recurrence / reference lookups."""
        self.sd.register_tickets(tickets)

    def process(self, t: Ticket, approve: bool = False, now: Optional[datetime] = None, use_llm: Optional[bool] = None) -> AgentResult:
        """use_llm: None -> use the model if one is configured; False -> force the deterministic offline path for this run
        (used for side-by-side comparisons); True -> use the model if configured."""
        t0 = time.time()
        trace: list = []
        llm_on = self.llm.available and (True if use_llm is None else bool(use_llm))
        usage_before = self.llm.snapshot()
        reader_report, writer_report = None, None

        def step(stage, agent, detail, started):
            trace.append({"stage": stage, "agent": agent, "detail": detail, "ms": int((time.time() - started) * 1000)})

        s = time.time()
        self.audit.write(t.ticket_id, "intake", {"missing_fields": t.raw_missing_fields, "leak_guard_dropped": t.leak_guard_dropped,
                                                  "kb_hash": self.kb.kb_hash, "mode": self.llm.name if llm_on else "offline-extractive"})
        step("intake", "supervisor", f"ticket accepted; dropped answer fields: {t.leak_guard_dropped or 'none'}; missing: {t.raw_missing_fields or 'none'}", s)

        # 1. triage & enrichment (tool calls)
        s = time.time()
        e = self.triage.run(t, now)
        warn = [f.code for f in e.findings if f.severity in ("warn", "block")]
        step("triage_enrich", "triage_agent",
             f"tools: get_store, get_asset, version_matrix, prior_tickets, sla_for -> domain={e.domain}, "
             f"priority {t.priority}->{e.suggested_priority}, flags={warn or 'none'}", s)

        # 1b. LLM ticket reader (optional): structured reading of messy text, merged with the rules under strict conditions
        if llm_on:
            s = time.time()
            try:
                raw = self.reader.read(t, e)
                reader_report = self.reader.merge(t, e, raw)
                step("llm_read", "ticket_reader", f"{self.llm.name}: applied={reader_report['applied'] or 'nothing'}; flags={reader_report['flags'] or 'none'}", s)
            except (LLMError, ValueError) as exc:          # any model failure -> carry on with the rules alone
                reader_report = {"used": False, "error": str(exc)[:300]}
                step("llm_error", "ticket_reader", f"reader failed, rules-only triage kept: {str(exc)[:160]}", s)

        # 2. retrieval
        s = time.time()
        r = self.retriever.retrieve(t, e)
        step("retrieve", "knowledge_agent",
             f"primary={r.primary} ({r.primary_score:.2f}) secondary={r.secondary}; sufficient={r.sufficient}; docs={[d for d, _, _ in r.docs][:4]}", s)

        # 3. policy
        s = time.time()
        d = self.policy.decide(t, e, r)
        if d.route == "l2_escalation":
            r = self.retriever.retrieve(t, e, escalating=True)          # add the escalation SOP as evidence
        if reader_report and reader_report.get("raw"):
            self.reader.second_opinion(d, reader_report)          # advisory only: can add human review, never changes the route
        if e.llm_flags:
            d.human_review = True
            d.review_reasons += [f"LLM reader: {x}" for x in e.llm_flags]
        step("decide", "policy_engine", f"route={d.route} confidence={d.confidence} triggers={d.hard_triggers or 'none'}; {d.reasons[0]}", s)

        # 4. resolution guidance
        s = time.time()
        allowed = {c.id for c in self.kb.chunks if c.doc in set(r.doc_paths())} | {cid for cid in self.kb.by_id if cid.startswith("system-specs/version-matrix.csv#")}
        steps, skipped = self.resolver.extractive_steps(t, e, r, d)
        llm_used, llm_msg, llm_l2, dropped = False, None, None, []
        if llm_on and r.sufficient and d.route in ("l1_guided", "l2_escalation", "needs_clarification"):
            writer_report = {"used": False, "steps_written": 0, "steps_kept": 0, "error": None}
            try:
                out = self.resolver.llm_steps(t, e, r, d)
                if out:
                    cand = self.resolver.steps_from_llm(out)
                    kept, dropped = self.guard.check_steps(cand, allowed, llm_generated=True)
                    writer_report.update(steps_written=len(cand), steps_kept=len(kept), dropped=dropped)
                    if len(kept) >= min(config.MIN_GROUNDED_STEPS, max(1, len(cand))) and kept:
                        steps, llm_used, llm_msg = kept, True, out.get("requester_message")
                        for i, st in enumerate(steps, 1):
                            st.n = i
                        l2s = out.get("l2_summary")
                        if isinstance(l2s, str) and 10 <= len(l2s) <= 700 and self.guard.check_message(l2s, t, e, audience="internal")["passed"]:
                            llm_l2 = l2s.strip()
                    writer_report["used"] = llm_used
            except (LLMError, ValueError) as exc:  # LLM failure must never break the run - fall back to extractive
                writer_report["error"] = str(exc)[:300]
                trace.append({"stage": "llm_error", "agent": "knowledge_agent", "detail": f"{type(exc).__name__}: {exc}", "ms": 0})
        step("resolve", "knowledge_agent",
             f"{'LLM-written (' + self.llm.name + ')' if llm_used else 'extractive'} steps={len(steps)}; skipped(already tried)={len(skipped)}; LLM-dropped={len(dropped)}", s)

        # 5. guardrail: grounding (runs on extractive too - same verifier)
        s = time.time()
        if not llm_used:
            steps, dropped_ext = self.guard.check_steps(steps, allowed, llm_generated=False)
            dropped = dropped + dropped_ext
        if e.get("known_issue") and r.sufficient and d.route in ("l1_guided", "l2_escalation") and e.version_info:
            key = f"system-specs/version-matrix.csv#{e.version_info['key']}"
            ch = self.kb.by_id.get(key)
            if ch:
                kn = Step(n=len(steps) + 1, title="Known issue for this version", text=ch.text, source_id=key,
                          source_label=ch.label, kind="note")
                ok, sc, _ = self.guard.check_step(kn.text, [key], allowed, 0.6)
                if ok:
                    steps.append(kn)
        # downgrade if L1 guidance can't be grounded well enough
        guided = [x for x in steps if x.kind != "note"]
        if d.route == "l1_guided" and len(guided) < config.MIN_GROUNDED_STEPS:
            d.route = "l2_escalation"
            d.reasons.insert(0, f"Only {len(guided)} grounded step(s) survived checks - escalating rather than sending thin guidance")
            d.human_review = True
            d.review_reasons.append("insufficient grounded guidance")
            d.autonomy = "A3_prepare_handoff (human confirms P1/P2)"
            r = self.retriever.retrieve(t, e, escalating=True)
            steps = [x for x in steps if x.kind in ("immediate", "note")][:2]
        for i, x in enumerate(steps, 1):
            x.n = i
        step("guardrail_grounding", "guardrails", f"{len([x for x in steps if x.grounded])}/{len(steps)} steps grounded; dropped={[x['reason'][:60] for x in dropped]}", s)

        # 6. communicate
        s = time.time()
        questions = comms.build_questions(t, e, d)
        msg = comms.requester_message(t, e, d, steps, questions, r)
        msg_check = self.guard.check_message(msg, t, e)
        if llm_used and isinstance(llm_msg, str) and llm_msg.strip():
            llm_msg = llm_msg.replace("{{NAME}}", comms._first(t.submitted_by))
            m2 = self.guard.check_message(llm_msg, t, e)
            if m2["passed"]:
                msg, msg_check = llm_msg, m2
                writer_report["message_used"] = True
            else:
                writer_report["message_rejected"] = m2["issues"]
        if not msg_check["passed"]:                                     # template itself failed policy -> never send
            d.human_review = True
            d.review_reasons.append("message policy: " + "; ".join(msg_check["issues"]))
        hand = comms.l2_handoff(t, e, d, steps, r) if d.route == "l2_escalation" else None
        if hand:
            hand["runbook_escalation_criteria_met"] = self.retriever.escalation_hits(r.primary, t, e) if r.primary else []
            if llm_l2:
                hand["ai_summary"] = llm_l2
        actions = self._propose(t, e, d, steps, r, msg, hand, msg_ok=msg_check["passed"])
        act_check = self.guard.check_actions(actions)
        report = self.guard.report(steps_in=len(steps) + len(dropped), steps_kept=len(steps), dropped=dropped, msg_check=msg_check,
                                   act_check=act_check, t=t, e=e, sufficient=r.sufficient, llm_generated=llm_used)
        note = comms.ticket_note(t, e, d, steps, r, report, self.kb.kb_hash, skipped)
        step("communicate", "action_agent", f"reply {len(msg)} chars; questions={len(questions)}; handoff={'yes' if hand else 'no'}; message_check={'ok' if msg_check['passed'] else 'FAIL'}", s)

        # 7. HITL gate + simulated execution
        s = time.time()
        ok_guard = True                                # message policy failures are handled by holding the reply (see _propose)
        for a in actions:
            self.itsm.execute(t.ticket_id, a, approved=approve, guardrails_passed=ok_guard)
        pend = [a.action for a in actions if a.status == "proposed"]
        step("act", "action_agent", f"executed={[a.action for a in actions if 'simulated' in a.status]}; awaiting approval={pend}", s)
        self.audit.write(t.ticket_id, "decision", {"route": d.route, "confidence": d.confidence, "human_review": d.human_review,
                                                    "triggers": d.hard_triggers, "steps": [x.source_id for x in steps]})

        return AgentResult(
            ticket_id=t.ticket_id, decision=d, enrichment=e,
            retrieved=[{"doc": doc, "score": round(sc, 2), "why": why} for doc, sc, why in r.docs],
            steps=steps, questions=questions, requester_message=msg, ticket_note=note, l2_handoff=hand,
            actions=actions, guardrails=report, trace=trace, mode=self.llm.name if llm_on else "offline-extractive",
            elapsed_ms=int((time.time() - t0) * 1000),
            llm=({"provider": self.llm.provider, "model": self.llm.name, "reader": reader_report, "writer": writer_report,
                  "usage": self.llm.usage_since(usage_before)} if llm_on else None))

    # ------------------------------------------------------------------------------------------
    def _propose(self, t: Ticket, e: Enrichment, d: Decision, steps: list, r: Retrieval, msg: str, hand, msg_ok: bool = True) -> list:
        A = self.itsm.propose
        acts = []
        held = d.human_review or not msg_ok       # human review / failed message policy => even low-risk outbound messages wait
        reply_action = {"non_it": "redirect_non_it", "needs_clarification": "request_clarification"}.get(d.route, "post_requester_reply")
        a = A(reply_action, t.ticket_id, {"text": msg}, "Send message to requester")
        if held:
            a.requires_approval = True
            a.rationale += " (held: " + ("; ".join(d.review_reasons) or "message policy failed") + ")"
        acts.append(a)
        acts.append(A("post_internal_note", t.ticket_id, {"text": "see ticket_note"}, "Record triage summary for L1/L2"))
        if e.suggested_priority != t.priority:
            acts.append(A("suggest_priority_change", t.ticket_id, {"new_priority": e.suggested_priority, "reasons": e.priority_reasons},
                          "SOP override rules: " + "; ".join(e.priority_reasons)))
        if hand:
            acts.append(A("escalate_to_l2", t.ticket_id, {"to": hand["to"], "subject": hand["subject"]}, "L2 handoff per SOP-IT-L1-002 (human confirms)"))
        # automations: only with a verified asset, only if grounded in a retrieved step, never if already tried
        asset = e.asset if (e.asset and not e.has("asset_domain_mismatch")) else e.resolved_asset
        if asset and d.route == "l1_guided":
            body = " ".join(s.text.lower() for s in steps)
            tried = set(e.tried_steps)
            acts.append(A("run_diagnostic_ping", asset["asset_id"], {}, "Read-only reachability check"))
            if "print spooler" in body and "spooler_restart" not in tried:
                acts.append(A("restart_print_spooler", asset["asset_id"], {"store": t.store_id}, "Runbook step: restart Print Spooler"))
            if e.domain == "POS" and "cache" in body and "clear_cache" not in tried:
                acts.append(A("clear_pos_cache", asset["asset_id"], {}, "Runbook step: clear POS cache"))
            if e.domain == "POS" and re.search(r"restart (the )?(serve\w*\s*)?(pos )?(application|app)|close (serve|food)\w* pos", body):
                acts.append(A("remote_restart_pos_app", asset["asset_id"], {}, "Runbook step: restart POS application"))
        return acts
