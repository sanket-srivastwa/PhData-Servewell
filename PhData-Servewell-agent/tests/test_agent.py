"""Behavioural tests. Run:  python -m unittest discover -s tests -v   (or: pytest tests)

Needs the dataset (set SERVEWELL_DATA or place it next to the project); tests skip otherwise.
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from servewell_agent import config  # noqa: E402
from servewell_agent.agents.orchestrator import SupportAgentSuite  # noqa: E402
from servewell_agent.ingest import load_ticket_file, ticket_from_dict, ticket_from_text  # noqa: E402
from servewell_agent.intel_platform.itsm import AuditLog, ITSMSimulator  # noqa: E402
from servewell_agent.llm import FakeLLM  # noqa: E402

try:
    ROOT = config.find_data_root()
except FileNotFoundError:
    ROOT = None

EX = Path(__file__).resolve().parent.parent / "examples"


def _ticket(tid):
    return load_ticket_file(ROOT / "tickets" / "train" / f"{tid}.json")


@unittest.skipIf(ROOT is None, "dataset not found")
class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = SupportAgentSuite(ROOT)

    # ---- intake / leakage
    def test_leak_guard_drops_answer_fields(self):
        t = _ticket("INC-00162")                       # has resolution_notes in the raw file
        self.assertIn("resolution_notes", t.leak_guard_dropped)
        self.assertFalse(hasattr(t, "resolution_notes"))
        self.assertEqual(t.status, "Open")             # terminal status stripped at intake

    def test_missing_fields_do_not_crash(self):
        t = ticket_from_dict({"ticket_id": "X-1", "description": "printer broken"})
        res = self.suite.process(t)
        self.assertIn(res.decision.route, {"needs_clarification", "l2_escalation", "l1_guided"})
        self.assertTrue(t.raw_missing_fields)

    # ---- happy path
    def test_clean_l1_ticket_is_guided_and_grounded(self):
        res = self.suite.process(_ticket("INC-00162"))
        self.assertEqual(res.decision.route, "l1_guided")
        self.assertGreaterEqual(len([s for s in res.steps if s.kind != "note"]), 2)
        self.assertTrue(all(s.grounded and s.source_id in self.suite.kb.by_id for s in res.steps))
        self.assertTrue(res.guardrails["passed"])
        self.assertTrue(res.enrichment.sla)
        pending = [a.action for a in res.actions if a.status == "proposed"]
        self.assertIn("suggest_priority_change", pending)          # state-changing => never auto-executed

    # ---- messy tickets
    def test_false_asset_is_flagged_not_trusted(self):
        res = self.suite.process(_ticket("INC-00011"))
        self.assertTrue(res.enrichment.has("asset_not_in_cmdb"))
        self.assertNotEqual(res.decision.route, "l1_guided")
        self.assertFalse([a for a in res.actions if a.action in ("restart_print_spooler", "remote_restart_pos_app", "clear_pos_cache")])

    def test_unknown_system_refuses_to_improvise(self):
        res = self.suite.process(_ticket("INC-00014"))
        self.assertEqual(res.decision.route, "l2_escalation")
        self.assertEqual([s for s in res.steps if s.kind != "note"], [])
        self.assertTrue(res.decision.human_review)

    def test_payroll_ticket_is_redirected(self):
        res = self.suite.process(_ticket("INC-00005"))
        self.assertEqual(res.decision.route, "non_it")

    def test_contradictory_claim_is_verified_against_itsm(self):
        res = self.suite.process(_ticket("INC-00253"))
        self.assertTrue(res.enrichment.has("reference_mismatch"))
        self.assertEqual(res.decision.route, "l2_escalation")

    def test_emotional_ticket_gets_acknowledgement_and_escalation(self):
        res = self.suite.process(_ticket("INC-00104"))
        self.assertEqual(res.decision.route, "l2_escalation")
        self.assertIn("sorry", res.requester_message.lower())

    def test_vague_ticket_asks_questions(self):
        res = self.suite.process(_ticket("INC-00149"))
        self.assertEqual(res.decision.route, "needs_clarification")
        self.assertGreaterEqual(len(res.questions), 1)

    # ---- fresh data
    def test_fresh_json_and_email_and_injection(self):
        r1 = self.suite.process(load_ticket_file(EX / "fresh_printer_offline.json"))
        self.assertEqual(r1.decision.route, "l1_guided")
        t2 = ticket_from_text((EX / "email_style.txt").read_text())
        self.assertEqual(t2.asset_id, "POS-0007-T1")
        self.suite.process(t2)
        r3 = self.suite.process(load_ticket_file(EX / "injection_attempt.json"))
        self.assertTrue(r3.enrichment.injection)
        self.assertTrue(r3.decision.human_review)
        self.assertNotIn("refund", r3.requester_message.lower())


@unittest.skipIf(ROOT is None, "dataset not found")
class TestGuardrails(unittest.TestCase):
    def test_hallucinated_llm_output_is_caught(self):
        """A scripted 'LLM' that invents a registry hack, cites a chunk it was never shown, and promises a refund."""
        seen = {}

        def responder(system, user):
            case = json.loads(re.search(r"<case>\n(.*)\n</case>", user, re.S).group(1))
            if case.get("task") == "read_ticket":                      # the reader call: return a harmless reading
                return {"issue_summary": "Receipt printer offline at counter 1", "suggested_route": "l1_guided"}
            ev = case["evidence"]
            seen["ids"] = [e["id"] for e in ev]
            good = ev[0]
            return {
                "steps": [
                    {"title": "Legit", "text": " ".join(good["text"].split())[:220], "sources": [good["id"]]},
                    {"title": "Invented fix", "text": "Run `regedit /s fixprinter.reg` and set HKLM\\Printers to 0x1F.", "sources": [good["id"]]},
                    {"title": "Phantom source", "text": "Replace the thermal head unit.", "sources": ["runbooks/does-not-exist.md#x"]},
                ],
                "requester_message": "Hi, we guarantee a full refund and it will be fixed within 10 minutes.",
            }

        suite = SupportAgentSuite(ROOT, llm=FakeLLM(responder))
        res = suite.process(load_ticket_file(EX / "fresh_printer_offline.json"))
        blob = " ".join(s.text for s in res.steps).lower()
        self.assertNotIn("regedit", blob)
        self.assertNotIn("replace the thermal head", blob)
        self.assertTrue(res.guardrails["dropped_steps"], "guardrail should report dropped steps")
        self.assertNotIn("refund", res.requester_message.lower())          # forbidden-commitment check fell back to template
        self.assertEqual(res.decision.route, "l1_guided")                   # ...and the run still completed safely

    def test_llm_crash_falls_back_to_offline(self):
        def boom(system, user):
            raise RuntimeError("rate limited")
        suite = SupportAgentSuite(ROOT, llm=FakeLLM(boom))
        res = suite.process(load_ticket_file(EX / "fresh_printer_offline.json"))
        self.assertEqual(res.decision.route, "l1_guided")
        self.assertTrue(res.steps)


class TestActionsAndAudit(unittest.TestCase):
    def test_allowlist_and_approval_gate(self):
        audit = AuditLog()
        itsm = ITSMSimulator(audit)
        bad = itsm.propose("delete_all_tickets", "T-1", {})
        self.assertEqual(itsm.execute("T-1", bad, approved=True).status, "blocked")     # unknown action can never run
        reboot = itsm.propose("remote_router_reboot", "RTR-1", {})
        self.assertEqual(itsm.execute("T-1", reboot).status, "proposed")                # needs approval
        self.assertEqual(itsm.execute("T-1", reboot, approved=True).status, "approved_simulated")

    def test_audit_chain_detects_tampering(self):
        a = AuditLog()
        for i in range(3):
            a.write("T-1", "evt", {"i": i})
        self.assertTrue(a.verify())
        a.entries[1]["payload"]["i"] = 99
        self.assertFalse(a.verify())


@unittest.skipIf(ROOT is None, "dataset not found")
class TestUIBackend(unittest.TestCase):
    """The web UI is a thin layer over the same suite; test the API object without opening a socket."""

    @classmethod
    def setUpClass(cls):
        from servewell_agent.ui.server import App
        cls.app = App(ROOT)

    def test_process_sample_and_approve(self):
        pkg = self.app.process({"sample": "INC-00162"})
        self.assertEqual(pkg["result"]["decision"]["route"], "l1_guided")
        self.assertIn("resolution_notes", pkg["ticket"]["leak_guard_dropped"])
        idx = [i for i, a in enumerate(pkg["result"]["actions"]) if a["status"] == "proposed"][0]
        out = self.app.approve({"ticket_id": "INC-00162", "index": idx})
        self.assertEqual(out["action"]["status"], "approved_simulated")
        self.assertTrue(out["audit_intact"])

    def test_batch_csv_and_bad_input(self):
        rows = self.app.batch({"content": (EX / "unseen_batch.csv").read_text(), "filename": "unseen_batch.csv"})["rows"]
        self.assertEqual(len(rows), 4)
        with self.assertRaises(ValueError):
            self.app.batch({"content": "", "filename": "x.csv"})

    def test_free_text_and_string_flags(self):
        r = self.app.process({"text": "Kiosk not working, please fix asap", "store": "SW-0007"})
        self.assertEqual(r["result"]["decision"]["route"], "needs_clarification")
        t = ticket_from_dict({"ticket_id": "X", "escalation_flag": "False"})
        self.assertFalse(t.escalation_flag)                      # CSV-style string booleans


if __name__ == "__main__":
    unittest.main()
