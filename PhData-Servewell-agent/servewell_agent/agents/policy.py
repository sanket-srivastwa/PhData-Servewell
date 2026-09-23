"""Policy engine - where the *autonomy line* is drawn.

Routing is decided by a transparent, auditable stack (not by an LLM):

  1. Hard rules from the SOP ("Common Escalation Triggers", "Do-Not-Escalate list", mandatory info)
  2. An interpretable escalation-propensity model (logistic regression on ~18 named features,
     coefficients in models/escalation_model.json) for the grey zone
  3. Confidence + guardrail results decide the *autonomy level* (auto-send vs human review)

The LLM never overrides the policy (a 'second opinion' comparison is the documented next step).
That is the core safety trade-off of the design.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

from .. import config
from ..models import Decision, Enrichment, Ticket
from .knowledge import Retrieval, Retriever

PERSIST = r"persist|still (not|fail|happen|occur|the same)|despite|no change|did(n.?t| not) (help|work|resolve)|not resolved|same (error|issue|problem)|continues|remains|even after|after (a |the )?(restart|reboot)"
INTEGRATION = r"loyalty|\bsync|zomato|swiggy|aggregator|gateway|portal|webhook|duplicate (order|charge)|payment (timeout|processor)|kitchen display|\bkds\b"
HARDWARE = r"hardware|solenoid|replace(ment)?|motor|compressor|no power|not powering|burn|smoke|leak|blank display|unresponsive|dead|damaged|cracked"
SECURITY = r"malware|ransom|unauthori[sz]ed|breach|hacked|phishing|virus|suspicious (login|activity)"
DATA_LOSS = r"data loss|corrupt|database (error|corruption)|lost (all )?(transactions|orders|data)"
INTEGRATION_SUBCATS = {"Loyalty Sync", "Duplicate Orders", "Portal Down", "Zomato Down", "Swiggy Down", "Sync Failure",
                       "Payment Timeout", "Order Acceptance"}
STORE_NETWORK_SUBCATS = {"Router Offline", "Intermittent", "DHCP", "VPN"}

FEATURES = ["hist_any", "hist_n", "flag", "p1", "p2", "peak", "tried_n", "persist", "impact_high", "recurrence",
            "dissatisfied", "integration", "hardware", "error_code", "subcat_prior", "esc_hits", "store_network"]


def extract_features(t: Ticket, e: Enrichment, esc_hits: int, subcat_prior: float) -> dict:
    low = (t.text + " " + t.history_text).lower()
    return {
        "hist_any": 1.0 if t.ticket_history else 0.0,
        "hist_n": float(len(t.ticket_history)),
        "flag": 1.0 if t.escalation_flag else 0.0,
        "p1": 1.0 if t.priority == "P1" else 0.0,
        "p2": 1.0 if t.priority == "P2" else 0.0,
        "peak": 1.0 if any("peak" in r for r in e.priority_reasons) else 0.0,
        "tried_n": float(len(e.tried_steps)),
        "persist": 1.0 if re.search(PERSIST, low) else 0.0,
        "impact_high": 1.0 if e.impact in ("store_wide", "multiple_devices") else 0.0,
        "recurrence": float(min(e.recurrence_count, 3)),
        "dissatisfied": 1.0 if e.dissatisfied else 0.0,
        "integration": 1.0 if (t.subcategory in INTEGRATION_SUBCATS or re.search(INTEGRATION, low)) else 0.0,
        "hardware": 1.0 if re.search(HARDWARE, low) else 0.0,
        "error_code": 1.0 if e.error_codes else 0.0,
        "subcat_prior": subcat_prior,
        "esc_hits": float(min(esc_hits, 4)),
        "store_network": 1.0 if (t.subcategory in STORE_NETWORK_SUBCATS and e.impact != "single_device") else 0.0,
    }


class EscalationModel:
    """Tiny pure-python logistic scorer (weights fitted offline by scripts/train_models.py)."""

    def __init__(self, path: Optional[Path] = None):
        self.w, self.mu, self.sd, self.b = {}, {}, {}, 0.0
        self.subcat_rate: dict = {}
        self.global_rate = 0.65
        self.loaded = False
        p = Path(path or config.MODELS_DIR / "escalation_model.json")
        if p.exists():
            d = json.loads(p.read_text())
            self.w, self.mu, self.sd, self.b = d["coef"], d["mean"], d["std"], d["intercept"]
            self.subcat_rate = d.get("subcat_rate", {})
            self.global_rate = d.get("global_rate", 0.65)
            self.loaded = True

    def prior_for(self, subcat: str) -> float:
        return self.subcat_rate.get(subcat, self.global_rate)

    def score(self, feats: dict) -> tuple:
        """returns (probability of needing L2, per-feature contributions)"""
        if not self.loaded:
            # conservative fallback if no model file: lean on the strongest visible signals
            z = -0.4 + 1.6 * feats["hist_any"] + 1.2 * feats["flag"] + 0.8 * feats["tried_n"] * 0.3 + 1.0 * feats["dissatisfied"] + 0.8 * feats["integration"]
            return 1 / (1 + math.exp(-z)), {}
        contrib, z = {}, self.b
        for f in FEATURES:
            x = (feats[f] - self.mu[f]) / (self.sd[f] or 1.0)
            c = self.w[f] * x
            contrib[f] = round(c, 3)
            z += c
        return 1 / (1 + math.exp(-z)), contrib


class PolicyEngine:
    def __init__(self, retriever: Retriever, model: Optional[EscalationModel] = None):
        self.rt = retriever
        self.model = model or EscalationModel()

    def decide(self, t: Ticket, e: Enrichment, r: Retrieval) -> Decision:
        d = Decision(route="l1_guided")
        low = (t.text + " " + t.history_text).lower()
        hits = self.rt.escalation_hits(r.primary, t, e) if r.primary else []
        feats = extract_features(t, e, len(hits), self.model.prior_for(t.subcategory))
        p, contrib = self.model.score(feats)
        d.escalation_score = round(p, 3)
        d.reasons.append(f"escalation propensity {p:.2f} (threshold {config.ESCALATION_THRESHOLD:.2f})")
        if hits:
            d.reasons.append(f"{len(hits)} of the runbook's own 'When to escalate' criteria already met")
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:3]
        if top:
            d.reasons.append("top drivers: " + ", ".join(f"{k}({v:+.2f})" for k, v in top))

        # ---------------- hard rules, in priority order -------------------------------------
        if e.non_it == "dominant":
            d.route = "non_it"
            d.hard_triggers.append("non_it_request")
            d.reasons.insert(0, "Payroll/HR matter - out of IT scope (SOP Do-Not-Escalate: route to owning function, not L2)")
            d.confidence = 0.85
            return self._autonomy(d, t, e, r)

        if e.has("unknown_system"):
            d.route = "l2_escalation"
            d.hard_triggers.append("unknown_system")
            d.reasons.insert(0, "System is not covered by the KB / version matrix - agent will not guess (SOP: unfamiliar to L1 => L2)")
            d.confidence = 0.8
            d.human_review = True
            d.review_reasons.append("no grounding available for this system")
            return self._autonomy(d, t, e, r)

        if e.vague and not e.dissatisfied:
            d.route = "needs_clarification"
            d.hard_triggers.append("vague_ticket")
            d.reasons.insert(0, "Ticket lacks the SOP mandatory information - ask before acting")
            d.confidence = 0.75
            return self._autonomy(d, t, e, r)

        if e.has("asset_not_in_cmdb") and not e.dissatisfied:
            d.route = "needs_clarification"
            d.hard_triggers.append("asset_not_in_cmdb")
            d.reasons.insert(0, "Asset ID does not exist in the CMDB - must be verified before any asset-specific action or L2 handoff")
            d.confidence = 0.75
            return self._autonomy(d, t, e, r)

        triggers = []
        if re.search(SECURITY, low):
            triggers.append("security_incident")
        if re.search(DATA_LOSS, low):
            triggers.append("data_loss_or_corruption")
        if e.dissatisfied:
            triggers.append("customer_dissatisfied_or_requested_escalation")
        if e.impact == "store_wide" and t.subcategory != "Guest Wi-Fi" and (e.domain == "Wi-Fi / Network" or t.subcategory in STORE_NETWORK_SUBCATS):
            triggers.append("store_wide_network_failure")
        if e.has("reopened_after_claimed_fix"):
            triggers.append("previous_fix_did_not_hold")
        d.hard_triggers += triggers

        if triggers:
            d.route = "l2_escalation"
            d.reasons.insert(0, "SOP escalation trigger(s): " + ", ".join(triggers))
        elif p >= config.ESCALATION_THRESHOLD:
            d.route = "l2_escalation"
            d.reasons.insert(0, "Escalation propensity above threshold - L1 steps look exhausted or issue is backend-level")
        else:
            d.route = "l1_guided"
            d.reasons.insert(0, "L1-solvable: runbook applies and no escalation trigger fired")

        # ---------------- grounding sufficiency -------------------------------------------------
        if d.route == "l1_guided" and not r.sufficient:
            d.route = "l2_escalation"
            d.reasons.insert(0, "No runbook grounds this issue well enough (KB coverage gap) - escalating instead of improvising")
            d.human_review = True
            d.review_reasons.append("KB coverage gap")

        # confidence = distance from the decision boundary, tempered by retrieval strength
        margin = abs(p - config.ESCALATION_THRESHOLD) * 2
        conf = 0.5 + 0.5 * margin
        if triggers:
            conf = max(conf, 0.8)
        conf *= 0.6 + 0.4 * min(1.0, r.primary_score) if r.primary else 0.6
        if r.gap:
            conf *= 0.85
            d.reasons.append(f"KB coverage gap: no runbook is named for '{t.subcategory}' - nearest match used, confidence reduced")
        d.confidence = round(min(conf, 0.99), 2)
        return self._autonomy(d, t, e, r)

    def _autonomy(self, d: Decision, t: Ticket, e: Enrichment, r: Retrieval) -> Decision:
        """Autonomy ladder: what the agent may do without a human."""
        if d.confidence < config.LOW_CONFIDENCE:
            d.human_review = True
            d.review_reasons.append(f"low confidence ({d.confidence})")
        if e.injection:
            d.human_review = True
            d.review_reasons.append("prompt-injection-like content in ticket")
        if e.suggested_priority == "P1" and (t.priority == "P1" or d.route == "l2_escalation"):
            d.human_review = True
            d.review_reasons.append("P1: human confirms before customer-facing send / L2 page")
        if d.route == "l2_escalation":
            d.autonomy = "A3_prepare_handoff (human confirms P1/P2)"
        elif d.route in ("needs_clarification", "non_it"):
            d.autonomy = "A1_message_only"
        else:
            d.autonomy = "A2_guide_and_propose_actions (state-changing actions need approval)"
        return d
