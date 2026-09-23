"""Agent 1 - Triage & Enrichment.

Responsibilities (all deterministic tool calls; no LLM needed for correctness):
  * validate the ticket against systems of record (CMDB asset, store, version matrix)
  * detect the "messy ticket" patterns that break naive agents: vague, emotional, non-IT,
    mis-categorised, wrong/unknown asset or system, contradictory references, prompt injection
  * apply the SOP priority-override rules and compute SLA clocks
  * extract what has *already been tried* so the guidance never repeats it

Output is an `Enrichment` (typed handoff) with auditable `Finding`s - each carries evidence.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

from .. import config
from ..intel_platform.kb import KnowledgeBase
from ..intel_platform.structured import DOMAIN_ASSET_TYPE, StructuredData
from ..models import Enrichment, Finding, Ticket
from ..textutil import content_tokens, find_error_codes

DOMAIN_LEX = {
    "POS": ["pos", "terminal", "register", "checkout", "pin pad", "pinpad", "cash drawer", "scanner", "barcode",
            "loyalty", "settlement", "end-of-day", "end of day", "eod", "foodtech", "orbitpos", "till", "splash screen"],
    "Printers": ["printer", "receipt", "paper jam", "thermal", "garbled", "blank receipt", "epson", "star mc", "starmc", "spooler", "print"],
    "Wi-Fi / Network": ["router", "wifi", "wi-fi", "dhcp", "ip address", "vpn", "internet", "switch", "ethernet", "ssid",
                        "netlink", "cisco", "network", "lease"],
    "Soft Serve": ["soft serve", "soft-serve", "dispens", "hopper", "cooling", "freez", "motor", "creamtech", "frostypro",
                   "cleaning cycle", "auger", "mix low"],
    "Kiosks": ["kiosk", "self-order", "self order", "touchscreen", "touchpoint"],
    "Online Orders": ["online order", "zomato", "swiggy", "aggregator", "portal", "kds", "kitchen display", "menu sync",
                      "duplicate order", "webhook", "online ordering"],
}
NETWORK_OUTAGE_MARKERS = [
    r"router.{0,25}(light|indicator|led).{0,20}\boff\b", r"(light|led)s?.{0,25}router.{0,20}\boff\b",
    r"router (light|lights|indicator)", r"wall router", r"wi-?fi (network )?(is )?not (broadcasting|showing)",
    r"wi-?fi (light|led)", r"wi-?fi isn.?t showing", r"wi-?fi name is visible but", r"internet lights?",
    r"(all|other) (other )?(devices|systems).{0,80}(unable|cannot|can.?t|not).{0,40}(internet|connect|network)",
    r"no connectivity to other", r"lost? all connectivity",
]
NON_IT_STRONG = ["salary", "payroll", "pay slip", "payslip", "personal card", "wages", "reimbursement", "bonus", "appraisal", "hr portal", "hr payroll"]
NON_IT_WEAK = ["shift schedule", "scheduled shifts", "clock in", "clock-in", "leave balance", "attendance", "hours worked", "employee id", "my shifts"]
EMOTION = [r"\bfed up\b", r"absolutely (done|frustrated)", r"\bfrustrat", r"pathetic", r"ridiculous", r"sick of", r"can.?t take this",
           r"nobody (from|seems|tells)", r"no one (cares|helps)", r"runaround", r"zero support", r"why can.?t you people", r"unacceptable",
           r"\bterrible\b", r"furious", r"\bangry\b", r"third time", r"\bagain\b.*\b(broken|not working|failed)", r"doesn.?t (ever )?work", r"nothing ever works",
           r"hanging up on me", r"like an idiot", r"tickets don.?t fix", r"i am (absolutely )?done"]
HEDGES_STRONG = [r"or something", r"can.?t read", r"don.?t know what", r"not sure (what|which)", r"somehow", r"\bweird\b", r"all weird", r"\bstuff\b", r"\bkinda\b", r"multiple devices are affected but"]
GENERIC_FAILURE = [r"(is|are) (broken|not working)\b", r"nothing is working", r"not working\W*$", r"is down or something", r"not connecting\W", r"printing wrong"]
URGENCY_ONLY = [r"\basap\b", r"immediately", r"please fix", r"fix (it )?(now|urgent)"]
PERSONAL_HR = [r"\b(my|our staff.?s?|his|her)\s+(\w+\s+)?(salary|pay\b|payroll|wages)", r"salary (hasn.?t|was|didn.?t|deposit|statement|slip)", r"payroll (records|inquiry|correction|discrepan)", r"misfiled", r"hr/payroll", r"not an it (issue|matter)", r"hr matter"]
SECONDARY_HR = r"(\balso\b|that too|on top of that|by the way)[^.]{0,60}(salary|payroll|\bpay\b|wages)"
INJECTION = [r"ignore (all |any )?(previous|prior|above) (instructions|rules)", r"system prompt", r"you are now", r"disregard (the )?(rules|guidelines|instructions)",
             r"reveal your", r"act as (an? )?(admin|root)", r"override (the )?(policy|guardrail)", r"<script", r"run the following command", r"\bsudo\b", r"jailbreak"]
CANON_TRIED = {
    "restart_device": r"(restart|reboot|power[- ]?cycl|turned (it )?off and (back )?on|force[d ]?\s*restart|hard restart|soft restart)",
    "cable_check": r"(cable|cord|usb|ethernet)s?\b.{0,50}(check|verif|reseat|secure|firm|intact|seated|connected)|(check|verif|reseat)\w*.{0,30}(cable|cord)",
    "clear_cache": r"\bcache\b",
    "driver_reinstall": r"driver.{0,30}(reinstall|updat)|(reinstall|updat)\w*.{0,20}driver",
    "router_restart": r"(restart|reboot|power[- ]?cycl)\w*.{0,25}router|router.{0,30}(restart|reboot|power)",
    "spooler_restart": r"spooler",
    "queue_clear": r"(clear|cleared).{0,20}queue",
    "test_print": r"test (print|receipt)",
    "breaker_reset": r"breaker",
    "network_check": r"(ping|connectivity|network|internet).{0,30}(verified|checked|tested|stable|confirmed)",
    "factory_reset": r"factory reset",
    "safe_mode": r"safe mode",
}
IMPACT_STORE_WIDE = [r"entire store", r"all terminals", r"all (the )?devices", r"all other devices", r"store-?wide", r"nothing is working",
                     r"everything", r"complete(ly)? (down|outage)", r"all pos", r"whole store"]
IMPACT_MULTI = [r"multiple (devices|terminals|orders)", r"several (devices|terminals)", r"other (devices|terminals)", r"guests can.?t"]
PEAK_LANG = [r"peak", r"lunch (rush|service)", r"dinner (rush|service)", r"breakfast rush", r"customers (are )?(waiting|frustrated|complaining)"]
ERROR_CENTRIC = {"Error Code", "Payment Error", "Software Update", "Terminal Startup", "Payment Timeout", "Driver Issue"}


def _has(term: str, text: str) -> bool:
    """word-start match: 'pos' must not fire inside 'supposed'."""
    return re.search(r"\b" + re.escape(term), text) is not None


def _any(patterns, text):
    return [p for p in patterns if re.search(p, text, re.I)]


def _bump(priority: str, n: int = 1) -> str:
    i = config.PRIORITY_ORDER.index(priority) if priority in config.PRIORITY_ORDER else 2
    return config.PRIORITY_ORDER[max(0, i - n)]


class TriageAgent:
    def __init__(self, sd: StructuredData, kb: KnowledgeBase, as_of_offset_min: int = 5):
        self.sd, self.kb, self.as_of_offset = sd, kb, as_of_offset_min

    # ------------------------------------------------------------------------------------------
    def run(self, t: Ticket, now: Optional[datetime] = None) -> Enrichment:
        e = Enrichment()
        text = t.text
        low = text.lower()
        full_low = (text + " " + t.history_text).lower()

        e.store = self.sd.get_store(t.store_id)
        e.claimed_domain = t.category
        if not t.store_id:
            e.findings.append(Finding("store_missing", "warn", "Ticket has no store_id."))
        elif not e.store:
            e.findings.append(Finding("store_not_found", "warn", f"Store {t.store_id} not found in store master.", {"store_id": t.store_id}))

        # ---- prompt-injection scan (ticket text is untrusted data) ------------------------------
        inj = _any(INJECTION, low)
        if inj:
            e.injection = True
            e.findings.append(Finding("prompt_injection", "warn", "Ticket text contains instruction-like content; treated as data only.", {"patterns": inj}))

        # ---- domain re-classification -----------------------------------------------------------
        scores = {d: sum(1 for w in lex if _has(w, low)) for d, lex in DOMAIN_LEX.items()}
        e.domain = t.category or max(scores, key=scores.get)
        markers = _any(NETWORK_OUTAGE_MARKERS, low)
        if t.category and t.category != "Wi-Fi / Network" and markers:
            own = scores.get(t.category, 0)
            e.findings.append(Finding("miscategorized", "warn",
                                      f"Symptoms describe a store network/router outage, but ticket is filed under '{t.category}'. Re-routing to Wi-Fi / Network.",
                                      {"claimed": t.category, "markers": markers[:3], "claimed_domain_hits": own}))
            e.domain = "Wi-Fi / Network"
        elif not t.category:
            e.findings.append(Finding("category_missing", "warn", f"No category given; inferred '{e.domain}' from text."))

        # ---- system / asset validation ----------------------------------------------------------
        asset = self.sd.get_asset(t.asset_id)
        e.asset = asset
        if not t.asset_id:
            e.findings.append(Finding("asset_missing", "warn", "No asset_id on ticket."))
        elif not asset:
            e.findings.append(Finding("asset_not_in_cmdb", "warn", f"Asset {t.asset_id} does not exist in the CMDB - not using it for enrichment.", {"asset_id": t.asset_id}))
        else:
            if t.store_id and asset["store_id"] != t.store_id:
                e.findings.append(Finding("asset_store_mismatch", "warn", f"Asset {asset['asset_id']} belongs to {asset['store_id']}, ticket is from {t.store_id}."))
            want = DOMAIN_ASSET_TYPE.get(e.domain)
            if want and asset["asset_type"] != want:
                e.findings.append(Finding("asset_domain_mismatch", "info",
                                          f"Ticket domain is {e.domain} but asset {asset['asset_id']} is a {asset['asset_type']}; using store-level lookup for the right device.",
                                          {"asset_type": asset["asset_type"], "expected": want}))
                e.resolved_asset = self.sd.resolve_asset(t.store_id, e.domain)
        if (not asset or e.has("asset_domain_mismatch")) and not e.resolved_asset:
            cand = self.sd.resolve_asset(t.store_id, e.domain)
            if cand:
                e.resolved_asset = cand
                if not asset:
                    e.findings.append(Finding("asset_candidate", "info", f"Store {t.store_id} has exactly one {cand['asset_type']}: {cand['asset_id']} - asking requester to confirm.", {"asset_id": cand["asset_id"]}))

        # effective system/version = system of record wins over what the ticket claims
        ver = None
        if asset and not e.has("asset_domain_mismatch"):
            ver = asset["system_version"]
            if t.system_version and t.system_version != ver:
                e.findings.append(Finding("version_mismatch", "warn", f"Ticket says '{t.system_version}' but CMDB says '{ver}'.", {"ticket": t.system_version, "cmdb": ver}))
        elif e.resolved_asset:
            ver = e.resolved_asset["system_version"]
        ticket_ver_known = self.kb.is_known_system(t.system_version)
        if t.system_version and not ticket_ver_known:
            e.findings.append(Finding("unknown_system", "block",
                                      f"'{t.system_version}' is not in the version matrix or any spec sheet - the KB cannot ground guidance for it.",
                                      {"system_version": t.system_version}))
        if not ver and ticket_ver_known:
            ver = t.system_version
        e.version_info = self.kb.version_info(ver)
        e.findings.append(Finding("effective_system", "info", f"Effective system for retrieval: {ver or 'unknown'}", {"system_version": ver}))
        self._ver = ver

        # ---- text-derived signals ---------------------------------------------------------------
        e.error_codes = find_error_codes(text + " " + t.history_text)
        e.tried_steps = self._tried(t)
        emo = _any(EMOTION, low)
        caps = [w for w in re.findall(r"\b[A-Z]{4,}\b", t.description) if w not in {"DHCP", "IST", "VPN", "POS", "ASAP", "USB", "WIFI", "LOYALTY", "HTTP", "HTTPS", "EOD", "TODAY"}]
        e.dissatisfied = len(emo) >= 2 or (len(emo) >= 1 and len(caps) >= 2) or bool(re.search(r"(escalate|speak to|talk to).{0,25}(manager|management)", low))
        if e.dissatisfied:
            e.findings.append(Finding("dissatisfied_customer", "warn", "Requester is frustrated / dissatisfied; SOP treats this as an escalation trigger and requires an acknowledging tone.", {"signals": emo[:4], "caps": caps[:4]}))
        e.issue_core = self._issue_core(t)

        # non-IT (scan description *and* L1 history - INC-00279 only reveals it in the history)
        nit = [w for w in NON_IT_STRONG if w in full_low]
        niw = [w for w in NON_IT_WEAK if w in full_low]
        personal = _any(PERSONAL_HR, full_low)
        pts = 2 * len(nit) + len(niw) + 4 * len(personal)
        it_pts = sum(scores.values()) + (2 if e.error_codes else 0)
        e.non_it_terms = nit + niw
        secondary = bool(re.search(SECONDARY_HR, full_low))
        if pts >= 4 and personal and not secondary and (pts >= it_pts or re.search(r"misfiled|not an it", full_low)):
            e.non_it = "dominant"
            e.findings.append(Finding("non_it_request", "warn", "Request is primarily about payroll/HR/personal matters, not IT.", {"terms": e.non_it_terms, "personal_patterns": personal[:2], "non_it_pts": pts, "it_pts": it_pts}))
        elif personal:
            e.non_it = "partial"
            e.findings.append(Finding("mixed_non_it", "info", "Ticket bundles an HR/payroll request with an IT issue; handle the IT part, redirect the rest.", {"terms": e.non_it_terms}))

        # vagueness (specificity, not diagnostic uncertainty: "not sure if it is X or Y" is fine)
        vs = 0
        hedge = _any(HEDGES_STRONG, low)
        generic = _any(GENERIC_FAILURE, low)
        if hedge:
            vs += 2
        if generic:
            vs += 1
        if _any(URGENCY_ONLY, low):
            vs += 1
        if not e.error_codes:
            vs += 1
        if not t.asset_id:
            vs += 1
        if len(t.description.split()) < 45:
            vs += 1
        if not re.search(r"\b(since|at|around)\b.{0,12}\d|\bterminal \d|\bcounter \d|\bT\d\b|\bmorning|\bevening|\btoday\b|\byesterday|\bafter\b", low):
            vs += 1
        if t.subcategory == "Error Code" and not e.error_codes:
            vs += 2
            e.findings.append(Finding("error_code_not_captured", "info", "Subcategory is 'Error Code' but no code is quoted in the ticket."))
        e.vague_score = vs
        e.vague = vs >= 5
        if e.vague:
            e.findings.append(Finding("vague_ticket", "warn", "Too little detail to act on (no error code / device specifics).", {"score": vs, "hedges": hedge[:3], "generic": generic[:2]}))

        # impact
        if _any(IMPACT_STORE_WIDE, low) or (markers and "all" in low):
            e.impact = "store_wide"
        elif _any(IMPACT_MULTI, low):
            e.impact = "multiple_devices"

        # ---- references to other tickets (contradiction / recurrence) ---------------------------
        refs = sorted({r for r in re.findall(r"INC-\d{5}", text) + list(t.related_tickets) if r != t.ticket_id})
        claim = re.search(r"(resolved|fixed|closed|confirmed|marked resolved|resolution)", low) and re.search(r"(again|still|despite|contradict|however|but)", low)
        for r in refs:
            other = self.sd.lookup_ticket(r)
            if other is None:
                e.findings.append(Finding("unverifiable_reference", "warn", f"Ticket cites {r}, which is not in the ITSM store - the 'already fixed' claim cannot be verified.", {"ref": r}))
            else:
                mism = (other.store_id != t.store_id) or (other.subcategory and other.subcategory != t.subcategory)
                if mism:
                    e.findings.append(Finding("reference_mismatch", "warn",
                        f"{r} is a {other.category}/{other.subcategory} ticket at {other.store_id}; it does not match this ticket ({t.category}/{t.subcategory} at {t.store_id}), so it cannot be relied on as the 'previous fix'.",
                        {"ref": r, "ref_store": other.store_id, "ref_subcategory": other.subcategory}))
                else:
                    e.findings.append(Finding("referenced_ticket", "info", f"Referenced {r}: {other.category}/{other.subcategory} at {other.store_id}.", {"ref": r}))
        if refs and claim:
            e.findings.append(Finding("reopened_after_claimed_fix", "warn", "Requester says a previous fix did not hold - treat as recurrence, not a new issue.", {"refs": refs}))

        # ---- recurrence & priority overrides (SOP) ----------------------------------------------
        e.prior_tickets = self.sd.prior_tickets(t, config.RECURRENCE_WINDOW_DAYS)
        text_recur = 2 if re.search(r"third time|3rd time|keeps (happening|going)|recurring", low) else (1 if re.search(r"\bagain\b", low) else 0)
        e.recurrence_count = max(len(e.prior_tickets), text_recur) + (1 if e.has("reopened_after_claimed_fix") else 0)
        self._priority(t, e, low, bool(markers))

        # ---- SLA -----------------------------------------------------------------------------------
        base = t.date_opened or datetime.now()
        now = now or (base + timedelta(minutes=self.as_of_offset))
        sla = self.sd.sla_for(e.suggested_priority, e.domain if e.domain in DOMAIN_ASSET_TYPE or e.domain in ("Online Orders",) else "General")
        if sla:
            e.sla = {
                "priority": e.suggested_priority,
                "first_response_due": (base + timedelta(minutes=sla["first_response_minutes"])).isoformat(timespec="minutes"),
                "resolution_due": (base + timedelta(minutes=sla["resolution_target_minutes"])).isoformat(timespec="minutes"),
                "escalate_to_l2_by": (base + timedelta(minutes=sla["escalation_to_l2_minutes"])).isoformat(timespec="minutes"),
                "minutes_elapsed_at_agent_start": int((now - base).total_seconds() // 60),
                "first_response_minutes": sla["first_response_minutes"],
            }

        # ---- warranty / known issues ----------------------------------------------------------------
        a = e.resolved_asset if e.has("asset_domain_mismatch") else asset
        w = self.sd.warranty_status(a, t.date_opened)
        if w == "out_of_warranty":
            e.findings.append(Finding("out_of_warranty", "info", f"{a['asset_id']} was out of warranty on {t.date_opened.date()} (expired {a['warranty_expiry']}): replacement needs a commercial decision, not a vendor RMA.", {"asset_id": a["asset_id"]}))
        if e.version_info and e.version_info.get("known_bugs", "None known") != "None known":
            bug = e.version_info["known_bugs"]
            if len(content_tokens(bug) & content_tokens(text)) >= 1 and len(content_tokens(bug) - {"during", "after", "with", "on", "due"}) > 0:
                e.findings.append(Finding("known_issue", "info", f"Version matrix lists a known issue for {ver}: '{bug}' (patch: {e.version_info.get('patch_available')}).", {"bug": bug, "patch": e.version_info.get("patch_available")}))

        # ---- mandatory info for escalation (SOP step 3) --------------------------------------------
        if not t.asset_id or not asset:
            e.missing_mandatory.append("asset_id")
        if (t.subcategory in ERROR_CENTRIC or re.search(r"error", low)) and not e.error_codes:
            e.missing_mandatory.append("error_code")
        if not e.tried_steps and not t.ticket_history:
            e.missing_mandatory.append("steps_already_tried")
        if re.search(r"multiple devices.*not sure|not sure which", low):
            e.missing_mandatory.append("affected_devices")
        return e

    # ------------------------------------------------------------------------------------------
    def _tried(self, t: Ticket) -> list:
        text = (t.description + " " + t.history_text).lower()
        return [k for k, pat in CANON_TRIED.items() if re.search(pat, text)]

    def _issue_core(self, t: Ticket) -> str:
        sents = re.split(r"(?<=[\.\!\?])\s+", t.description)
        keep = []
        allw = [w for lex in DOMAIN_LEX.values() for w in lex]
        for s in sents:
            sl = s.lower()
            if any(re.search(p, sl) for p in EMOTION):
                continue
            if any(_has(w, sl) for w in allw) or re.search(r"\d", s) or find_error_codes(s):
                keep.append(s.strip())
        core = " ".join(keep[:3]) or t.subject
        return core[:420]

    def _priority(self, t: Ticket, e: Enrichment, low: str, network_outage: bool) -> None:
        base = t.priority if t.priority in config.PRIORITY_ORDER else "P3"
        p, reasons = base, []
        if t.date_opened and any(a <= t.date_opened.hour < b for a, b in config.PEAK_WINDOWS):
            p = _bump(p); reasons.append("peak service hours (+1)")
        elif _any(PEAK_LANG, low):
            p = _bump(p); reasons.append("issue reported during peak service (+1)")
        if e.recurrence_count > config.RECURRENCE_THRESHOLD - 1 and (e.recurrence_count >= config.RECURRENCE_THRESHOLD):
            p = _bump(p); reasons.append(f"recurring issue ({e.recurrence_count} in {config.RECURRENCE_WINDOW_DAYS} days) (+1)")
        elif e.recurrence_count == 1 and e.has("reopened_after_claimed_fix"):
            p = _bump(p); reasons.append("previous fix did not hold (+1)")
        if e.dissatisfied and config.PRIORITY_ORDER.index(p) > 1:
            p = "P2"; reasons.append("customer dissatisfaction/escalation => at least P2 and route to manager")
        if e.impact == "store_wide" and t.category in ("Wi-Fi / Network", "POS") and config.PRIORITY_ORDER.index(p) > 1:
            p = "P2"; reasons.append("store-wide impact => at least P2")
        e.suggested_priority = p
        e.priority_reasons = reasons
        if p != base:
            e.findings.append(Finding("priority_override", "info", f"Priority {base} -> {p}: " + "; ".join(reasons), {"from": base, "to": p}))
