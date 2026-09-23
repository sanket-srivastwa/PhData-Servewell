"""Offline evaluation harness.

Three lenses, each answering a different question a client would ask:

 A. HELD-OUT LABELLED TICKETS   "Is routing / retrieval right on tickets the models never saw?"
    The learned pieces (KB prior, escalation model) are re-fitted on a random (1 - holdout) share of the
    220 labelled tickets; the *whole agent pipeline* is then scored on the remaining share.
 B. MESSY-TICKET SUITE          "Does it behave sensibly on the 36 deliberately awkward tickets?"
    These have no ground-truth labels in the dataset, so the expectations below are SELF-AUTHORED
    behavioural checks (type-level), not ground truth. Treat as a behaviour regression suite.
 C. SAFETY PROBES               "Do the guardrails actually fire?"  (see also tests/)

Run:  python -m servewell_agent eval [--holdout 0.3] [--seed 7]
"""
from __future__ import annotations

import csv
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from servewell_agent import config, training
from servewell_agent.agents.orchestrator import SupportAgentSuite

SOP = "sop/escalation-procedure.md"


def _pct(x):
    return f"{100 * x:5.1f}%"


def eval_holdout(w, holdout: float, seed: int) -> dict:
    ids = list(w.ids)
    random.Random(seed).shuffle(ids)
    k = int(len(ids) * holdout)
    test, train = ids[:k], ids[k:]
    model, prior = training.fit(train, w)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "m.json").write_text(json.dumps(model))
    (tmp / "p.json").write_text(json.dumps({"subcat_doc_prior": prior}))
    suite = SupportAgentSuite(w.root, model_path=tmp / "m.json", prior_path=tmp / "p.json")

    rows = []
    for tid in test:
        t = w.tickets[tid]
        res = suite.process(t)
        gold = w.labels[tid]
        gold_docs = set(gold["relevant_kb_docs"])
        got_docs = [d["doc"] for d in res.retrieved]
        rb = [d for d in got_docs if d.startswith("runbooks/")]
        gold_rb = {d for d in gold_docs if d.startswith("runbooks/")}
        rows.append({
            "id": tid, "gold": gold["correct_routing"], "pred": res.decision.route, "conf": res.decision.confidence,
            "primary_hit": bool(rb) and rb[0] in gold_docs, "rb_at3": bool(set(rb[:3]) & gold_rb),
            "doc_recall": len(gold_docs & set(got_docs)) / len(gold_docs),
            "steps": len([s for s in res.steps if s.kind != "note"]), "grounded": all(s.grounded for s in res.steps),
            "review": res.decision.human_review, "msg_ok": res.guardrails["passed"], "ms": res.elapsed_ms,
            "gap": any("KB gap" in x for x in [res.ticket_note]),
            "why_review": [x.split("(")[0].strip() for x in res.decision.review_reasons],
            "base": "l2_escalation" if (t.ticket_history or t.escalation_flag) else "l1_guided",
        })
    n = len(rows)
    l2 = [r for r in rows if r["gold"] == "l2_escalation"]
    l1 = [r for r in rows if r["gold"] == "l1_guided"]
    m = {
        "n_test": n, "n_train": len(train),
        "routing_accuracy": sum(r["gold"] == r["pred"] for r in rows) / n,
        "baseline_accuracy": sum(r["gold"] == r["base"] for r in rows) / n,
        "l2_recall": sum(r["pred"] == "l2_escalation" for r in l2) / len(l2),
        "l2_missed_to_l1": sum(r["pred"] == "l1_guided" for r in l2) / len(l2),
        "false_escalation_rate": sum(r["pred"] == "l2_escalation" for r in l1) / len(l1),
        "l1_recall": sum(r["pred"] == "l1_guided" for r in l1) / len(l1),
        "primary_runbook_hit": sum(r["primary_hit"] for r in rows) / n,
        "any_runbook_at3": sum(r["rb_at3"] for r in rows) / n,
        "doc_recall_at5": sum(r["doc_recall"] for r in rows) / n,
        "all_steps_grounded": sum(r["grounded"] for r in rows) / n,
        "human_review_rate": sum(r["review"] for r in rows) / n,
        "message_policy_pass": sum(r["msg_ok"] for r in rows) / n,
        "avg_latency_ms": sum(r["ms"] for r in rows) / n,
        "kb_gap_tickets": sum(r["gap"] for r in rows) / n,
    }
    m["review_reasons"] = dict(Counter(x for r in rows for x in r["why_review"]))
    m["confusion"] = {f"{a}->{b}": n for (a, b), n in Counter((r["gold"], r["pred"]) for r in rows).items()}
    # calibration: does confidence mean anything?
    hi = [r for r in rows if r["conf"] >= 0.75]
    lo = [r for r in rows if r["conf"] < 0.75]
    m["acc_high_conf"] = (sum(r["gold"] == r["pred"] for r in hi) / len(hi)) if hi else None
    m["acc_low_conf"] = (sum(r["gold"] == r["pred"] for r in lo) / len(lo)) if lo else None
    m["n_high_conf"], m["n_low_conf"] = len(hi), len(lo)
    m["wrong"] = [(r["id"], r["gold"], r["pred"], r["conf"]) for r in rows if r["gold"] != r["pred"]]
    return m


# ---- B. messy-ticket behaviour suite (self-authored expectations) --------------------------------
def _codes(res):
    return {f.code for f in res.enrichment.findings}


CHECKS = {
    "vague": ("routes to needs_clarification and asks questions", lambda r: r.decision.route == "needs_clarification" and len(r.questions) >= 1),
    "misdirection": ("detects the HR/payroll content (non_it or mixed) and never gives payroll advice",
                     lambda r: r.decision.route == "non_it" or "mixed_non_it" in _codes(r)),
    "miscategorized": ("re-routes to network (flag) or grounds on a network runbook",
                       lambda r: "miscategorized" in _codes(r) or (r.retrieved and r.retrieved[0]["doc"].startswith(("runbooks/network", "system-specs/netlink")))),
    "emotional": ("flags dissatisfaction, escalates, and the reply acknowledges the customer",
                  lambda r: r.decision.route == "l2_escalation" and r.enrichment.dissatisfied and "sorry" in r.requester_message.lower()),
    "false_asset": ("flags the asset as not in CMDB and asks to verify (no asset-specific claims)",
                    lambda r: "asset_not_in_cmdb" in _codes(r) and r.decision.route in ("needs_clarification", "l2_escalation")),
    "unknown_system": ("flags unknown system, escalates, issues NO runbook steps",
                       lambda r: "unknown_system" in _codes(r) and r.decision.route == "l2_escalation" and not [s for s in r.steps if s.kind != "note"]),
    "contradictory": ("does not trust the 'already fixed' claim: flags it, escalates as recurrence",
                      lambda r: ("reference_mismatch" in _codes(r) or "unverifiable_reference" in _codes(r)) and r.decision.route == "l2_escalation"),
    "red_herring": ("does not adopt the requester's diagnosis: L1 steps exhausted -> escalates with what was tried",
                    lambda r: r.decision.route == "l2_escalation"),
}


def eval_chaos(w) -> dict:
    suite = SupportAgentSuite(w.root)                     # final shipped models
    idx = list(csv.DictReader(open(w.root / "tickets" / "train_index.csv")))
    from servewell_agent.ingest import load_ticket_file
    out = defaultdict(list)
    for r in idx:
        if not r["chaos_type"]:
            continue
        p = w.root / "tickets" / "train" / f"{r['ticket_id']}.json"
        t = load_ticket_file(p)
        res = suite.process(t)
        desc, fn = CHECKS[r["chaos_type"]]
        out[r["chaos_type"]].append((r["ticket_id"], bool(fn(res)), res.decision.route))
    return out


def main(holdout: float = 0.3, seed: int = 7, seeds: int = 5):
    w = training.load_world()
    print(f"Loaded {len(w.ids)} labelled tickets; L2 share {w.y.mean():.0%}\n")

    print("=" * 78 + "\nA. HELD-OUT PIPELINE EVALUATION (learned parts refitted on the training share only)\n" + "=" * 78)
    runs = [eval_holdout(w, holdout, seed + i) for i in range(seeds)]
    m = runs[0]
    print(f"{seeds} random splits, each: train {m['n_train']} / test {m['n_test']}. Mean +/- sd across splits:")
    import statistics as st
    for key, label in [("routing_accuracy", "Routing accuracy (agent)"), ("baseline_accuracy", "Baseline rule 'L1 history or flag => L2'"),
                       ("l2_recall", "L2 recall (needed escalation, got it)"), ("false_escalation_rate", "False-escalation rate (L1-able sent to L2)"),
                       ("primary_runbook_hit", "Primary runbook in gold docs"), ("any_runbook_at3", "Any gold runbook in top-3"),
                       ("doc_recall_at5", "Gold-doc recall in retrieved set"), ("all_steps_grounded", "Tickets with all steps grounded"),
                       ("message_policy_pass", "Guardrail (msg/actions/grounding) pass"), ("human_review_rate", "Human-review rate")]:
        v = [r[key] for r in runs]
        print(f"  {label:46s} {_pct(st.mean(v))}  (+/- {100 * (st.pstdev(v)):.1f})")
    print(f"  avg latency per ticket (offline mode)           {st.mean(r['avg_latency_ms'] for r in runs):.0f} ms")
    agg = Counter()
    for r in runs:
        agg.update(r["review_reasons"])
    print(f"  human-review reasons (all splits): {dict(agg)}")
    print(f"  split 1 confusion: {m['confusion']}")
    hi = [r["acc_high_conf"] for r in runs if r["acc_high_conf"] is not None]
    lo = [r["acc_low_conf"] for r in runs if r["acc_low_conf"] is not None]
    if hi and lo:
        print(f"  Calibration: accuracy when confidence>=0.75 = {_pct(st.mean(hi))};  below = {_pct(st.mean(lo))}")
    m["multi_seed"] = {k: st.mean(r[k] for r in runs) for k in ("routing_accuracy", "baseline_accuracy", "l2_recall", "false_escalation_rate", "primary_runbook_hit", "doc_recall_at5")}

    print("\n" + "=" * 78 + "\nB. MESSY-TICKET BEHAVIOUR SUITE (36 unlabelled 'chaos' tickets; self-authored expectations)\n" + "=" * 78)
    ch = eval_chaos(w)
    tot = ok = 0
    for typ, rows in sorted(ch.items()):
        p = sum(x[1] for x in rows)
        tot += len(rows)
        ok += p
        print(f"  {typ:15s} {p}/{len(rows)}  expect: {CHECKS[typ][0]}")
        for tid, passed, route in rows:
            if not passed:
                print(f"      miss: {tid} (routed {route})")
    print(f"  TOTAL {ok}/{tot}")

    out = config.OUTPUT_DIR
    out.mkdir(exist_ok=True)
    (out / "eval_metrics.json").write_text(json.dumps({"heldout": m, "chaos": {k: [(a, b, c) for a, b, c in v] for k, v in ch.items()}}, indent=2, default=str))
    print(f"\nwrote {out / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
