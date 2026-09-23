"""AI-vs-rules A/B comparison - the honest answer to "does the LLM actually help?".

Every ticket is processed twice by the SAME suite: once forced to the deterministic offline path
(`use_llm=False`) and once with the model enabled. We then compare:

  * messy-ticket behaviour checks (the 36 'chaos' tickets; expectations are self-authored, see run_eval.py)
  * routing accuracy on a sample of labelled normal tickets (optional; costs extra API calls)
  * how often the model changed the evidence, disagreed with the policy, had steps blocked, or failed
  * cost side: calls, tokens, latency (free tiers rate-limit, so this also tells you how long a run takes)

Run:   python -m servewell_agent --llm groq ab --max 36 --normal 0
"""
from __future__ import annotations

import csv
import random
import time
from collections import Counter, defaultdict
from typing import Callable, Optional

from servewell_agent import training
from servewell_agent.ingest import load_ticket_file

from evaluation.run_eval import CHECKS


def _pct(n, d):
    return f"{(100 * n / d):.0f}%" if d else "-"


def run_ab(suite, max_messy: int = 36, normal_n: int = 0, seed: int = 7,
           progress: Optional[Callable[[int, int, str], None]] = None) -> dict:
    if not suite.llm.available:
        raise RuntimeError("no language model configured - connect one first (UI: AI settings, or --llm groq)")
    w = training.load_world()
    idx = list(csv.DictReader(open(w.root / "tickets" / "train_index.csv", encoding="utf-8-sig")))
    messy = [r for r in idx if r["chaos_type"]][:max_messy]
    normal = []
    if normal_n:
        ids = sorted(w.ids)
        random.Random(seed).shuffle(ids)
        normal = ids[:normal_n]
    total = len(messy) + len(normal)
    done = 0
    rows, t_start = [], time.time()
    before = suite.llm.snapshot()

    def both(t):
        off = suite.process(t, use_llm=False)
        on = suite.process(t, use_llm=True)
        return off, on

    for r in messy:
        t = load_ticket_file(w.root / "tickets" / "train" / f"{r['ticket_id']}.json")
        off, on = both(t)
        _, fn = CHECKS[r["chaos_type"]]
        rows.append(_row(t.ticket_id, r["chaos_type"], off, on, bool(fn(off)), bool(fn(on)), None))
        done += 1
        if progress:
            progress(done, total, t.ticket_id)
    for tid in normal:
        t = w.tickets[tid]
        off, on = both(t)
        gold = w.labels[tid]["correct_routing"]
        rows.append(_row(tid, "normal", off, on, off.decision.route == gold, on.decision.route == gold, gold))
        done += 1
        if progress:
            progress(done, total, tid)

    usage = suite.llm.usage_since(before)
    return _summarise(rows, usage, time.time() - t_start, suite.llm.name)


def _row(tid, kind, off, on, ok_off, ok_on, gold) -> dict:
    llm = on.llm or {}
    rd = (llm.get("reader") or {})
    wr = (llm.get("writer") or {})
    so = rd.get("second_opinion") or {}
    return {
        "ticket_id": tid, "kind": kind, "gold": gold,
        "route_off": off.decision.route, "route_on": on.decision.route,
        "ok_off": ok_off, "ok_on": ok_on,
        "review_off": off.decision.human_review, "review_on": on.decision.human_review,
        "reader_ok": bool(rd.get("used")), "reader_error": rd.get("error"),
        "applied": rd.get("applied") or [], "flags": rd.get("flags") or [],
        "so_agree": so.get("agree"), "so_route": so.get("route"),
        "writer_used": bool(wr.get("used")), "writer_error": wr.get("error"),
        "steps_written": wr.get("steps_written", 0), "steps_dropped": len(wr.get("dropped") or []),
        "msg_used": bool(wr.get("message_used")),
    }


def _summarise(rows: list, usage: dict, secs: float, model: str) -> dict:
    by_kind = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        k = by_kind[r["kind"]]
        k[0] += 1
        k[1] += r["ok_off"]
        k[2] += r["ok_on"]
    messy = [r for r in rows if r["kind"] != "normal"]
    normal = [r for r in rows if r["kind"] == "normal"]
    rd_rows = [r for r in rows if r["reader_ok"]]
    out = {
        "model": model, "tickets": len(rows), "seconds": round(secs, 1),
        "messy": {"n": len(messy), "pass_off": sum(r["ok_off"] for r in messy), "pass_on": sum(r["ok_on"] for r in messy)},
        "normal": {"n": len(normal), "acc_off": sum(r["ok_off"] for r in normal), "acc_on": sum(r["ok_on"] for r in normal),
                   "l2_recall_off": _recall(normal, "route_off"), "l2_recall_on": _recall(normal, "route_on")},
        "by_type": {k: {"n": v[0], "off": v[1], "on": v[2]} for k, v in sorted(by_kind.items())},
        "review_rate_off": sum(r["review_off"] for r in rows), "review_rate_on": sum(r["review_on"] for r in rows),
        "reader_calls_ok": len(rd_rows), "reader_failures": sum(bool(r["reader_error"]) for r in rows),
        "writer_used": sum(r["writer_used"] for r in rows), "writer_failures": sum(bool(r["writer_error"]) for r in rows),
        "steps_written": sum(r["steps_written"] for r in rows), "steps_dropped": sum(r["steps_dropped"] for r in rows),
        "evidence_changed": sum(bool(r["applied"]) for r in rows),
        "second_opinion_agree": sum(1 for r in rd_rows if r["so_agree"]), "second_opinion_n": len(rd_rows),
        "route_changed": [r["ticket_id"] for r in rows if r["route_on"] != r["route_off"]],
        "improved": [r["ticket_id"] for r in rows if r["ok_on"] and not r["ok_off"]],
        "regressed": [r["ticket_id"] for r in rows if r["ok_off"] and not r["ok_on"]],
        "usage": usage, "rows": rows,
    }
    return out


def _recall(rows, key):
    l2 = [r for r in rows if r["gold"] == "l2_escalation"]
    return sum(1 for r in l2 if r[key] == "l2_escalation")


def format_report(m: dict) -> str:
    L = [f"AI vs RULES  |  model {m['model']}  |  {m['tickets']} tickets  |  {m['seconds']} s", "=" * 70]
    ms = m["messy"]
    L.append(f"Messy-ticket checks:   rules only {ms['pass_off']}/{ms['n']}   ->   rules + AI {ms['pass_on']}/{ms['n']}")
    for k, v in m["by_type"].items():
        if k != "normal":
            L.append(f"   {k:16s} {v['off']}/{v['n']}  ->  {v['on']}/{v['n']}")
    if m["normal"]["n"]:
        n = m["normal"]
        L.append(f"Routing accuracy (labelled sample of {n['n']}):  {_pct(n['acc_off'], n['n'])}  ->  {_pct(n['acc_on'], n['n'])}")
    L.append(f"Human-review rate:     {_pct(m['review_rate_off'], m['tickets'])}  ->  {_pct(m['review_rate_on'], m['tickets'])}")
    L.append(f"Improved by AI: {m['improved'] or 'none'}")
    L.append(f"Regressed with AI: {m['regressed'] or 'none'}")
    L.append(f"Route changed on {len(m['route_changed'])} tickets; evidence changed by the reader on {m['evidence_changed']}")
    L.append(f"Second opinion agrees with policy on {m['second_opinion_agree']}/{m['second_opinion_n']}")
    L.append(f"Writer: {m['writer_used']} tickets used AI steps; {m['steps_dropped']} of {m['steps_written']} written steps blocked by the grounding check")
    L.append(f"Failures (fell back to rules): reader {m['reader_failures']}, writer {m['writer_failures']}")
    u = m["usage"]
    L.append(f"Cost side: {u['calls']} calls, {u['prompt_tokens']} prompt + {u['completion_tokens']} completion tokens, "
             f"{u['latency_ms'] // max(1, u['calls'])} ms avg per call, {u['retries']} rate-limit retries")
    return "\n".join(L)
