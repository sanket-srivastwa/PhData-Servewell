"""Command line: `python -m servewell_agent <command>`

  demo <INC-id | file.json>     narrated single-ticket run (the live demo)
  run  <file|dir|csv|jsonl>     batch: fresh / unseen tickets -> out/results.jsonl + summary table
  text "<free text>"            intake a plain-text / email style report
  eval [--holdout 0.3]          offline evaluation (routing, retrieval, grounding, messy-ticket suite)
  ui [--port 8000]              local web UI (no terminal needed after launch)
  ab [--max 36]                 AI-vs-rules comparison on the messy tickets (needs --llm)
  list [--chaos TYPE]           show sample ticket ids from the dataset
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import config
from .agents.orchestrator import SupportAgentSuite
from .ingest import load_any, load_ticket_file, ticket_from_text
from .llm import LLMError, make_llm
from .report import render, summary_row


def _find_ticket(root: Path, ref: str):
    p = Path(ref)
    if p.exists():
        return load_ticket_file(p)
    for sub in ("raw", "train", "test"):
        f = root / "tickets" / sub / f"{ref}.json"
        if f.exists():
            return load_ticket_file(f)
    raise SystemExit(f"ticket {ref!r} not found (give an INC id from the dataset or a path to a .json/.txt file)")


def _suite(args) -> SupportAgentSuite:
    root = Path(args.data_root) if args.data_root else config.find_data_root()
    try:
        llm = make_llm(args.llm, model=args.model)
    except LLMError as exc:
        raise SystemExit(f"LLM setup failed: {exc}")
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    return SupportAgentSuite(root, llm=llm, audit_path=config.OUTPUT_DIR / "audit.jsonl")


def cmd_demo(args):
    s = _suite(args)
    t = _find_ticket(s.root, args.ticket)
    res = s.process(t, approve=args.approve)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, default=str))
    else:
        print(render(res, t, show_trace=not args.no_trace))
        print(f"\n[audit log intact: {s.audit.verify()}  |  KB version {s.kb.kb_hash}  |  mode: {s.llm.name}]")


def cmd_run(args):
    s = _suite(args)
    tickets = load_any(Path(args.path))
    s.ingest_history(tickets)
    out = config.OUTPUT_DIR
    out.mkdir(exist_ok=True)
    rows = []
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for t in tickets:
            res = s.process(t, approve=args.approve)
            f.write(json.dumps({"ticket_id": t.ticket_id, "route": res.decision.route, "confidence": res.decision.confidence,
                                "human_review": res.decision.human_review, "suggested_priority": res.enrichment.suggested_priority,
                                "flags": [x.code for x in res.enrichment.findings if x.severity in ("warn", "block")],
                                "docs": [d["doc"] for d in res.retrieved[:5]], "steps": [x.source_id for x in res.steps],
                                "requester_message": res.requester_message, "ticket_note": res.ticket_note,
                                "l2_handoff": res.l2_handoff}, ensure_ascii=False) + "\n")
            rows.append(res)
            print(summary_row(res))
    from collections import Counter
    c = Counter(r.decision.route for r in rows)
    print(f"\n{len(rows)} tickets -> {dict(c)}; human review on {sum(r.decision.human_review for r in rows)}; wrote {out / 'results.jsonl'}")


def cmd_text(args):
    s = _suite(args)
    t = ticket_from_text(args.body, store_id=args.store)
    print(render(s.process(t), t, show_trace=not args.no_trace))


def cmd_eval(args):
    from evaluation.run_eval import main as eval_main
    eval_main(holdout=args.holdout, seed=args.seed, seeds=args.seeds)


def cmd_ab(args):
    from evaluation.run_llm_ab import format_report, run_ab
    s = _suite(args)
    if not s.llm.available:
        raise SystemExit("ab needs a language model, e.g.:  python -m servewell_agent --llm groq ab --max 12")
    print(f"Comparing rules-only vs rules + {s.llm.name} (free tiers are rate-limited, so this takes a few minutes)...")
    m = run_ab(s, max_messy=args.max, normal_n=args.normal, progress=lambda d, t, i: print(f"  [{d}/{t}] {i}", flush=True))
    print("\n" + format_report(m))
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "ab_metrics.json").write_text(json.dumps(m, indent=2, default=str))


def cmd_ui(args):
    from .ui.server import serve
    root = Path(args.data_root) if args.data_root else None
    serve(port=args.port, open_browser=not args.no_browser, data_root=root, llm_spec=args.llm, model=args.model)


def cmd_list(args):
    root = Path(args.data_root) if args.data_root else config.find_data_root()
    with open(root / "tickets" / "train_index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if args.chaos and r["chaos_type"] != args.chaos:
            continue
        print(f"{r['ticket_id']}  {r['category']:16s} {r['subcategory']:22s} {r['priority']}  chaos={r['chaos_type'] or '-'}")


def main():
    ap = argparse.ArgumentParser(prog="servewell_agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", help="folder containing kb/ data/ tickets/ (default: auto-detect or $SERVEWELL_DATA)")
    ap.add_argument("--llm", metavar="PROVIDER", choices=["auto", "groq", "gemini"],
                    help="use a language model: groq | gemini | auto (first key found in the environment). Default: deterministic offline mode")
    ap.add_argument("--model", help="model id for --llm (default: picked automatically from what your account can use)")
    ap.add_argument("--approve", action="store_true", help="simulate a human approving all pending actions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo"); p.add_argument("ticket"); p.add_argument("--no-trace", action="store_true"); p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("run"); p.add_argument("path"); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("text"); p.add_argument("body"); p.add_argument("--store"); p.add_argument("--no-trace", action="store_true"); p.set_defaults(fn=cmd_text)
    p = sub.add_parser("eval"); p.add_argument("--holdout", type=float, default=0.3); p.add_argument("--seed", type=int, default=7); p.add_argument("--seeds", type=int, default=5); p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("ab"); p.add_argument("--max", type=int, default=36); p.add_argument("--normal", type=int, default=0); p.set_defaults(fn=cmd_ab)
    p = sub.add_parser("ui"); p.add_argument("--port", type=int, default=8000); p.add_argument("--no-browser", action="store_true"); p.set_defaults(fn=cmd_ui)
    p = sub.add_parser("list"); p.add_argument("--chaos"); p.set_defaults(fn=cmd_list)
    args = ap.parse_args()
    args.fn(args)
