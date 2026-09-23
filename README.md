# ServeWell IT Support Agent Suite - local POC

A small, runnable proof of concept of an L1 IT-support agent suite that would run on the PhData
Intelligence Platform. The platform's building blocks (retrieval over runbooks, structured IT data,
ITSM action tools) are simulated locally from the synthetic dataset.

It takes a ticket, checks it against the systems of record, retrieves and cites the right runbook,
decides the route (L1 guided / L2 escalation / ask questions / not IT), writes the reply and the
ticket note, prepares the SOP-format L2 handoff, and proposes actions behind an approval gate. The
whole thing runs with no LLM and no network (deterministic mode). With a language model connected (using Groq or Gemini
free tiers) it also reads messy tickets, rewrites runbook steps in plain language, drafts the reply and
an L2 summary, and gives an advisory second opinion on routing.

## Quick start (web UI, no terminal after launch)

```bash
pip install -r requirements.txt
python -m servewell_agent ui            # opens http://127.0.0.1:8000 ; Ctrl+C in the terminal to stop
```

In the browser: pick a sample ticket (clean, messy, or a fresh example) or paste your own, click Run agent,
and read the result card by card. Tick Present step by step to reveal one stage per click while you narrate,
and untick Technical detail for the business view. The tab runs the held-out evaluation and shows the hash-chained audit log.
The UI uses the standard library only and binds to 127.0.0.1 (localhost, nothing exposed).
Use `--port 8080`

## Command-line quick start

```bash
pip install -r requirements.txt              # numpy + scikit-learn (training/eval only)
export SERVEWELL_DATA=/path/to/PhData/PhData   # folder that contains kb/ data/ tickets/ labels/
                                                 # (auto-detected if the repo is cloned next to this project)

python scripts/train_models.py               # fits the two small learned parts, prints cross-validated numbers
python -m servewell_agent demo INC-00162     # narrated end-to-end run (the live demo)
python -m servewell_agent demo INC-00104     # frustrated requester -> L2 handoff package
python -m servewell_agent demo INC-00014     # unknown system -> refuses to improvise
python -m servewell_agent run examples/unseen_batch.csv     # fresh data (CSV / JSON / JSONL / folder)
python -m servewell_agent demo examples/fresh_printer_offline.json
python -m servewell_agent text "POS-0007-T1 shows a black screen after the power cut" --store SW-0007
python -m servewell_agent demo examples/injection_attempt.json   # prompt-injection probe
python -m servewell_agent --approve demo INC-00162           # simulate a human approving pending actions
python -m servewell_agent eval                                # held-out evaluation + messy-ticket suite
python -m unittest discover -s tests -v                       # 36 behavioural tests
python -m servewell_agent --llm groq demo INC-00162           # with a model (needs GROQ_API_KEY); see 'Using a language model'
python -m servewell_agent --llm groq ab --max 12               # does the AI help? rules-only vs rules+AI on messy tickets
```

## What happens to a ticket

```
ticket -> intake (leak guard) -> Triage agent -> Knowledge agent -> Policy engine -> Resolve -> Guardrails
       -> Action agent (reply, note, L2 handoff, proposed actions) -> HITL gate -> audit log
```

| Stage | Code | What it does |
|---|---|---|
| Intake | `ingest.py` | Accepts JSON / CSV / JSONL / plain text. Missing fields are recorded, not fatal. Drops `resolution_notes` (in the training data it contains the answer). |
| Triage & enrichment | `agents/triage.py` | Looks up store, asset (CMDB), version matrix, SLA, similar recent tickets. Detects vague, emotional, non-IT, mis-categorised, unknown-system, wrong-asset, contradictory-reference and injection-like tickets. Applies the SOP priority-override rules. Extracts what was already tried. |
| Retrieval | `intel_platform/kb.py`, `agents/knowledge.py` | Section-aware chunking (every chunk is citeable), BM25, document ranking boosted by domain / effective system version / subcategory prior. |
| Policy | `agents/policy.py` | Routing = SOP hard rules + an interpretable logistic model (coefficients in `models/escalation_model.json`). The LLM never makes this decision; it only supplies evidence (see below) and an advisory opinion. |
| AI reader (optional) | `agents/reader.py` | LLM turns the raw ticket into structured fields; merged with the rules under strict conditions (see below). |
| Resolution | `agents/resolver.py` | Offline: extractive (runbook text + citation, skipping steps already tried). LLM mode: the model selects and rewrites the same evidence in plain language, drafts the reply and an L2 summary; all of it passes the same guardrails. |
| Guardrails | `agents/guardrails.py` | Grounding verifier (citation exists, content supported, every command/code/number appears in the cited chunk), message policy (no refunds/guarantees, no PII/internal mailboxes, acknowledgement for upset requesters), action allow-list, injection flag. |
| Communication | `agents/comms.py` | Requester reply, internal ticket note, SOP-IT-L1-002 L2 handoff. |
| Actions | `intel_platform/itsm.py` | Allow-listed catalog with risk classes. Anything not read-only or auto-approvable waits for a human. Hash-chained audit log. |
| Supervisor | `agents/orchestrator.py` | Fixed workflow, typed handoffs, a trace of every stage. |

## Evaluation (offline, on the 220 labelled training tickets)

`python -m servewell_agent eval` re-fits the learned parts on 70% of the labelled tickets and scores
the whole pipeline on the other 30%, over 5 random splits (mean +/- sd, 66 test tickets each).

| Metric | Result |
|---|---|
| Routing accuracy (L1 vs L2) | 84.8% (+/- 2.3) vs 78.8% for the rule "L1 history or flag => L2" |
| L2 recall (needed escalation, was escalated) | 89.5% |
| False-escalation rate (L1-able sent to L2) | 23.1% |
| Primary runbook is one of the labelled relevant docs | 96.1% |
| Labelled relevant docs found in the retrieved set | 85.9% |
| Accuracy when confidence >= 0.75 / below | 93% / 73% |
| Messy-ticket suite (36 unlabelled tickets, self-authored checks) | 34 / 36 |

Read these honestly:

* Labels look LLM-generated and are noisy; some contradict themselves. Do not read 85% as a ceiling or a floor.
* Grounding shows 100% because deterministic mode quotes the runbook. The interesting test is
  `tests/test_agent.py::test_hallucinated_llm_output_is_caught` (a scripted LLM invents a command and a
  source; both are dropped and the refund promise is rejected).
* The messy-ticket expectations are my own behavioural checks, not dataset ground truth.
* Human-review rate is about a third of tickets (mostly "P1 escalation: human confirms" and low confidence).
* The provider adapters (Groq, Gemini) are tested against a local mock server that imitates the providers'
  documented request/response shapes. They have not been run against the live services from the build environment,
  and I have not measured whether the AI improves results: run the comparison (`ab` command, or the dashboard tab).

## Using a language model (free options)

Using Groq free model No card needed, OpenAI-compatible, very fast. Get a key at console.groq.com/keys.
Free limits are per minute and per day and vary by model, so batches are slow; check the console for current numbers.
Gemini (aistudio.google.com/apikey) also has a free tier: Flash-Lite models have by far the biggest daily allowance, larger Flash
models only a handful of requests per day, and Google may use free-tier inputs to improve its products. Model names change often,
so the agent lists what your key can use and pre-selects a sensible one instead of hard-coding a name.

Easiest: start the UI (`python -m servewell_agent ui`), click AI: off (top right), choose a provider, paste the key, click
Connect, then Use this model. The key is kept in memory only (never written to disk, never echoed, scrubbed from errors).
Or set an environment variable before launching: `GROQ_API_KEY` or `GEMINI_API_KEY`, and use `--llm groq|gemini`.

What the model does (two calls per ticket):
1. Reads the ticket (`agents/reader.py`): issue summary, real domain, frustration, what was already tried, quoted error codes,
   non-IT content, vagueness, missing info, and a suggested route. Merge rules: it can add evidence and raise flags; an error code must
   literally appear in the ticket; it cannot make a ticket "non-IT" on its own; its route suggestion is advisory and can only send a
   ticket to human review. The added evidence flows into the policy engine and can change the route (usually toward escalation).
2. Writes the guidance (`agents/resolver.py`): plain-language steps that must cite retrieved chunks, the reply, and an L2 summary.
   Steps failing the grounding check (unknown citation, invented command/code/number, weak support) are dropped; a reply that promises
   refunds or guaranteed times is rejected and the template is used.
If a call fails (rate limit, bad JSON, network) the run silently falls back to the rules for that stage and the trace says so.
Phone numbers and e-mails are removed before text is sent, and the greeting uses a placeholder filled in locally.

The UI's Use AI switch re-runs the same ticket with and without the model so you can compare, and the dashboard's
Does the AI help? card runs the rules-vs-AI comparison on the messy tickets.

## Design decisions worth defending

* Fixed workflow, not a free-roaming agent - reproducible, testable, auditable.
* Tools for facts, model for language - assets, SLAs, history and version bugs are lookups.
* Policy decides, the LLM does not - the model's suggested route is advisory and can only send a ticket to a human.
* Keyword retrieval first - error codes need exact match; corpus is ~400 KB. Dense re-ranking is the next step.
* Refuse rather than improvise - unknown systems and KB gaps escalate with a flag.
* Autonomy ladder - read-only and low-risk replies are automatic; L2 handoffs, priority changes and remote actions need approval; P1 escalations and injection-like tickets hold the reply.

## Known limitations / next steps

* Telemetry does not exist in the dataset: automations are simulated and no device data is invented.
* Extractive guidance is faithful but not always minimal (it may include a marginal step); the AI writer is meant to fix that (unmeasured).
* The escalation model has ~17 features trained on 220 tickets; the subcategory prior carries a lot of the signal.
* The KB itself has gaps and conflicts (no loyalty-sync runbook; EOL dates differ across three documents; SLA matrix vs SOP L2 targets). The agent flags gaps; someone has to fix them.
* Next: dense re-ranker, live ITSM connector, shadow-mode harness, weekly regression run on new tickets, feedback capture from L1 staff.

## Layout

```
servewell_agent/   config, models, ingest, llm (Groq/Gemini over urllib), report, training, cli
  ui/              server.py (stdlib HTTP + JSON API), index.html (single-page app)
  intel_platform/  kb (retrieval), structured (CMDB/SLA/history), itsm (actions + audit)
  agents/          triage, knowledge, policy, resolver, guardrails, comms, orchestrator
evaluation/        run_eval.py (held-out + messy suite)   run_llm_ab.py (rules vs rules+AI)     scripts/  train_models.py     tests/  test_agent.py
models/            learned weights (JSON)     examples/  fresh tickets     out/  run outputs (audit.jsonl, results.jsonl)
```
