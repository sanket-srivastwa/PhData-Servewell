# Owner's guide: how the ServeWell IT Support Agent works, inside out

Read this once end to end, then use the table of contents as a lookup. Every number here was taken from the
running code or a measured run, and the places where something is unmeasured or weak are marked **(gap)**.

1. The 60-second version
2. The data: every file, who reads it, and how
3. The code map
4. One ticket, traced with real values
5. Retrieval, in detail (chunking, BM25, ranking, thresholds)
6. Triage rules catalogue
7. The policy engine: rules, model, confidence, autonomy
8. Guardrails
9. Actions, approvals, audit log
10. The AI layer: where the prompts are, what is sent, how output is merged
11. Training, testing, and what the numbers mean
12. Configuration constants
13. Is it really "agents"?
14. Scaling
15. Known weaknesses (the honest list)
16. Questions you will be asked, with answers
17. How to prove things live

---

## 1. The 60-second version

A ticket goes through a fixed sequence: **intake -> triage/enrichment -> retrieval -> policy decision -> guidance ->
guardrails -> communication -> actions (with human approval) -> audit**. One function, `SupportAgentSuite.process()` in
`agents/orchestrator.py`, runs that sequence and returns one `AgentResult` object. The CLI, the web UI and the
evaluation scripts all call that same function.

Without an AI model the whole thing runs on: exact lookups (asset, store, SLA, ticket history), keyword rules, a
keyword search engine (BM25) over the runbooks, a small logistic-regression score, and string checks. With a model
connected, it adds two calls per ticket: a **reader** (turns the messy ticket into structured fields) and a **writer**
(rewrites the retrieved runbook text into plain language and drafts the reply). Everything the model returns is treated
as untrusted data: validated, checked against the source text, and never allowed to trigger an action.

## 2. The data: every file, who reads it, and how

The "lookup between all the data" is not one mechanism. Each file has its own, and the **glue is IDs and metadata**
on the ticket (`store_id`, `asset_id`, `category`, `subcategory`, `date_opened`, `system_version`, free text).

| File(s) | Size | Read by | How it is used |
|---|---|---|---|
| `data/stores.csv` | 120 stores | `intel_platform/structured.py` | Loaded into a dict keyed by `store_id`. **Exact lookup**: store name, region, region manager (used in the "not IT" redirect). |
| `data/assets.csv` | 477 assets | `structured.py` | Dict keyed by `asset_id`. **Exact lookup** -> asset type, store, system version, warranty date. If the ID is missing the ticket gets the flag `asset_not_in_cmdb`. Also "find the one printer/POS/router in this store" when the ticket's asset is wrong. |
| `data/sla_matrix.csv` | priority x category | `structured.py` | **Table scan** by (priority, category), falling back to "General". Gives first-response / resolution / L2-handoff minutes. Deadlines = ticket open time + minutes. |
| `tickets/train/*.json` (256) | history corpus | `ingest.py`, `structured.py` | Loaded as *history*. **Filtered scan**: earlier tickets from the same store, in the 30 days before, with the same asset or subcategory (recurrence). Also **exact lookup** of tickets the requester cites ("INC-00015 fixed this"). |
| `kb/runbooks/*.md` (32), `kb/faq/*.md` (6), `kb/sop/*.md` (3), `kb/system-specs/*.md` (7) | 48 docs, ~529 KB | `intel_platform/kb.py` | Split into **926 chunks**, indexed for **BM25 keyword search** (section 5). |
| `kb/system-specs/version-matrix.csv` | one row per system version | `kb.py` | **Exact lookup** by "system + version". Gives known bugs and patch info. Rows are citeable evidence but are not part of the search index. |
| `labels/train_labels.json` (220) | answers | `training.py`, `scripts/train_models.py`, `evaluation/`, UI badge | **Training and testing only.** Never read when processing a live ticket. |
| `tickets/train_index.csv` | chaos-type tags | evaluation, UI sample picker | Tells us which tickets are the 36 "messy" ones. Never read by the agent itself. |
| `models/escalation_model.json`, `models/kb_prior.json` | learned | `agents/policy.py`, `kb.py` | Output of training: 17 coefficients, and a subcategory -> helpful-documents table. |
| `resolution_notes` field inside each ticket | the answer | nobody | **Deliberately dropped at intake** (leak guard, `ingest.py`). |

So the honest answer to "is it just text matching?" is: **text matching is one of six mechanisms.** IDs are joined by
dictionary lookups; free text is searched with BM25; rules are regular expressions; the SOP is encoded as
if-statements; one small statistical model scores "needs L2"; the guardrails are string and set checks.

## 3. The code map

About 5,200 lines in total (tests and UI included).

| File | Lines | Job |
|---|---|---|
| `agents/orchestrator.py` | 248 | The supervisor. `process()` runs every stage and builds the trace. |
| `agents/triage.py` | 322 | `TriageAgent.run()`: lookups, messy-ticket detectors, priority, SLA. Output: `Enrichment`. |
| `agents/knowledge.py` | 105 | `Retriever`: builds the query, ranks documents, picks the primary runbook, checks "sufficient". |
| `agents/policy.py` | 204 | Feature extraction, the escalation model, the hard rules, confidence, autonomy. |
| `agents/reader.py` | 200 | AI reader: prompt, schema validation, merge rules, second opinion. |
| `agents/resolver.py` | 176 | Picks steps (extractive), or asks the model to write them (AI). Holds the writer prompt. |
| `agents/guardrails.py` | 132 | Grounding checker, message policy, action allow-list, report. |
| `agents/comms.py` | 188 | Reply, internal note, L2 handoff, clarifying questions. Templates. |
| `intel_platform/kb.py` | 317 | Chunker, BM25, document ranking. |
| `intel_platform/structured.py` | 96 | CMDB / store / SLA / ticket-history "tools". |
| `intel_platform/itsm.py` | 119 | Action catalogue, approval gate, hash-chained audit log. |
| `llm.py` | 407 | Groq / Gemini over plain HTTP, throttling, retries, key scrubbing, model auto-pick. |
| `models.py` | 146 | The typed hand-off objects: `Ticket`, `Enrichment`, `Decision`, `Step`, `ProposedAction`, `AgentResult`. |
| `ingest.py` | 132 | JSON / CSV / JSONL / plain text -> `Ticket`; leak guard. |
| `textutil.py` | 57 | Tokeniser, stemmer, error-code regex. |
| `training.py`, `scripts/train_models.py` | 135, 37 | Fits the two learned parts; cross-validation. |
| `evaluation/run_eval.py`, `run_llm_ab.py` | 184, 148 | Held-out evaluation; rules-vs-AI comparison. |
| `ui/server.py`, `ui/index.html` | 381, 552 | Local web app (standard library server + one HTML page). |
| `tests/` | 625 | 36 tests, including a mock provider server. |

Reading order that works: `models.py` -> `orchestrator.py` -> `triage.py` -> `policy.py` -> `guardrails.py` ->
`kb.py` -> `reader.py`/`resolver.py`.

## 4. One ticket, traced with real values (INC-00162, receipt printer stopped printing)

1. **Intake** (`ingest.py`): file becomes a `Ticket`. `resolution_notes` is dropped and the drop is recorded.
2. **Triage** (`triage.py`): store found; asset POS-0111-T1 found (FoodTech POS v4.2.1, warranty expired 2025-05-26);
   no error code; "power cycling" -> tried step `restart_device`. Opened at 19:18, inside the 18:00-21:00 peak window, so
   priority P2 -> P1 (SOP rule). SLA deadlines computed from the P1 row.
3. **Retrieval** (`knowledge.py`): query = subject + cleaned issue text + subcategory + error codes + system version.
   Document ranking: `pos-printer-integration.md` 1.27, `printer-offline.md` 0.83, `printer-driver-reinstall.md` 0.73,
   `foodtech-pos.md` 0.61. Primary runbook = the first; "sufficient" because its score is above 0.30.
4. **Decision** (`policy.py`): no hard rule fires; escalation score about 0.13 (threshold 0.50) -> `l1_guided`.
5. **Guidance** (`resolver.py`): immediate steps + best diagnosis/resolution chunks from the primary runbook + one from
   the secondary, minus anything already tried. Each step carries its chunk id.
6. **Guardrails**: each step's cited chunk must be in the retrieved set; commands/codes/numbers must appear in it.
7. **Communication** (`comms.py`): reply and note from templates (or the model's text if it passed the checks).
8. **Actions**: reply and note run automatically (simulated); the priority change waits for approval; a read-only ping is
   simulated. Nothing leaves the machine.
9. **Audit**: intake and decision events chained by hash.

Offline this takes about 10 ms per ticket. With a model it takes as long as two API calls (seconds).

## 5. Retrieval, in detail

### Chunking (`kb.py: chunk_markdown`)
* A new chunk starts at every `##`, `###` or `####` heading; each chunk remembers its heading path
  (e.g. `runbooks/printer-offline.md > L1 Diagnosis Steps > For USB Printers`). That path is the citation.
* Sections longer than 1,500 characters are cut at numbered items or bold titles and regrouped to about 950 characters.
* FAQ files split per `**Q:` question.
* Each chunk gets a **section type** from its `##` heading: immediate, diagnosis, resolution, escalate, symptoms, overview, etc.
* Result: 926 chunks; median 482 characters, 90% under 954, largest 5,921 (a table).
* **(gap)** Nine chunks are over 1,500 characters (tables are not split), some are tiny fragments (85 characters), there is
  no overlap between chunks, and no parent-context expansion.

### The search (`kb.py: BM25`)
* **Tokenising** (`textutil.py`): lowercase, split on spaces and punctuation, keep hyphen/dot parts too (so `ERR-4521`
  is searchable whole and in parts), drop stop-words, strip endings (-ing, -ed, -s, ...). Heading words are added to a chunk's
  text twice so headings count more.
* **Index**: for each word, the list of chunks containing it and how often. Built at start-up in about 0.2 s (vocabulary
  4,170 words). It is rebuilt from the files every start, and stamped with a content hash.
* **Scoring** (BM25, k1 = 1.4, b = 0.75): a chunk scores higher when it contains rare query words (rarity = IDF),
  more occurrences help but with diminishing returns, and long chunks are slightly penalised.

### Document ranking (`kb.py: rank_docs`)
Score per document = best chunk score + 0.25 x its next two chunks, normalised, then:

`0.55 x bm25` + `0.12` if the doc's domain matches the ticket's + `0.30 x` word overlap between the subcategory and the
runbook file name + `0.30 x` learned prior (share of training tickets of this subcategory that used this doc) + `0.15` if
it is the spec sheet for the ticket's system - `0.15` for non-escalation SOPs.

Example (INC-00162): `pos-printer-integration.md` = bm25 1.00 + domain + subcat-name 1.00 + prior 1.00 = 1.27.

### From documents to steps
* Primary runbook = top-ranked runbook. Secondary = next runbook if its score is at least 60% of the primary's (and >= 0.5).
* `sufficient` = primary score >= 0.30 **and** the system is known. If not, no steps are issued.
* Extractive steps: up to 2 immediate chunks; top 2 diagnosis and top 2 resolution chunks by BM25 relevance, dropping anything
  under 55% of the best; 1 chunk from the secondary runbook; maximum 6. Chunks whose heading says they are a step the
  requester already did are skipped. For an L2 route only 2 safe immediate chunks are shown; for "needs clarification", 1.

### Measured quality (held-out, 5 splits)
Primary runbook is one of the labelled relevant docs: **96%**. Labelled relevant docs found in the retrieved set: **86%**.
Early test without the learned prior: about 80%. **(gap)** These are document-level only; there are no chunk-level labels,
so whether the right *steps* are chosen is unmeasured.

## 6. Triage rules catalogue (`triage.py`)

| Detector | Rule (summary) |
|---|---|
| Asset checks | Missing -> `asset_missing`; not in CMDB -> `asset_not_in_cmdb`; wrong store -> `asset_store_mismatch`; wrong type for the category -> `asset_domain_mismatch` (then look up the right device in the store); version differs from CMDB -> `version_mismatch`. |
| Unknown system | Ticket's system version not in the version matrix or any spec sheet -> `unknown_system` (blocks guidance). |
| Miscategorised | Symptom patterns for a store-wide router outage ("router light off", "wall router", "all other devices can't connect") while filed under another category -> re-route to Wi-Fi / Network. |
| Frustration | Two or more emotion phrases, or one plus two ALL-CAPS words, or "speak to a manager" -> `dissatisfied_customer`. |
| Non-IT | Points: strong terms (salary, payroll) x2, weak terms x1, personal-payroll patterns x4. **Dominant** if points >= 4 with a personal pattern, no "also ... salary" phrasing, and points >= IT points (or "misfiled"). Otherwise "partial" (fix IT part, redirect the rest). Also scans the L1 history. |
| Vague | Score: strong hedge (+2), generic failure phrase (+1), urgency-only (+1), no error code (+1), no asset (+1), under 45 words (+1), no time/sequence detail (+1), "Error Code" subcategory without a code (+2). **Vague if >= 5.** |
| Already tried | 12 canonical patterns (restart, cable check, clear cache, driver reinstall, router restart, spooler, queue clear, test print, breaker, network check, factory reset, safe mode) on the description and L1 notes. |
| Error codes | Regex for hex codes, `E-402`, `ERR-4521`, `E-PRINT-502`, `E04`, ... device models and asset IDs are excluded. |
| References | Any `INC-#####` in the text is looked up. Not found -> `unverifiable_reference`; found but different store/subcategory -> `reference_mismatch`; plus "fixed ... but still" -> `reopened_after_claimed_fix`. |
| Priority (SOP) | +1 peak hours (11-14, 18-21) or peak language; +1 recurrence (2+ in 30 days) or a fix that did not hold; frustrated -> at least P2; store-wide POS/network -> at least P2. Advisory: changing it needs approval. |
| Impact | Regex lists: store-wide ("all terminals", "entire store"), multiple devices, single. |
| Injection | Patterns like "ignore previous instructions", "system prompt", `<script` -> flagged, hold for human review. |
| Other findings | Out-of-warranty at ticket date; known issue in the version matrix if the ticket shares a word with the bug text. |

**(gap)** These are hand-written patterns tuned while looking at the 36 messy training tickets. The panel's tickets may
phrase things differently. The AI reader exists partly to cover that.

## 7. The policy engine (`policy.py`)

**Step 1: the escalation score.** A logistic regression on 17 named features, fitted on the 220 labelled tickets.
Fitted weights (standardised features; positive = pushes toward L2):

| Feature | Weight | Meaning |
|---|---|---|
| `hist_n` | +1.02 | number of L1 history entries |
| `subcat_prior` | +0.92 | how often this subcategory needed L2 in training |
| `flag` | +0.83 | ticket already flagged for escalation |
| `integration` | +0.81 | loyalty / sync / aggregator / gateway / portal / duplicate-order language |
| `hardware` | +0.60 | hardware-failure language |
| `esc_hits` | +0.43 | how many of the runbook's own "when to escalate" bullets are already met |
| `recurrence` | +0.35 | repeat issue |
| `peak` | +0.26 | peak hours |
| `hist_any` | +0.24 | any L1 history |
| `persist` | +0.19 | "still", "despite", "no change" |
| `error_code`, `p2`, `impact_high`, `p1` | +0.12, +0.11, +0.08, +0.01 | small |
| `tried_n` | -0.08 | count of steps already tried |
| `dissatisfied`, `store_network` | 0.00 | no variation in the normal training tickets, so the model ignores them; **hard rules handle these** |

Intercept +1.38. Cross-validated: accuracy 86%, L2 recall 91%, L2 precision 88%, AUC 0.91. Baseline rule "history or flag"
scores 79.5%. A `words_log` feature (description length) was **removed**: it looked like a data-generator artefact and
removing it cost nothing.

**Step 2: hard rules, in this order** (first match wins):
1. Non-IT dominant -> `non_it`.
2. Unknown system -> `l2_escalation`, human review.
3. Vague and not frustrated -> `needs_clarification`.
4. Asset not in CMDB and not frustrated -> `needs_clarification`.
5. SOP triggers (security incident, data loss, frustrated/asks manager, store-wide network failure, previous fix did not hold) -> `l2_escalation`.
6. Score >= 0.50 -> `l2_escalation`; otherwise `l1_guided`.
7. If it is `l1_guided` but retrieval is not sufficient -> `l2_escalation` with human review (KB gap).
8. Later, in the orchestrator: fewer than 2 grounded steps survive -> `l2_escalation`.

**Step 3: confidence.** Roughly: distance of the score from 0.50, raised to at least 0.8 when a hard trigger fired,
scaled by retrieval strength, capped at 0.99.

**Step 4: autonomy / human review.** Review is required when confidence < 0.55, when the ticket looks like an injection
attempt, when a P1 ticket is being escalated (or was filed P1), when the AI's reader raised a flag, or when the AI's second opinion
says a would-be L1 ticket should go elsewhere. Held tickets keep even low-risk messages waiting for approval.

## 8. Guardrails (`guardrails.py`)

* **Grounding**, per step: (1) cited chunk exists and is in the retrieved evidence set; (2) every "hard entity" in the step
  (backtick commands, file paths, hex/error codes, versions, IPs, numbers with units) appears literally in the cited chunk;
  (3) enough of the step's content words appear in the source (60% for quoted text, 35% for AI-written text). Fail -> step dropped.
* **Coverage**: fewer than 2 grounded steps -> escalate instead of sending thin advice.
* **Message policy**: no refunds/compensation/guarantees/"100%"/promised fix times; no internal mailboxes to a store; no
  requester phone echoed; an acknowledgement is required if the requester is upset; under 3,000 characters.
* **Action allow-list**: only catalogued actions exist; state-changing ones can never auto-execute.
* **Input**: ticket text is untrusted; instruction-like patterns are flagged.

## 9. Actions, approvals, audit (`itsm.py`)

| Action | Risk | Approval |
|---|---|---|
| post internal note, post requester reply, request clarification, redirect non-IT | reversible | automatic (held if the ticket needs human review) |
| run diagnostic ping | read-only | automatic (simulated) |
| restart print spooler, remote restart POS app, clear POS cache | reversible | approval |
| suggest priority change, escalate to L2, remote router reboot | state-changing | approval |

Automations are proposed only with a verified asset, only if the runbook step mentions it, and never if already tried.
Execution is **simulated**: no telemetry exists in the dataset, so results say "SIMULATED". The audit log is JSON lines
where each entry includes the hash of the previous one; editing any entry breaks the chain and `verify()` returns false.

## 10. The AI layer

**Where the prompts live**
* Reader: `agents/reader.py`, constant `SYSTEM`; the user message is built in `TicketReader.read()`.
* Writer: `agents/resolver.py`, constant `Resolver.WRITER_SYSTEM`; the user message is built in `Resolver.llm_steps()`.
* Connection test: `llm.py`, `HTTPLLM.test()` (a tiny "reply with JSON" prompt).
Both real prompts are a **system message** (the rules) plus a **user message** that is `<case>` + JSON + `</case>`.

**Reader call** (about 900 characters of case JSON): subject, description (phones/e-mails redacted, 1,500 characters max),
filed category/subcategory/priority, last 4 L1 notes, three facts from the systems of record (asset exists, system known, asset type
matches), plus the allowed values. Returns: issue summary, real domain, wrong-category flag, sentiment, asks-for-manager, steps
already tried, quoted error codes, non-IT content, specificity, missing info, suggested route.

**Merge rules** (`TicketReader.merge`):
* Issue summary replaces the rule-based one (used for the search query, reply and handoff).
* Tried steps: union, canonical values only.
* Error codes: added only if they literally appear in the ticket; otherwise ignored **and flagged**.
* Angry or asks for manager -> frustration flag (can escalate).
* Non-IT: can add "partial"; can never make it "dominant" on its own (flagged for review instead).
* Wrong category: re-classified only for a clear, different IT domain (also flagged for review).
* Vague: can tip a borderline case (rule score >= 3), not create one.
* Missing info: accepted only when it can be verified from the ticket.
* Second opinion: differing route from the policy on an L1 ticket -> human review; otherwise logged only.

**Writer call** (about 4,400 characters): issue summary, category, system, error codes, already-tried list, whether the
requester is upset, the route, and the top 6 retrieved chunks (each cut to 900 characters). Returns steps (each with cited chunk
ids), the reply (starting "Hi {{NAME}},", filled in locally so the name is not sent), and an L2 summary. Every step then passes the
grounding guardrail; the reply passes the message policy or the template is used instead.

**Failure handling**: any LLM error (rate limit, bad JSON, network) is caught per stage; the stage falls back to rules and the
trace records why. Rate limits: minimum spacing between calls (Groq 2.2 s, Gemini 4.5 s), honour `Retry-After` up to 45 s,
3 retries, quota-exhausted messages fail fast. Keys stay in memory and are scrubbed from every error message.

**(gap)** The writer's evidence is the top 6 chunks by keyword score *without* filtering by section type, so it can include
low-value chunks (e.g. "Affected Systems") and dilute the useful steps. Filtering to immediate/diagnosis/resolution chunks is
a small, likely worthwhile fix. Also **(gap)**: the adapters were tested only against a mock of the providers' documented
formats, and nobody has yet measured whether the AI improves outcomes.

## 11. Training, testing, and what the numbers mean

* **Training** happens only in `scripts/train_models.py` (and inside the evaluation). It fits (a) the logistic regression,
  (b) the subcategory -> documents table. It writes two JSON files in `models/`. The UI never trains. **The language model
  is not trained by us at all.**
* **Data split**: 220 labelled tickets (65% are L2, 35% L1). The shipped models are trained on all 220.
* **Honest test** (`python -m servewell_agent eval`, or the Evaluation tab): for each of 5 random splits, refit on 70%
  and score the whole pipeline on the unseen 30% (66 tickets). Averages: routing accuracy **84.8%** (baseline 78.8%),
  L2 recall **89.5%**, false escalations **23.1%**, primary runbook **96%**, doc recall **86%**.
* **Messy-ticket suite**: 34 of 36 checks pass. Those tickets have no labels; the expectations are **my own** ("vague ticket
  asks questions", "unknown system gets no steps"), and I tuned detectors while looking at these tickets, so it is a regression
  suite, not an unbiased test.
* **Unseen data**: the panel's tickets. Nothing has seen them.
* **The "Dataset label" badge** in the UI compares to a label the shipped model was trained on; it is **not** a test.
* **Label quality**: labels look LLM-generated and are noisy. Example: the learned prior links "Loyalty Sync" to a network
  runbook because the labels do. Do not treat 85% as a ceiling or a floor.
* **Precision vs recall**: the threshold (0.50) trades missed escalations against unnecessary ones. Lower it to catch more L2
  cases at the price of more false escalations. Other values were not measured.

## 12. Configuration constants (`config.py`)

`PEAK_WINDOWS` 11-14 and 18-21 · `RECURRENCE_WINDOW_DAYS` 30 · `RECURRENCE_THRESHOLD` 2 · `TOP_DOCS` 5 ·
`MIN_PRIMARY_RUNBOOK_SCORE` 0.30 · `GROUNDING_MIN_OVERLAP_EXTRACTIVE` 0.60 · `GROUNDING_MIN_OVERLAP_LLM` 0.35 ·
`MIN_GROUNDED_STEPS` 2 · `ESCALATION_THRESHOLD` 0.50 · `LOW_CONFIDENCE` 0.55 · `LLM_MAX_TOKENS` 1,500.

## 13. Is it really "agents"?

Offline, the "agents" are **workflow components**: fixed order, no model choosing what to do next. They are called agents
because each has a defined role, its own tools (lookups) and its own limits. With a model, the reader and writer are the only
model-driven parts, and they are still single calls, not loops. The accurate name is an **agentic workflow**.
A true agent would let a model pick which tool to call next and iterate until done. We chose not to, for predictability,
testability and auditability, and because the brief asks how we know the system is safe. Say: "the reasoning that must be
reproducible is in code; the language work is delegated to a model; the loop is a possible phase-two step once
evaluation shows where flexibility earns its risk."

## 14. Scaling

**Measured** (one process, offline, this laptop-class sandbox): start-up 0.2 s, 10 ms per ticket, 37 MB of memory, 529 KB of
knowledge. Compute is not the bottleneck. Not load-tested; the demo web server uses a global lock and is not a production server.
The brief gives no ticket volume: ask the client.

**Does scaling need AI?** No. The offline path scales cheaply. The AI is the expensive, slow, rate-limited part, so scale
plans are about *using it selectively*:
* Call the reader only when rules see messy signals or low confidence; call the writer only for L1-guided tickets (measure first).
* Cache guidance by (runbook, tried steps, system).
* Smaller, faster model for reading; larger only for writing, if it earns it.
* Move from a free tier to a paid or private endpoint (Bedrock / Azure / Vertex) with the client's data terms.
* Queue-based workers with back-pressure; when limits hit, fall back to the offline path (already built per stage).

**Production architecture**: ITSM webhook -> queue -> stateless workers -> ITSM API for actions. Replace the CSV lookups with CMDB/ITSM
APIs plus caching; put the audit log in an append-only store; version the index by its content hash and rebuild on KB change; run
the regression evaluation in CI whenever runbooks or prompts change; partition knowledge by brand/region; enable autonomy per
incident type (shadow mode -> assisted -> automatic).

**More domains** = mostly data: add runbooks and re-index. Retrain the two learned parts when labels exist. New categories still
need some code (lexicons/patterns in triage) - it is not zero-code.

## 15. Known weaknesses (say them before you are asked)

1. Messy-ticket detectors are hand-tuned patterns; unseen phrasing may slip through. **(the AI reader is meant to help; unmeasured)**
2. Retrieval quality is measured at document level only.
3. Some chunks are huge tables or tiny fragments; no overlap.
4. Keyword search misses vocabulary mismatches ("till" vs "POS terminal").
5. The writer's evidence is not filtered by section type.
6. Extractive guidance can include a marginal step.
7. 23% false escalations; about a third of tickets need a human.
8. The peak-hours priority rule pushes many tickets to P1, which inflates review load.
9. Labels are noisy; the learned prior inherits their quirks.
10. The AI's benefit is unmeasured; provider adapters are untested against the live services.
11. Automations are simulated; there is no telemetry in the data.
12. The knowledge base itself has gaps and conflicts (no loyalty-sync runbook; EOL dates differ in three documents; the SLA matrix and
    the SOP disagree on L2 targets; runbooks cite "v8.x" while the fleet runs v4/v5).

## 16. Questions you will be asked

**Where is the AI?** Optional. Two calls per ticket: reader (after triage) and writer (during guidance). Everything else is code.
**Why not let the AI decide the route?** Auditability and variance. The model supplies evidence and an advisory opinion; the
policy decides and can be explained line by line.
**What if the AI lies?** Steps must cite retrieved chunks and every command/code/number must appear in the source; invented codes
are ignored and flagged; refund promises are rejected. Tested with a scripted hallucinating model.
**Prompt injection?** Ticket text is data. The reader/writer prompts say so; outputs are validated; injection-like text is flagged and
holds the reply; the model can't trigger actions.
**How do you know it's accurate?** Held-out evaluation: 84.8% routing vs 78.8% baseline, 89.5% escalation recall, plus the
messy suite. Caveats: noisy labels, small sample, my own messy-ticket expectations.
**Why BM25 and not embeddings?** Exact codes matter, the corpus is 529 KB, it is explainable and needs no service. Hybrid with a dense
re-ranker is the phase-one upgrade.
**How did you choose the 0.50 threshold?** Default operating point; cross-validation supports it. A missed escalation is costlier, so a
lower threshold is a business choice not yet measured.
**Why is P1 escalation reviewed by a human?** Pilot safety: a wrong page is expensive. Loosen per incident type once measured.
**What did you remove and why?** `resolution_notes` (answer leak) and the description-length feature (looked like a generator artefact).
**What is trained on your side?** Two small JSON models. Not the LLM.
**Why do some features have weight 0?** No variation in normal training tickets; hard rules cover those cases.
**Does it scale?** Compute yes (10 ms/ticket); AI is the constraint; see section 14. Not load-tested.
**What is simulated?** All actions and device telemetry.
**What would you do with more time?** Chunk-level evaluation, dense re-ranker, filter writer evidence, measure the AI's effect,
shadow-mode harness with real L1 feedback, live ITSM connector.
**Why 3 agents?** Different tools, permissions and failure modes per role. Honestly: offline they are workflow stages.
**What if the KB is wrong?** The agent cites sources so errors are traceable; conflicts (EOL dates) are flagged as findings to fix.
**What is human-in-the-loop here?** Approval for state-changing actions; review for P1 escalations, low confidence, injection, AI flags.

## 17. How to prove things live

* Show the trace: any ticket, "Trace" card (or `python -m servewell_agent demo INC-00162`).
* Show grounding blocking a lie: `python -m unittest tests.test_llm -v` (hallucination test) or `tests.test_agent`.
* Show the leak guard: the blue banner on the ticket card.
* Show rules vs AI: toggle **Use AI** on a messy ticket; open the "AI reader vs rules" table.
* Show the audit chain: Evaluation tab -> Audit log -> badge; edit-and-verify is in `tests/test_agent.py`.
* Step through the code: breakpoint at the first line of `SupportAgentSuite.process()`.
