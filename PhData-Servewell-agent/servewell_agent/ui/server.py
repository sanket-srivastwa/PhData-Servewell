"""Local web UI for the agent suite - standard library only (no Flask/Streamlit to install).

    python -m servewell_agent ui            # then open http://127.0.0.1:8000

The server binds to 127.0.0.1 only. It wraps the same `SupportAgentSuite` the CLI uses; nothing in the
agent logic is different in the UI. Endpoints:

    GET  /                     the single-page app (index.html)
    GET  /api/info             mode, KB version, counts
    GET  /api/samples          dataset + example tickets for the picker
    GET  /api/ticket/<id>      one ticket (answer fields already stripped by the leak guard)
    POST /api/process          run one ticket: {sample|ticket|text[,store]}
    GET  /api/result/<id>      last result for a ticket id
    POST /api/approve          simulate a human approving a pending action
    GET  /api/audit            last audit-log entries + hash-chain verification
    POST /api/run_eval         start the offline evaluation in the background
    GET  /api/eval             evaluation status / output / metrics
    POST /api/llm/connect      list the models a provider key can use   {provider, api_key?}
    POST /api/llm/use          switch the language model on            {provider, model, api_key?}
    POST /api/llm/off          back to deterministic offline mode
    POST /api/run_ab           start the AI-vs-rules comparison (background)
    GET  /api/ab               comparison status / metrics
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from .. import config
from ..agents.orchestrator import SupportAgentSuite
from ..ingest import load_ticket_dir, load_ticket_file, ticket_from_dict, ticket_from_text
from ..llm import PROVIDERS, LLMError, NullLLM, env_key, make_llm, pick_model

HERE = Path(__file__).resolve().parent
EXAMPLES = config.PROJECT_ROOT / "examples"


class App:
    def __init__(self, data_root: Optional[Path] = None, llm_spec=None, model: Optional[str] = None):
        self.root = Path(data_root) if data_root else config.find_data_root()
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        self.llm_error = None
        try:
            llm = make_llm(llm_spec, model=model)
        except LLMError as exc:
            llm, self.llm_error = NullLLM(), str(exc)
        self.suite = SupportAgentSuite(self.root, llm=llm, audit_path=config.OUTPUT_DIR / "ui_audit.jsonl")
        self._pending_keys: dict = {}              # provider -> key typed in the UI; memory only, never persisted or echoed
        self.ab_state = {"running": False, "done": 0, "total": 0, "current": "", "error": None}
        self.lock = threading.Lock()
        self.results: dict = {}                    # ticket_id -> (ticket, result)
        self.samples: dict = {}                    # sample id -> ticket
        self.sample_meta: list = []
        self.labels = {}
        lp = self.root / "labels" / "train_labels.json"
        if lp.exists():
            self.labels = {x["ticket_id"]: x for x in json.loads(lp.read_text(encoding="utf-8"))}
        self.chaos = {}
        ip = self.root / "tickets" / "train_index.csv"
        if ip.exists():
            with open(ip, newline="", encoding="utf-8-sig") as f:
                self.chaos = {r["ticket_id"]: r["chaos_type"] for r in csv.DictReader(f)}
        self._load_samples()
        self.eval_state = {"running": False, "output": "", "error": None}

    # ---------------------------------------------------------------- samples
    def _load_samples(self):
        if EXAMPLES.is_dir():
            for p in sorted(EXAMPLES.iterdir()):
                if p.suffix.lower() in (".json", ".txt"):
                    t = load_ticket_file(p)
                    sid = f"ex:{p.name}"
                    self.samples[sid] = t
                    self.sample_meta.append({"id": sid, "group": "fresh", "ticket_id": t.ticket_id, "subject": t.subject[:90],
                                             "category": t.category or "(free text)", "subcategory": t.subcategory, "priority": t.priority,
                                             "kind": "Fresh example"})
        d = self.root / "tickets" / "train"
        if d.is_dir():
            for t in load_ticket_dir(d):
                self.samples[t.ticket_id] = t
                chaos = self.chaos.get(t.ticket_id, "")
                self.sample_meta.append({"id": t.ticket_id, "group": chaos or "clean", "ticket_id": t.ticket_id, "subject": t.subject[:90],
                                         "category": t.category, "subcategory": t.subcategory, "priority": t.priority,
                                         "kind": ("Messy: " + chaos) if chaos else "Clean"})

    # ---------------------------------------------------------------- views
    def ticket_view(self, t) -> dict:
        return {"ticket_id": t.ticket_id, "store_id": t.store_id, "store_name": t.store_name, "submitted_by": t.submitted_by,
                "date_opened": t.date_opened.isoformat() if t.date_opened else None, "priority": t.priority,
                "category": t.category, "subcategory": t.subcategory, "asset_id": t.asset_id, "system_version": t.system_version,
                "subject": t.subject, "description": t.description, "history": t.ticket_history,
                "escalation_flag": t.escalation_flag, "leak_guard_dropped": t.leak_guard_dropped,
                "missing_fields": t.raw_missing_fields}

    def label_for(self, tid: str, route: str) -> Optional[dict]:
        g = self.labels.get(tid)
        if not g:
            return None
        return {"gold": g["correct_routing"], "match": g["correct_routing"] == route,
                "note": "Training label. The shipped model was fit on these tickets, so this is not a held-out test."}

    # ---------------------------------------------------------------- actions
    def process(self, body: dict) -> dict:
        if "sample" in body:
            t = self.samples.get(body["sample"])
            if not t:
                raise KeyError(f"unknown sample {body['sample']}")
        elif "ticket" in body:
            t = ticket_from_dict(body["ticket"], strip_status=True)
        else:
            text = (body.get("text") or "").strip()
            if not text:
                raise ValueError("empty ticket")
            t = ticket_from_text(text, ticket_id=body.get("ticket_id") or "INC-UI-0001", store_id=(body.get("store") or None))
        if self.ab_state["running"]:
            raise ValueError("the AI-vs-rules comparison is running; wait for it to finish (or refresh the page later)")
        use_llm = body.get("use_llm")
        with self.lock:
            res = self.suite.process(t, use_llm=use_llm)
            self.results[t.ticket_id] = (t, res)
        return self._package(t, res)

    def _package(self, t, res) -> dict:
        return {"ticket": self.ticket_view(t), "result": res.to_dict(), "label": self.label_for(t.ticket_id, res.decision.route),
                "audit_intact": self.suite.audit.verify(), "kb_hash": self.suite.kb.kb_hash, "mode": res.mode}

    def result(self, tid: str) -> dict:
        if tid not in self.results:
            raise KeyError(tid)
        t, res = self.results[tid]
        return self._package(t, res)

    def approve(self, body: dict) -> dict:
        tid, idx = body["ticket_id"], int(body["index"])
        t, res = self.results[tid]
        act = res.actions[idx]
        with self.lock:
            self.suite.itsm.execute(tid, act, approved=True, guardrails_passed=True)
        from dataclasses import asdict
        return {"action": asdict(act), "audit_intact": self.suite.audit.verify()}

    def audit(self) -> dict:
        return {"intact": self.suite.audit.verify(), "entries": self.suite.audit.entries[-60:][::-1]}

    def info(self) -> dict:
        p = config.OUTPUT_DIR / "eval_metrics.json"
        llm = self.suite.llm
        return {"mode": llm.name, "llm": llm.available, "provider": llm.provider, "kb_hash": self.suite.kb.kb_hash,
                "docs": len(self.suite.kb.docs), "chunks": len(self.suite.kb.chunks), "samples": len(self.sample_meta),
                "has_eval": p.exists(), "data_root": str(self.root), "llm_error": self.llm_error,
                "llm_usage": llm.snapshot(),
                "providers": [{"id": k, "label": v["label"], "env_key": bool(env_key(k)), "env_var": v["env"][0]} for k, v in PROVIDERS.items()]}

    # ---------------------------------------------------------------- language model settings
    def _key_for(self, provider: str, given: Optional[str]) -> str:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        key = (given or "").strip() or self._pending_keys.get(provider) or env_key(provider)
        if not key:
            raise ValueError(f"no API key for {provider}: paste one, or set {PROVIDERS[provider]['env'][0]} before starting the UI")
        return key

    def llm_connect(self, body: dict) -> dict:
        provider = body.get("provider", "")
        key = self._key_for(provider, body.get("api_key"))
        cls = PROVIDERS[provider]["cls"]
        try:
            probe = cls(key, "", body.get("base_url") or None, 0)
            models = probe.list_models()
        except LLMError as exc:
            raise ValueError(str(exc))
        self._pending_keys[provider] = key
        return {"models": models, "suggested": pick_model(provider, models)}

    def llm_use(self, body: dict) -> dict:
        provider, model = body.get("provider", ""), (body.get("model") or "").strip()
        key = self._key_for(provider, body.get("api_key"))
        try:
            llm = make_llm(provider, model=model or None, api_key=key, base_url=body.get("base_url") or None, min_interval=body.get("min_interval"))
            check = llm.test()
        except LLMError as exc:
            raise ValueError(str(exc))
        self._pending_keys[provider] = key
        with self.lock:
            self.suite.set_llm(llm)
        self.llm_error = None
        return {"ok": True, "name": llm.name, "test_ms": check["ms"]}

    def llm_off(self) -> dict:
        with self.lock:
            self.suite.set_llm(None)
        return {"ok": True}

    # ---------------------------------------------------------------- AI vs rules (background)
    def run_ab(self, body: dict) -> dict:
        if self.ab_state["running"]:
            return {"started": False, "reason": "already running"}
        if not self.suite.llm.available:
            raise ValueError("connect a language model first (AI settings, top right)")
        n = max(1, min(int(body.get("max") or 12), 36))
        self.ab_state.update(running=True, done=0, total=n, current="", error=None)

        def work():
            try:
                import sys
                sys.path.insert(0, str(config.PROJECT_ROOT))
                from evaluation.run_llm_ab import run_ab
                m = run_ab(self.suite, max_messy=n, normal_n=int(body.get("normal") or 0),
                           progress=lambda d, t, i: self.ab_state.update(done=d, total=t, current=i))
                (config.OUTPUT_DIR / "ab_metrics.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
            except Exception:                     # noqa: BLE001 - surfaced to the UI
                self.ab_state["error"] = traceback.format_exc()[-1200:]
            finally:
                self.ab_state["running"] = False

        threading.Thread(target=work, daemon=True).start()
        return {"started": True, "total": n}

    def ab_status(self) -> dict:
        p = config.OUTPUT_DIR / "ab_metrics.json"
        m = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        return {**self.ab_state, "metrics": m}

    # ---------------------------------------------------------------- evaluation (background)
    def run_eval(self) -> dict:
        if self.eval_state["running"]:
            return {"started": False, "reason": "already running"}
        self.eval_state.update(running=True, output="", error=None)

        def work():
            buf = io.StringIO()
            try:
                import sys
                sys.path.insert(0, str(config.PROJECT_ROOT))
                from evaluation.run_eval import main as eval_main
                with contextlib.redirect_stdout(buf):
                    eval_main(holdout=0.3, seed=7, seeds=5)
            except Exception:                      # noqa: BLE001 - surface any failure to the UI
                self.eval_state["error"] = traceback.format_exc()
            finally:
                self.eval_state["output"] = buf.getvalue()
                self.eval_state["running"] = False

        threading.Thread(target=work, daemon=True).start()
        return {"started": True}

    def eval_status(self) -> dict:
        p = config.OUTPUT_DIR / "eval_metrics.json"
        metrics = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        return {**self.eval_state, "metrics": metrics}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ServeWellUI/1.0"

        def log_message(self, fmt, *args):          # keep the terminal quiet
            pass

        def _send(self, code: int, payload, ctype: str = "application/json"):
            body = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") or ctype == "application/json" else ""))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

        def do_GET(self):
            path = unquote(self.path.split("?")[0])
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, (HERE / "index.html").read_bytes(), "text/html")
                if path == "/api/info":
                    return self._send(200, app.info())
                if path == "/api/samples":
                    return self._send(200, app.sample_meta)
                if path.startswith("/api/ticket/"):
                    t = app.samples.get(path.split("/api/ticket/")[1])
                    return self._send(200, app.ticket_view(t)) if t else self._send(404, {"error": "not found"})
                if path.startswith("/api/result/"):
                    return self._send(200, app.result(path.split("/api/result/")[1]))
                if path == "/api/audit":
                    return self._send(200, app.audit())
                if path == "/api/eval":
                    return self._send(200, app.eval_status())
                if path == "/api/ab":
                    return self._send(200, app.ab_status())
                return self._send(404, {"error": "not found"})
            except KeyError as exc:
                return self._send(404, {"error": f"not found: {exc}"})
            except Exception as exc:               # noqa: BLE001
                traceback.print_exc()
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                body = self._body()
                if path == "/api/process":
                    return self._send(200, app.process(body))
                if path == "/api/approve":
                    return self._send(200, app.approve(body))
                if path == "/api/run_eval":
                    return self._send(200, app.run_eval())
                if path == "/api/llm/connect":
                    return self._send(200, app.llm_connect(body))
                if path == "/api/llm/use":
                    return self._send(200, app.llm_use(body))
                if path == "/api/llm/off":
                    return self._send(200, app.llm_off())
                if path == "/api/run_ab":
                    return self._send(200, app.run_ab(body))
                return self._send(404, {"error": "not found"})
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                return self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:               # noqa: BLE001
                traceback.print_exc()
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(port: int = 8000, open_browser: bool = True, data_root: Optional[Path] = None, llm_spec=None, model: Optional[str] = None):
    app = App(data_root, llm_spec, model)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{port}"
    print(f"ServeWell agent UI running at {url}   (mode: {app.suite.llm.name}, KB {app.suite.kb.kb_hash}, {len(app.sample_meta)} sample tickets)")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
