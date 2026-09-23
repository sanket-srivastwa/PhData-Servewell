"""A tiny local HTTP server that mimics the three provider APIs (request/response *shapes*), so the adapters
and the whole LLM pipeline can be tested without keys or network. It is NOT a model: it derives its answers
from the request with simple heuristics, and can be told to misbehave (hallucinate, rate-limit, reject params).

Shapes are taken from the providers' public API docs; they have not been verified against the live services
from this sandbox.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GOOD_KEY = "test-key-123"


class State:
    mode = "ok"                # ok | hallucinate | bad_json | fail500
    rate_limit_once = False    # next chat call returns 429 with Retry-After: 0
    reject_response_format = False
    seen = []                  # (path, headers, body) of every request


def _reader_answer(case: dict) -> dict:
    tk = case["ticket"]
    text = (tk["subject"] + " " + tk["description"])
    low = text.lower()
    tried = []
    if re.search(r"restart|reboot|power cycl", low):
        tried.append("restart_device")
    codes = re.findall(r"0x[0-9A-Fa-f]{3,8}|[A-Z]{2,4}-\d{3,4}|\bE\d{2}\b", text)
    angry = any(w in text for w in ("DONE", "fed up", "FED UP", "pathetic", "ridiculous"))
    wrong = ("router" in low and "off" in low and tk["filed_category"] != "Wi-Fi / Network")
    route = "l2_escalation" if (tk["l1_history"] or angry) else "l1_guided"
    out = {
        "issue_summary": " ".join(tk["description"].split())[:220] or tk["subject"],
        "true_domain": "Wi-Fi / Network" if wrong else (tk["filed_category"] if tk["filed_category"] in case["allowed_domains"] else "POS"),
        "wrong_category": wrong,
        "sentiment": "angry" if angry else "calm",
        "asks_for_manager_or_escalation": "manager" in low,
        "steps_already_tried": tried,
        "error_codes": codes[:2] + (["ZZ-9999"] if State.mode == "hallucinate" else []),
        "non_it": {"present": "salary" in low, "dominant": False, "kind": "payroll" if "salary" in low else "none"},
        "specificity": "vague" if len(tk["description"].split()) < 12 else "specific",
        "missing_info": [],
        "suggested_route": route,
        "route_reason": "mock heuristic",
    }
    return out


def _writer_answer(case: dict) -> dict:
    ev = case["evidence"]
    steps = []
    for c in ev[:3]:
        body = " ".join(c["text"].split())[:200]
        steps.append({"title": c["source"].split(">")[-1].strip()[:60], "text": body, "sources": [c["id"]]})
    msg = "Hi {{NAME}}, here is what to try, in order. If it still fails, reply and we will escalate."
    if State.mode == "hallucinate":
        steps.append({"title": "Invented", "text": "Run `regedit /s hack.reg` and set 0x1F.", "sources": [ev[0]["id"]]})
        steps.append({"title": "Phantom", "text": "Replace the thermal head.", "sources": ["runbooks/nope.md#x"]})
        msg = "Hi {{NAME}}, we guarantee a full refund within 10 minutes."
    return {"steps": steps, "requester_message": msg, "l2_summary": "Mock summary of the issue for the L2 engineer, with asset and what was tried."}


def _answer_text(system: str, user: str) -> str:
    m = re.search(r"<case>\n(.*)\n</case>", user, re.S)
    if not m:
        return json.dumps({"ok": True})
    case = json.loads(m.group(1))
    if State.mode == "bad_json":
        return "Sure! Here you go: {not json"
    out = _reader_answer(case) if case.get("task") == "read_ticket" else _writer_answer(case)
    return json.dumps(out)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self):
        h = self.headers
        key = (h.get("Authorization", "").replace("Bearer ", "") or h.get("x-goog-api-key") or h.get("x-api-key") or "")
        return key == GOOD_KEY, key

    def do_GET(self):
        ok, key = self._auth_ok()
        State.seen.append(("GET " + self.path, dict(self.headers), None))
        if not ok:
            return self._send(401, {"error": {"message": f"Invalid API key {key}"}})
        if self.path.startswith("/openai/v1/models"):
            return self._send(200, {"data": [{"id": "llama-3.3-70b-versatile", "active": True}, {"id": "whisper-large-v3", "active": True},
                                             {"id": "openai/gpt-oss-20b", "active": True}, {"id": "meta-llama/llama-guard-4-12b", "active": True}]})
        if self.path.startswith("/v1beta/models"):
            return self._send(200, {"models": [
                {"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-3.5-flash-lite-preview-09-2026", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
                {"name": "models/gemini-3.5-flash-image", "supportedGenerationMethods": ["generateContent"]}]})
        if self.path.startswith("/v1/models"):
            return self._send(200, {"data": [{"id": "claude-sonnet-5"}, {"id": "claude-haiku-4-5-20251001"}]})
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        State.seen.append(("POST " + self.path, dict(self.headers), body))
        ok, key = self._auth_ok()
        if not ok:
            return self._send(401, {"error": {"message": f"Invalid API key {key}"}})
        if State.mode == "fail500":
            return self._send(500, {"error": {"message": "boom"}})
        if State.rate_limit_once:
            State.rate_limit_once = False
            return self._send(429, {"error": {"message": "Rate limit reached. Please try again in 0.01s"}}, {"Retry-After": "0"})
        if self.path.endswith("/chat/completions"):
            if State.reject_response_format and "response_format" in body:
                return self._send(400, {"error": {"message": "response_format is not supported with this model"}})
            text = _answer_text(body["messages"][0]["content"], body["messages"][1]["content"])
            return self._send(200, {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 500, "completion_tokens": 120}})
        if ":generateContent" in self.path:
            text = _answer_text(body["systemInstruction"]["parts"][0]["text"], body["contents"][0]["parts"][0]["text"])
            return self._send(200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
                                    "usageMetadata": {"promptTokenCount": 400, "candidatesTokenCount": 100}})
        if self.path.endswith("/messages"):
            text = _answer_text(body["system"], body["messages"][0]["content"])
            return self._send(200, {"content": [{"type": "text", "text": text}], "usage": {"input_tokens": 450, "output_tokens": 110}})
        self._send(404, {"error": {"message": "not found"}})


def start():
    """returns (server, base_url)"""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
