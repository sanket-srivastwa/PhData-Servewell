"""LLM access behind one small interface - standard library only (urllib), no SDK to install.

Providers
---------
  groq       OpenAI-compatible chat completions (free tier, no card, very fast)   env: GROQ_API_KEY
  gemini     Google AI Studio generateContent (free tier on Flash / Flash-Lite)   env: GEMINI_API_KEY (or GOOGLE_API_KEY)

Design rules
------------
* The whole pipeline must run - and be evaluable - with NO LLM (`NullLLM`): that is the CI mode and the
  live-demo fallback. Every LLM call site catches `LLMError` and falls back.
* The LLM is a swappable component behind `complete_json`. It only does language work (reading messy
  text into structured fields, rewriting grounded steps, drafting messages). Its output is *data* that
  is validated and guardrail-checked; it never triggers an action by itself.
* Free tiers rate-limit hard (requests/min AND tokens/min). We throttle between calls, honour
  `Retry-After`, retry a few times, and give up with a clear message rather than hanging the UI.
* API keys live only in process memory. They are never logged, never written to disk, and are scrubbed
  from every error message.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

from . import config


class LLMError(Exception):
    """Any failure talking to (or understanding) the model. Callers fall back to offline behaviour."""


class LLMHTTPError(LLMError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status, self.message = status, message


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
PHONE = re.compile(r"(?<!\w)\+?\d[\d\-\s()]{8,}\d(?!\w)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def redact(text: str) -> str:
    """Data minimisation before text leaves the machine (phones, e-mails)."""
    return EMAIL.sub("[email]", PHONE.sub("[phone]", text or ""))


def parse_json_loose(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        out = json.loads(m.group(0))
    if not isinstance(out, dict):
        raise ValueError("model returned JSON that is not an object")
    return out


class LLM:
    provider = "none"
    name = "none"
    available = False

    def __init__(self):
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0, "retries": 0, "errors": 0}
        self.calls: list = []

    def snapshot(self) -> dict:
        return dict(self.stats)

    def usage_since(self, before: dict) -> dict:
        return {k: self.stats[k] - before.get(k, 0) for k in self.stats}

    def _record(self, purpose: str, ms: int, usage: dict):
        self.stats["calls"] += 1
        self.stats["latency_ms"] += ms
        self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        self.calls.append({"purpose": purpose, "ms": ms, **usage})
        del self.calls[:-200]

    def complete_json(self, system: str, user: str, max_tokens: Optional[int] = None, purpose: str = "") -> dict:  # pragma: no cover
        raise NotImplementedError


class NullLLM(LLM):
    provider = "offline"
    name = "offline-extractive"
    available = False


class FakeLLM(LLM):
    """Scripted responses for tests / guardrail demos: responder(system, user) -> dict."""
    provider = "fake"
    name = "fake"
    available = True

    def __init__(self, responder: Callable[[str, str], dict]):
        super().__init__()
        self.responder = responder

    def complete_json(self, system: str, user: str, max_tokens: Optional[int] = None, purpose: str = "") -> dict:
        t0 = time.time()
        try:
            out = self.responder(system, user)
        except Exception as exc:                                 # a crashing model is an LLMError to callers
            self.stats["errors"] += 1
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        self._record(purpose, int((time.time() - t0) * 1000), {})
        return out


# ------------------------------------------------------------------------------------------------
# HTTP providers
# ------------------------------------------------------------------------------------------------
class HTTPLLM(LLM):
    available = True
    default_base = ""
    min_interval = 0.0          # seconds between calls (free-tier RPM protection)
    max_retries = 3
    max_wait = 45.0             # never sleep longer than this for a rate limit; fail with a clear message instead

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, min_interval: Optional[float] = None):
        super().__init__()
        if not api_key:
            raise LLMError(f"no API key for {self.provider}")
        self._key = api_key.strip()
        self.model = model
        self.name = f"{self.provider}/{model}" if model else self.provider
        self.base = (base_url or self.default_base).rstrip("/")
        if min_interval is not None:
            self.min_interval = min_interval
        self._last = 0.0
        self._sleep = time.sleep                 # patched in tests

    # ---- plumbing ----------------------------------------------------------------------------
    def _scrub(self, s: str) -> str:
        return (s or "").replace(self._key, "***")[:400]

    def _headers(self) -> dict:                  # pragma: no cover - overridden
        raise NotImplementedError

    def _throttle(self):
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()

    def _http(self, method: str, url: str, body: Optional[dict] = None, timeout: int = 60):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, e.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise LLMError(f"network error talking to {self.provider}: {self._scrub(str(e))}") from e

    @staticmethod
    def _error_text(raw: bytes) -> str:
        try:
            j = json.loads(raw.decode("utf-8", "replace"))
            err = j.get("error", j)
            if isinstance(err, dict):
                return str(err.get("message") or err)
            return str(err)
        except Exception:
            return raw.decode("utf-8", "replace")[:300]

    @staticmethod
    def _retry_after(hdrs: dict, msg: str) -> Optional[float]:
        v = hdrs.get("retry-after")
        if v:
            try:
                return float(v)
            except ValueError:
                pass
        m = re.search(r"try again in ([\d.]+)\s*(ms|s|m)", msg or "", re.I)
        if m:
            x = float(m.group(1))
            return x / 1000 if m.group(2).lower() == "ms" else x * 60 if m.group(2).lower() == "m" else x
        return None

    def _request(self, method: str, url: str, body: Optional[dict] = None):
        for attempt in range(self.max_retries + 1):
            self._throttle()
            status, hdrs, raw = self._http(method, url, body)
            if status == 200:
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise LLMError("provider returned a non-JSON response") from exc
            msg = self._scrub(self._error_text(raw))
            if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                if status == 429 and re.search(r"per ?day|daily|limit: 0", msg, re.I):
                    raise LLMHTTPError(status, "free-tier quota exhausted for this model (" + msg + "). Pick another model or wait for the reset.")
                wait = self._retry_after(hdrs, msg)
                wait = wait if wait is not None else 2.0 ** (attempt + 1)
                if wait > self.max_wait:
                    raise LLMHTTPError(status, f"rate limited; provider asks to wait {wait:.0f}s ({msg})")
                self.stats["retries"] += 1
                self._sleep(wait + 0.2)
                continue
            raise LLMHTTPError(status, msg)
        raise LLMError("gave up after retries")   # pragma: no cover

    # ---- public --------------------------------------------------------------------------------
    def _raw(self, system: str, user: str, max_tokens: int) -> tuple:   # pragma: no cover - provider specific
        raise NotImplementedError

    def list_models(self) -> list:                                        # pragma: no cover - provider specific
        raise NotImplementedError

    def complete_json(self, system: str, user: str, max_tokens: Optional[int] = None, purpose: str = "") -> dict:
        t0 = time.time()
        try:
            text, usage = self._raw(system, user, max_tokens or config.LLM_MAX_TOKENS)
        except LLMError:
            self.stats["errors"] += 1
            raise
        self._record(purpose, int((time.time() - t0) * 1000), usage)
        try:
            return parse_json_loose(text)
        except Exception as exc:
            self.stats["errors"] += 1
            raise LLMError("model did not return valid JSON") from exc

    def test(self) -> dict:
        t0 = time.time()
        out = self.complete_json("You reply with JSON only.", 'Reply with exactly this JSON object: {"ok": true}', max_tokens=60, purpose="connection_test")
        return {"ok": True, "ms": int((time.time() - t0) * 1000), "reply": out}


class GroqLLM(HTTPLLM):
    provider = "groq"
    default_base = "https://api.groq.com/openai/v1"
    min_interval = 2.2          # ~27 requests/min: under the 30 RPM free-tier ceiling

    def _headers(self):
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json", "User-Agent": "servewell-agent-poc/1.0"}

    def list_models(self) -> list:
        j = self._request("GET", f"{self.base}/models")
        skip = ("whisper", "guard", "tts", "playai", "orpheus", "embed", "distil-whisper", "compound")
        ids = [m["id"] for m in j.get("data", []) if m.get("active", True) and not any(s in m["id"].lower() for s in skip)]
        return sorted(ids)

    def _raw(self, system, user, max_tokens):
        body = {"model": self.model, "temperature": 0, "max_completion_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}}
        if self.model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"
        optional = ["response_format", "reasoning_effort"]
        while True:
            try:
                j = self._request("POST", f"{self.base}/chat/completions", body)
                break
            except LLMHTTPError as e:                     # provider/model doesn't support an optional knob -> drop it, retry once per knob
                low = e.message.lower()
                drop = next((k for k in optional if k in body and (k in low or (k == "response_format" and ("json" in low)))), None)
                if e.status == 400 and drop:
                    body.pop(drop)
                    continue
                raise
        try:
            ch = j["choices"][0]
            text = ch["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected response shape from Groq") from exc
        if not text.strip():
            raise LLMError("empty answer" + (" (output truncated - try another model)" if ch.get("finish_reason") == "length" else ""))
        u = j.get("usage") or {}
        return text, {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0)}


class GeminiLLM(HTTPLLM):
    provider = "gemini"
    default_base = "https://generativelanguage.googleapis.com/v1beta"
    min_interval = 4.5          # 15 RPM free-tier ceiling on Flash-Lite

    def _headers(self):
        return {"x-goog-api-key": self._key, "Content-Type": "application/json", "User-Agent": "servewell-agent-poc/1.0"}

    def list_models(self) -> list:
        j = self._request("GET", f"{self.base}/models?pageSize=200")
        skip = ("image", "tts", "audio", "live", "embedding", "robotics", "computer-use", "veo", "imagen", "aqa", "learnlm", "gemma")
        out = []
        for m in j.get("models", []):
            name = m.get("name", "").replace("models/", "")
            if "generateContent" in (m.get("supportedGenerationMethods") or []) and not any(s in name.lower() for s in skip):
                out.append(name)
        return sorted(out)

    def _raw(self, system, user, max_tokens):
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": max(max_tokens, 1500), "responseMimeType": "application/json"}}
        j = self._request("POST", f"{self.base}/models/{self.model}:generateContent", body)
        cands = j.get("candidates") or []
        if not cands:
            why = (j.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise LLMError(f"Gemini returned no answer ({why})")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise LLMError(f"empty answer (finishReason={cands[0].get('finishReason')}); try a different model")
        u = j.get("usageMetadata") or {}
        return text, {"prompt_tokens": u.get("promptTokenCount", 0), "completion_tokens": u.get("candidatesTokenCount", 0)}


# ------------------------------------------------------------------------------------------------
# registry / factory
# ------------------------------------------------------------------------------------------------
PROVIDERS = {
    "groq": {"cls": GroqLLM, "env": ["GROQ_API_KEY"], "label": "Groq (free)",
             "prefer": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]},
    "gemini": {"cls": GeminiLLM, "env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "label": "Google Gemini (free)", "prefer": []},
}


def env_key(provider: str) -> Optional[str]:
    for var in PROVIDERS[provider]["env"]:
        if os.environ.get(var):
            return os.environ[var]
    return None


def _version_key(name: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", name)[:4]) or (0,)


def pick_model(provider: str, available: list) -> Optional[str]:
    """Best default from what the account can actually use (model names churn; never hard-code one)."""
    if not available:
        return None
    for p in PROVIDERS[provider]["prefer"]:
        if p in available:
            return p
    if provider == "gemini":
        lite = [m for m in available if "flash-lite" in m and not re.search(r"preview|exp|latest", m)] or [m for m in available if "flash-lite" in m]
        flash = [m for m in available if "flash" in m and "lite" not in m and not re.search(r"preview|exp", m)]
        pool = lite or flash or available          # Flash-Lite first: the largest free daily allowance
        return sorted(pool, key=_version_key)[-1]
    return sorted(available)[0]


def make_llm(spec=None, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None,
             min_interval: Optional[float] = None) -> LLM:
    """spec: None/False/'offline' -> NullLLM; True/'auto' -> first provider with a key in the environment;
    'groq' | 'gemini' -> that provider (key from `api_key` or the environment)."""
    if spec in (None, False, "", "offline", "none"):
        return NullLLM()
    provider = spec
    if spec in (True, "auto"):
        provider = next((p for p in PROVIDERS if env_key(p)), None)
        if not provider:
            print("[warn] --llm requested but no GROQ_API_KEY / GEMINI_API_KEY found; using offline mode")
            return NullLLM()
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r} (choose groq, gemini)")
    key = api_key or env_key(provider)
    if not key:
        raise LLMError(f"no API key for {provider}: set {PROVIDERS[provider]['env'][0]} or paste it in the UI")
    cls = PROVIDERS[provider]["cls"]
    if not model:
        probe = cls(key, "", base_url, min_interval)
        model = pick_model(provider, probe.list_models())
        if not model:
            raise LLMError(f"could not find a usable model for {provider}")
    return cls(key, model, base_url, min_interval)
