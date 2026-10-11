"""Gemini client built for the free tier.

Three things matter on the free tier and all three live here:
  * fallback   - every (key, model) pair has its own quota; a 429 moves on to the next pair
  * pacing     - a per-pair sliding-window limiter keeps us under the requests-per-minute cap
  * caching    - identical prompts are answered from a local SQLite cache, so re-running the
                 evaluation (or a demo question) costs no quota and gives identical output
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass

from google import genai
from google.genai import types

from .config import get_settings

log = logging.getLogger("analyst.llm")


class LLMUnavailable(RuntimeError):
    """Every key and model failed or is out of quota."""


@dataclass
class LLMResult:
    text: str
    model: str            # "gemini-3.5-flash-lite@key1", or "cache"
    cached: bool
    seconds: float

    def json(self) -> dict:
        return parse_json(self.text)


def parse_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating ``` fences and leading prose."""
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip()).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise


class _RateLimiter:
    """Sliding 60-second window per (key, model), plus error blocks (429s, timeouts)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.calls: dict[str, deque] = {}
        self.blocked_until: dict[str, float] = {}
        self.lock = threading.Lock()

    def _window(self, pair: str, now: float) -> float:
        q = self.calls.setdefault(pair, deque())
        while q and now - q[0] > 60:
            q.popleft()
        return 0.0 if len(q) < self.per_minute else 60 - (now - q[0])

    def blocked(self, pair: str) -> float:
        with self.lock:
            return max(self.blocked_until.get(pair, 0) - time.monotonic(), 0.0)

    def try_acquire(self, pair: str) -> bool:
        """Take a slot if the window has room and the pair is not blocked (atomic across threads)."""
        now = time.monotonic()
        with self.lock:
            if self.blocked_until.get(pair, 0) > now or self._window(pair, now) > 0:
                return False
            self.calls[pair].append(now)
            return True

    def ready_in(self, pair: str) -> float:
        now = time.monotonic()
        with self.lock:
            return max(self._window(pair, now), self.blocked_until.get(pair, 0) - now, 0.0)

    def block(self, pair: str, seconds: float) -> None:
        with self.lock:
            self.blocked_until[pair] = time.monotonic() + seconds


class _Cache:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("create table if not exists llm(k text primary key, text text, model text, ts real)")
        self.lock = threading.Lock()

    def get(self, k: str):
        with self.lock:
            return self.db.execute("select text, model from llm where k=?", (k,)).fetchone()

    def put(self, k: str, text: str, model: str) -> None:
        with self.lock:
            self.db.execute("insert or replace into llm values(?,?,?,?)", (k, text, model, time.time()))
            self.db.commit()


def _retry_after(err: Exception) -> float:
    """Seconds the API asks us to wait: "retryDelay: 41s" (per-minute limit) or "retry in 6h38m13s" (daily
    limit, which resets at midnight Pacific time). Daily limits block that key/model pair for hours."""
    msg = str(err)
    m = re.search(r"retry in\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", msg, re.I)
    if m and any(m.groups()):
        h, mi, se = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mi * 60 + se + 1
    m = re.search(r"retryDelay\W+(\d+(?:\.\d+)?)s", msg, re.I)
    if m:
        return float(m.group(1)) + 1
    if re.search(r"PerDay|per day|daily", msg, re.I):
        return 3600.0
    return 30.0


def _thinking(model: str, level: str):
    if model.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level=level)
    if model.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=0 if level == "minimal" else 1024)
    return None


class Gemini:
    def __init__(self, settings=None):
        self.s = settings or get_settings()
        if not self.s.api_keys:
            raise LLMUnavailable("No API key found. Put GEMINI_API_KEY in .env (see .env.example).")
        self.clients = [genai.Client(api_key=k) for k in self.s.api_keys]
        self.limiter = _RateLimiter(self.s.requests_per_minute)
        self.cache = _Cache(self.s.cache_dir / "llm_cache.sqlite")
        self.calls = 0          # real API calls made by this instance (cache hits excluded)

    def generate(self, prompt: str, system: str = "", *, json_mode: bool = False, temperature: float = 0.0,
                 thinking: str = "low", use_cache: bool = True, max_wait_s: float = 300) -> LLMResult:
        """Model preference is strict: a busy (rate-window full) model is *waited for*, not skipped.
        We only fall back to the next model when every key for the current one is erroring or out of quota.
        That keeps answers comparable: one model answers everything unless it is truly unavailable."""
        base = json.dumps([system, prompt, json_mode, temperature, thinking])
        ckey = lambda model: hashlib.sha256((base + model).encode()).hexdigest()  # noqa: E731
        if use_cache:
            for model in self.s.models:
                if hit := self.cache.get(ckey(model)):
                    return LLMResult(hit[0], "cache", True, 0.0)

        deadline = time.monotonic() + max_wait_s
        last: Exception | None = None
        while time.monotonic() < deadline:
            for model in self.s.models:
                pairs = [(f"{model}@key{i + 1}", c) for i, c in enumerate(self.clients)]
                if all(self.limiter.blocked(label) > 0 for label, _ in pairs):
                    continue                      # every key failing for this model: fall back to the next
                for label, client in pairs:
                    if not self.limiter.try_acquire(label):
                        continue
                    t0 = time.monotonic()
                    try:
                        r = client.models.generate_content(
                            model=model, contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system or None, temperature=temperature,
                                response_mime_type="application/json" if json_mode else None,
                                thinking_config=_thinking(model, thinking),
                                http_options=types.HttpOptions(timeout=60_000)))
                        text = (r.text or "").strip()
                        if not text:
                            raise ValueError("empty response")
                        self.calls += 1
                        self.cache.put(ckey(model), text, label)
                        return LLMResult(text, label, False, time.monotonic() - t0)
                    except Exception as e:  # noqa: BLE001 - any failure blocks this pair for a while
                        last = e
                        is_quota = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                        self.limiter.block(label, _retry_after(e) if is_quota else 5)
                        log.warning("%s failed (blocked %.0fs): %s", label, self.limiter.blocked(label), str(e)[:1500])
                # this model is usable but its keys are busy right now: wait for it rather than fall back
                nap = min(self.limiter.ready_in(label) for label, _ in pairs)
                break
            else:
                nap = min(self.limiter.ready_in(f"{m}@key{i + 1}") for m in self.s.models
                          for i in range(len(self.clients)))
            if time.monotonic() + nap > deadline:
                break
            time.sleep(min(max(nap, 0.2), 5))
        raise LLMUnavailable(f"All Gemini keys/models failed or are rate-limited. Last error: {last}")
