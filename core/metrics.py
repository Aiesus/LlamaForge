"""
Token / context monitor.

Polls llama-server's /metrics (Prometheus) and /slots endpoints to surface,
without any tokenizer, how loaded the model's context is and how big the last
request was. All counting is done by the engine — this just reads it.

Adaptive cadence: polls fast while the server is generating, slow when idle.
No tkinter imports — communicates via a registered callback.
"""
from __future__ import annotations
import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable


@dataclass
class TokenStats:
    ok:          bool  = False   # server reachable
    metrics_on:  bool  = False   # /metrics endpoint returned data
    n_ctx:       int   = 0       # configured context window
    ctx_used:    int   = 0       # tokens currently in KV cache
    ctx_ratio:   float = 0.0     # ctx_used / n_ctx (0..1)
    last_prompt: int   = 0       # tokens in the most recent request's prompt
    n_decoded:   int   = 0       # tokens generated in the current/last turn
    processing:  bool  = False   # a generation is in flight


TokenCallback = Callable[[TokenStats], None]


def _parse_prometheus(text: str) -> dict:
    """Parse `metric_name value` lines (ignores # comments and labels)."""
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # name{labels} value   OR   name value
        try:
            name, _, rest = line.partition(" ")
            if "{" in name:
                name = name.split("{", 1)[0]
            out[name] = float(rest.strip())
        except (ValueError, AttributeError):
            continue
    return out


def _find(metrics: dict, needle: str) -> float | None:
    for k, v in metrics.items():
        if k.endswith(needle) or needle in k:
            return v
    return None


class TokenMonitor:
    """
    Daemon thread that polls llama-server and fires callbacks with TokenStats.

    port_fn / api_key_fn are callables read each cycle so the monitor tracks
    config changes (e.g. a profile switch) without being recreated.
    """

    def __init__(self, port_fn: Callable[[], object], api_key_fn: Callable[[], str],
                 host: str = "localhost",
                 busy_interval: float = 1.0, idle_interval: float = 5.0):
        self._port_fn       = port_fn
        self._api_key_fn    = api_key_fn
        self._host          = host
        self._busy_interval = busy_interval
        self._idle_interval = idle_interval

        self._callbacks: list[TokenCallback] = []
        self._running = False
        self._thread: threading.Thread | None = None

        # Cumulative prompt-token counter from the previous poll, for deltas.
        self._prev_prompt_total: float | None = None
        self._last_prompt = 0

    def register(self, cb: TokenCallback) -> None:
        self._callbacks.append(cb)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            stats = self._fetch()
            for cb in self._callbacks:
                try:
                    cb(stats)
                except Exception:
                    pass
            time.sleep(self._busy_interval if stats.processing else self._idle_interval)

    def _get(self, path: str, timeout: float = 2.0) -> str | None:
        port = self._port_fn()
        if not port:
            return None
        url = f"http://{self._host}:{port}{path}"
        req = urllib.request.Request(url)
        key = (self._api_key_fn() or "").strip()
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None

    def _fetch(self) -> TokenStats:
        st = TokenStats()

        # ── /slots: status, generated count, n_ctx ────────────────────────────
        slots_raw = self._get("/slots")
        if slots_raw is None:
            # Server unreachable — reset delta baseline so a restart doesn't
            # produce a bogus huge "last prompt".
            self._prev_prompt_total = None
            return st
        st.ok = True
        try:
            slots = json.loads(slots_raw)
            slot  = slots[0] if isinstance(slots, list) and slots else {}
            st.n_ctx      = int(slot.get("n_ctx", 0) or 0)
            st.processing = bool(slot.get("is_processing", False))
            nt = slot.get("next_token") or []
            if isinstance(nt, list) and nt:
                st.n_decoded = int(nt[0].get("n_decoded", 0) or 0)
        except Exception:
            pass

        # ── /metrics: prompt-token delta + KV-cache fill ──────────────────────
        metrics_raw = self._get("/metrics")
        if metrics_raw:
            m = _parse_prometheus(metrics_raw)
            if m:
                st.metrics_on = True

                prompt_total = _find(m, "prompt_tokens_total")
                if prompt_total is not None:
                    if self._prev_prompt_total is None or prompt_total < self._prev_prompt_total:
                        # First read or counter reset (server restart) — baseline only.
                        self._prev_prompt_total = prompt_total
                    else:
                        delta = prompt_total - self._prev_prompt_total
                        if delta > 0:
                            self._last_prompt = int(delta)
                        self._prev_prompt_total = prompt_total
                st.last_prompt = self._last_prompt

                kv_tokens = _find(m, "kv_cache_tokens")
                kv_ratio  = _find(m, "kv_cache_usage_ratio")
                if kv_tokens is not None:
                    st.ctx_used = int(kv_tokens)
                    st.ctx_ratio = (st.ctx_used / st.n_ctx) if st.n_ctx else 0.0
                elif kv_ratio is not None:
                    st.ctx_ratio = kv_ratio
                    st.ctx_used  = int(kv_ratio * st.n_ctx) if st.n_ctx else 0

        return st
