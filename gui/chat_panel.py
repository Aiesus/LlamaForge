"""
Chat tab — lives in the main notebook.
Streams from /v1/chat/completions using the currently loaded model.
"""
from __future__ import annotations
import json
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gui.app import AppState
import urllib.request


class ChatPanel:

    def __init__(self, parent: tk.Widget, state: AppState, T: dict,
                 log_fn: Callable):
        self._state   = state
        self._T       = T
        self._log     = log_fn

        self._messages:  list[dict] = []
        self._streaming: bool       = False
        self._connected: bool       = False
        self._stop_evt              = threading.Event()

        self.frame = tk.Frame(parent, bg=T["bg2"])
        self.frame.pack(fill="both", expand=True)
        self._build()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        T = self._T

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self.frame, bg=T["bg2"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="CHAT", bg=T["bg2"], fg=T["accent"],
                 font=("Consolas", 8, "bold")).pack(side="left", padx=8, pady=(6, 2))

        ctrl = tk.Frame(hdr, bg=T["bg2"])
        ctrl.pack(side="right", padx=8, pady=(4, 0))
        tk.Button(ctrl, text="New Chat", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 8),
                  command=self._new_chat).pack(side="left", padx=2)

        # ── System prompt (collapsible) ────────────────────────────────────
        self._sys_open = False
        self._sys_toggle = tk.Button(
            self.frame, text="▶  System prompt",
            bg=T["bg2"], fg=T["fg2"], relief="flat", cursor="hand2",
            font=("Segoe UI", 8), anchor="w",
            command=self._toggle_sys,
        )
        self._sys_toggle.pack(fill="x", padx=8, pady=(0, 2))

        self._sys_frame = tk.Frame(self.frame, bg=T["bg2"])
        self._sys_text = tk.Text(
            self._sys_frame, height=3,
            bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
            font=("Consolas", 9), wrap="word",
            insertbackground=T["fg"], padx=6, pady=4,
        )
        self._sys_text.pack(fill="x", padx=8, pady=(0, 4))

        # ── Send row (packed before history so it always stays visible) ────
        btn_row = tk.Frame(self.frame, bg=T["bg2"])
        btn_row.pack(side="bottom", fill="x", padx=8, pady=(4, 8))

        self._send_btn = tk.Button(
            btn_row, text="⬆  Send  ↵",
            bg=T["accent"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9, "bold"), pady=3,
            command=self._send, state="disabled",
        )
        self._send_btn.pack(side="left")

        self._stop_btn = tk.Button(
            btn_row, text="■  Stop",
            bg=T["red"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9), pady=3,
            command=self._stop, state="disabled",
        )
        self._stop_btn.pack(side="left", padx=6)

        self._status_lbl = tk.Label(
            btn_row, text="Server not loaded",
            bg=T["bg2"], fg=T["fg2"], font=("Segoe UI", 8),
        )
        self._status_lbl.pack(side="right")

        # ── Input area (packed before history, anchored to bottom) ────────
        # Accent-colored top border makes the input box easy to find
        tk.Frame(self.frame, bg=T["accent"], height=2).pack(
            side="bottom", fill="x", padx=8)
        inp_wrap = tk.Frame(self.frame, bg=T["entry_bg"])
        inp_wrap.pack(side="bottom", fill="x", padx=8)
        tk.Label(inp_wrap, text="Message  (Shift+↵ for newline)",
                 bg=T["entry_bg"], fg=T["fg2"],
                 font=("Consolas", 7)).pack(anchor="w", padx=6, pady=(4, 0))
        self._input = tk.Text(
            inp_wrap, height=3,
            bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
            font=("Segoe UI", 10), wrap="word",
            insertbackground=T["fg"], padx=6, pady=4,
        )
        self._input.pack(fill="x", padx=2, pady=(0, 4))
        self._input.bind("<Return>",       self._on_ctrl_enter)
        self._input.bind("<Shift-Return>", lambda e: self._input.insert("insert", "\n") or "break")

        # ── History display ────────────────────────────────────────────────
        hist_wrap = tk.Frame(self.frame, bg=T["log_bg"])
        hist_wrap.pack(fill="both", expand=True, padx=8, pady=(0, 2))
        vsb = ttk.Scrollbar(hist_wrap, orient="vertical")
        vsb.pack(side="right", fill="y")
        self._hist = tk.Text(
            hist_wrap,
            bg=T["log_bg"], fg=T["fg"],
            font=("Segoe UI", 10), relief="flat", wrap="word",
            state="disabled", padx=10, pady=6,
            yscrollcommand=vsb.set,
        )
        self._hist.pack(side="left", fill="both", expand=True)
        vsb.config(command=self._hist.yview)

        self._hist.tag_config("lbl_user", foreground=T["accent"],
                              font=("Segoe UI", 8, "bold"))
        self._hist.tag_config("lbl_ai",   foreground=T["green"],
                              font=("Segoe UI", 8, "bold"))
        self._hist.tag_config("user",      foreground=T["fg"])
        self._hist.tag_config("assistant", foreground=T["fg"])
        self._hist.tag_config("dim",       foreground=T["fg2"])

    # ── System prompt toggle ──────────────────────────────────────────────────

    def _toggle_sys(self) -> None:
        self._sys_open = not self._sys_open
        if self._sys_open:
            self._sys_frame.pack(fill="x", after=self._sys_toggle)
            self._sys_toggle.config(text="▼  System prompt")
        else:
            self._sys_frame.pack_forget()
            self._sys_toggle.config(text="▶  System prompt")

    # ── State control ─────────────────────────────────────────────────────────

    def set_connected(self, connected: bool) -> None:
        self._connected = connected
        if connected:
            self._send_btn.config(state="normal")
            self._status_lbl.config(text="")
        else:
            self._send_btn.config(state="disabled")
            self._status_lbl.config(text="Server not loaded")
            if self._streaming:
                self._stop_evt.set()

    # ── Chat actions ──────────────────────────────────────────────────────────

    def _new_chat(self) -> None:
        # Keep any system prompt; clear conversation turns only
        self._messages = [m for m in self._messages if m["role"] == "system"]
        self._hist.config(state="normal")
        self._hist.delete("1.0", tk.END)
        self._hist.config(state="disabled")
        self._status_lbl.config(text="")
        if self._streaming:
            self._stop_evt.set()

    def _on_ctrl_enter(self, event) -> str:
        self._send()
        return "break"

    def _send(self) -> None:
        text = self._input.get("1.0", tk.END).strip()
        if not text or self._streaming or not self._connected:
            return

        sys_text = self._sys_text.get("1.0", tk.END).strip()

        # Build message list: keep existing history, append new user turn.
        # Only include system message when content is non-empty — empty string
        # wastes prompt tokens and can confuse some models.
        if self._messages and self._messages[0]["role"] == "system":
            if sys_text:
                self._messages[0]["content"] = sys_text   # update in place
            else:
                self._messages.pop(0)                     # remove now-empty system msg
        elif sys_text:
            self._messages.insert(0, {"role": "system", "content": sys_text})

        self._messages.append({"role": "user", "content": text})
        self._input.delete("1.0", tk.END)

        self._append_hist("You", text, "lbl_user", "user")

        self._streaming    = True
        self._token_count  = 0
        self._stream_t0    = time.monotonic()
        self._stop_evt.clear()
        self._send_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._status_lbl.config(text="Generating…")

        # Insert AI label and mark where tokens will be appended
        self._hist.config(state="normal")
        self._hist.insert(tk.END, "\nAI\n", "lbl_ai")
        self._stream_start = self._hist.index(tk.END)
        self._hist.config(state="disabled")
        self._hist.see(tk.END)

        threading.Thread(
            target=self._stream_worker,
            args=(list(self._messages),),
            daemon=True,
        ).start()

    def _stop(self) -> None:
        self._stop_evt.set()

    # ── Streaming ─────────────────────────────────────────────────────────────

    _PAINT_MS = 0.05   # batch tokens for 50ms before flushing to GUI

    def _stream_worker(self, messages: list) -> None:
        s   = self._state.settings
        key = self._state.api_key_server_var.get().strip()
        if s.proxy_enabled and not self._state.proxy_bypass_var.get():
            base = "http://localhost:8088"
        else:
            base = f"http://localhost:{self._state.port_var.get()}"

        body: dict = {
            "model":       self._state.model_var.get().split("/")[-1],
            "messages":    messages,
            "stream":      True,
            "temperature": round(self._state.temp_var.get(), 3),
            "top_p":       round(self._state.top_p_var.get(), 3),
        }
        payload = json.dumps(body).encode()

        headers = {
            "Content-Type":  "application/json",
            "Accept":        "text/event-stream",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"

        tokens:   list[str] = []
        pending:  list[str] = []
        error_msg: str      = ""
        last_paint          = time.monotonic()

        def _flush(final_tps: float = 0.0) -> None:
            if not pending:
                return
            batch = "".join(pending)
            pending.clear()
            elapsed = time.monotonic() - self._stream_t0
            tps = self._token_count / elapsed if elapsed > 0.1 else final_tps
            self._state.root.after(0, lambda b=batch, t=tps: self._paint_batch(b, t))

        try:
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=payload, headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                for raw in resp:
                    if self._stop_evt.is_set():
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        tok = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if tok:
                            tokens.append(tok)
                            pending.append(tok)
                            self._token_count += 1
                            now = time.monotonic()
                            if now - last_paint >= self._PAINT_MS:
                                _flush()
                                last_paint = now
                    except (KeyError, IndexError, json.JSONDecodeError):
                        pass
        except Exception as exc:
            if not self._stop_evt.is_set():
                error_msg = str(exc)

        full      = "".join(tokens)
        elapsed   = time.monotonic() - self._stream_t0
        final_tps = len(tokens) / elapsed if elapsed > 0.1 and tokens else 0.0
        _flush(final_tps)   # paint any remaining buffered tokens

        self._messages.append({"role": "assistant", "content": full})
        stopped = self._stop_evt.is_set()
        self._state.root.after(
            0, lambda: self._stream_done(error_msg, stopped, final_tps))

    def _paint_batch(self, text: str, tps: float = 0.0) -> None:
        try:
            self._hist.config(state="normal")
            self._hist.insert(tk.END, text, "assistant")
            self._hist.config(state="disabled")
            self._hist.see(tk.END)
            if tps > 0:
                self._status_lbl.config(text=f"{tps:.1f} t/s")
        except Exception:
            pass

    def _stream_done(self, error: str, stopped: bool, tps: float) -> None:
        self._streaming = False
        self._stop_btn.config(state="disabled")
        if self._connected:
            self._send_btn.config(state="normal")
        if error:
            self._hist.config(state="normal")
            self._hist.insert(tk.END, f"\n[Error: {error}]\n", "dim")
            self._hist.config(state="disabled")
            self._status_lbl.config(text=f"Error: {error[:40]}")
        elif stopped:
            self._hist.config(state="normal")
            self._hist.insert(tk.END, " [stopped]\n", "dim")
            self._hist.config(state="disabled")
            self._status_lbl.config(text=f"[stopped]  {tps:.1f} t/s" if tps else "[stopped]")
        else:
            self._hist.config(state="normal")
            self._hist.insert(tk.END, "\n", "")
            self._hist.config(state="disabled")
            self._status_lbl.config(text=f"{tps:.1f} t/s" if tps else "")
        self._hist.see(tk.END)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _append_hist(self, label: str, text: str,
                     lbl_tag: str, body_tag: str) -> None:
        self._hist.config(state="normal")
        self._hist.insert(tk.END, f"\n{label}\n", lbl_tag)
        self._hist.insert(tk.END, text + "\n", body_tag)
        self._hist.config(state="disabled")
        self._hist.see(tk.END)
