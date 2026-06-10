"""Server tab — binary/fork selector, port, parallel, API key, endpoints, update buttons."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Callable

from core.settings import DEFAULT_FORKS
from core.wsl      import update_build
from gui.widgets   import ToolTip, section, sep, entry, spinbox, combo, flag_row, cbk, grid_frame

LogFn = Callable[[str, str | None], None]


class ServerTab:

    def __init__(self, frame: tk.Frame, state, T: dict, log_fn: LogFn):
        self._frame = frame
        self._state = state
        self._T     = T
        self._log   = log_fn

    def build(self) -> None:
        T  = self._T
        sf = self._scrollable_frame()

        # ── Binary / Fork selector ────────────────────────────────────────────
        section(sf, "SERVER BINARY", T)
        bf = grid_frame(sf)
        tk.Label(bf, text="Server binary", bg=bf.cget("bg"), fg=T["fg2"],
                 font=("Segoe UI", 9), anchor="w").grid(
                     row=0, column=0, sticky="w", padx=(0, 8), pady=2)
        fork_values = [f["label"] for f in DEFAULT_FORKS]
        fork_combo = ttk.Combobox(bf, textvariable=self._state.llama_bin_var,
                                  values=fork_values,
                                  font=("Consolas", 9), state="normal", width=38)
        fork_combo.grid(row=0, column=1, sticky="ew", pady=2)
        ToolTip(fork_combo,
                "Path to llama-server binary in WSL.\n"
                "Official: standard upstream build.\n"
                "TurboQuant: adds turbo2/3/4 KV cache types.")

        # Fork descriptions
        for fork in DEFAULT_FORKS:
            row = tk.Frame(sf, bg=sf.cget("bg"))
            row.pack(fill="x", padx=12, pady=1)
            tk.Label(row, text=f"  {fork['label']}:", bg=sf.cget("bg"),
                     fg=T["accent"], font=("Segoe UI", 8, "bold")).pack(side="left")
            tk.Label(row, text=fork["description"], bg=sf.cget("bg"),
                     fg=T["fg2"], font=("Segoe UI", 8)).pack(side="left", padx=4)

        # Update buttons
        ubr = tk.Frame(sf, bg=sf.cget("bg"))
        ubr.pack(fill="x", padx=12, pady=(4, 8))
        s = self._state.settings
        for fork in DEFAULT_FORKS:
            root_key = fork["root_key"]
            root_dir = s.llama_root if root_key == "llama_root" else s.turbo_root
            tk.Button(
                ubr, text=f"Update {fork['label']}",
                bg=T["accent"], fg=T["bg"], relief="flat", cursor="hand2",
                font=("Segoe UI", 9, "bold"),
                command=lambda r=root_dir, lbl=fork["label"]: update_build(
                    s.wsl_distro, s.wsl_user, r, lbl, self._log)
            ).pack(side="left", padx=(0, 6))

        sep(sf, T)

        # ── Server params ─────────────────────────────────────────────────────
        section(sf, "SERVER SETTINGS", T)
        g = grid_frame(sf)
        r = 0
        entry(g, r, "Port (--port)",        self._state.port_var,            width=8,
              tip="TCP port the server listens on. Default: 8089"); r += 1
        spinbox(g, r, "Parallel slots (-np)", self._state.parallel_var,       1, 32,
                tip="Number of parallel request slots."); r += 1
        spinbox(g, r, "HTTP threads",         self._state.threads_http_var,  -1, 64,
                tip="Threads for the HTTP server. -1 = auto."); r += 1
        entry(g, r, "API key (--api-key)",   self._state.api_key_server_var, width=20,
              tip="Optional API key. Leave blank to disable."); r += 1
        entry(g, r, "Model alias",            self._state.alias_var,          width=28,
              tip="Name reported by /v1/models. Leave blank to use filename."); r += 1
        entry(g, r, "Timeout (--timeout)",   self._state.server_timeout_var, width=8,
              tip="HTTP read timeout in seconds. 0 = no timeout."); r += 1

        sep(sf, T)

        # ── Endpoints ─────────────────────────────────────────────────────────
        section(sf, "SERVER ENDPOINTS", T)
        ef = tk.Frame(sf, bg=sf.cget("bg"))
        ef.pack(fill="x", padx=12, pady=4)
        cbk(ef, "--cont-batching",  self._state.cont_batching_var,
            "Batch tokens from multiple requests. Greatly improves throughput.")
        cbk(ef, "--embeddings",     self._state.embeddings_var,
            "Expose embeddings endpoint. Required for RAG workflows.")
        cbk(ef, "--metrics",        self._state.metrics_var,
            "Expose Prometheus-compatible metrics at /metrics.")
        cbk(ef, "--props",          self._state.props_endpoint_var,
            "Expose server properties at /v1/props.")
        cbk(ef, "--slots",          self._state.slots_endpoint_var,
            "Expose slot state at /slots.")

    def _scrollable_frame(self) -> tk.Frame:
        T = self._T
        canvas = tk.Canvas(self._frame, bg=T["bg2"], highlightthickness=0)
        vsb    = tk.Scrollbar(self._frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        sf = tk.Frame(canvas, bg=T["bg2"])
        win = canvas.create_window((0, 0), window=sf, anchor="nw")
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.bind("<Enter>",
            lambda e: self._frame.winfo_toplevel().bind_all(
                "<MouseWheel>", lambda ev: canvas.yview_scroll(-1 if ev.delta > 0 else 1, "units")))
        canvas.bind("<Leave>",
            lambda e: self._frame.winfo_toplevel().unbind_all("<MouseWheel>"))
        return sf
