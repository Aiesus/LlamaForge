"""Advanced tab — RoPE, performance flags, speculative decoding, WSL, misc."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gui.app import AppState

from gui.widgets import section, sep, entry, spinbox, combo, slider_spin, cbk, flag_row, grid_frame

LogFn = Callable[[str, str | None], None]

ROPE_SCALING_VALUES = ["auto", "none", "linear", "yarn"]
PRIO_LEVELS = ["0", "1", "2", "3"]


class AdvancedTab:

    def __init__(self, frame: tk.Frame, state: AppState, T: dict, log_fn: LogFn):
        self._frame = frame
        self._state = state
        self._T     = T
        self._log   = log_fn

    def build(self) -> None:
        T  = self._T
        sf = self._scrollable_frame()

        # ── RoPE ──────────────────────────────────────────────────────────────
        section(sf, "ROPE / CONTEXT EXTENSION", T)
        g = grid_frame(sf)
        r = 0
        slider_spin(g, r, "RoPE freq base (--rope-freq-base)", self._state.rope_freq_base_var,
                    0.0, 10_000_000.0, 100.0,
                    "0 = use model default. Increase for context beyond training length."); r += 1
        combo(g, r, "RoPE scaling (--rope-scaling)", self._state.rope_scaling_var,
              ROPE_SCALING_VALUES,
              tip="Context extension method. 'auto' selects based on model metadata."); r += 1

        sep(sf, T)

        # ── Performance flags ──────────────────────────────────────────────────
        section(sf, "PERFORMANCE FLAGS", T)
        pf = tk.Frame(sf, bg=sf.cget("bg"))
        pf.pack(fill="x", padx=12, pady=4)

        flag_row(pf, "--prio (main thread priority)", self._state.prio_en_var,
                 self._state.prio_level_var, "combo",
                 "0=Normal 1=Medium 2=High 3=Realtime. Requires elevated privilege.",
                 val_values=PRIO_LEVELS, val_width=3)
        flag_row(pf, "--prio-batch (batch thread priority)", self._state.prio_batch_en_var,
                 self._state.prio_batch_level_var, "combo",
                 "Priority for the batch processing thread.",
                 val_values=PRIO_LEVELS, val_width=3)
        flag_row(pf, "--cache-reuse (KV cache reuse chunks)", self._state.cache_reuse_en_var,
                 self._state.cache_reuse_n_var, "entry",
                 "Min token overlap to consider KV cache reuse. 256 is a good starting point.",
                 val_width=6)

        sep(sf, T)

        # ── Speculative decoding ───────────────────────────────────────────────
        section(sf, "SPECULATIVE DECODING (MTP)", T)
        sd = tk.Frame(sf, bg=sf.cget("bg"))
        sd.pack(fill="x", padx=12, pady=4)

        cbk(sd, "--spec-type draft-mtp", self._state.spec_mtp_var,
            "Enable Multi-Token Prediction speculative decoding (DeepSeek / Qwen MTP models). "
            "Requires a model with built-in MTP heads — no separate draft model needed. "
            "Use --draft-max below to set how many tokens to draft per step.")
        flag_row(sd, "--draft-max  (tokens per draft step)", self._state.spec_draft_n_en_var,
                 self._state.spec_draft_n_var, "entry",
                 "How many tokens to speculatively draft per step when --spec-type draft-mtp is on. "
                 "2–4 is typical; higher values help on fast GPUs.",
                 val_width=3)
        flag_row(sd, "--draft-prio (draft thread priority)", self._state.prio_draft_en_var,
                 self._state.prio_draft_level_var, "combo",
                 "Priority for the speculative draft thread.",
                 val_values=PRIO_LEVELS, val_width=3)

        sep(sf, T)

        # ── WSL memory ────────────────────────────────────────────────────────
        section(sf, "WSL MEMORY (.wslconfig)", T)
        g2 = grid_frame(sf)
        tk.Label(g2, text="WSL memory limit", bg=g2.cget("bg"), fg=T["fg2"],
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=2)
        wf = tk.Frame(g2, bg=g2.cget("bg"))
        wf.grid(row=0, column=1, sticky="ew", pady=2)
        tk.Entry(wf, textvariable=self._state.wsl_memory_var, width=8,
                 bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                 font=("Consolas", 10), insertbackground=T["fg"]).pack(side="left")
        tk.Label(wf, text='e.g. "64GB"', bg=g2.cget("bg"),
                 fg=T["fg2"], font=("Segoe UI", 8)).pack(side="left", padx=6)

        from core.wsl import write_wsl_memory
        tk.Button(g2, text="Apply (restarts WSL)", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=self._apply_wsl_memory
                  ).grid(row=1, column=1, sticky="w", pady=(2, 4))

        sep(sf, T)

        # ── Misc flags ─────────────────────────────────────────────────────────
        section(sf, "MISC FLAGS", T)
        mf = tk.Frame(sf, bg=sf.cget("bg"))
        mf.pack(fill="x", padx=12, pady=4)
        cbk(mf, "--no-display-prompt", self._state.no_display_prompt_var,
            "Suppress prompt echo in server logs. Useful for privacy/log size.")

        sep(sf, T)

        # ── Extra flags ────────────────────────────────────────────────────────
        section(sf, "EXTRA FLAGS", T)
        ef = tk.Frame(sf, bg=sf.cget("bg"))
        ef.pack(fill="x", padx=12, pady=4)
        tk.Label(ef, text="Append to command:", bg=ef.cget("bg"), fg=T["fg2"],
                 font=("Segoe UI", 9)).pack(anchor="w")
        tk.Entry(ef, textvariable=self._state.extra_flags_var, width=50,
                 bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                 font=("Consolas", 9), insertbackground=T["fg"]).pack(fill="x", pady=3)

        sep(sf, T)

        # ── Command preview ────────────────────────────────────────────────────
        section(sf, "COMMAND PREVIEW", T)
        cp = tk.Frame(sf, bg=sf.cget("bg"))
        cp.pack(fill="x", padx=12, pady=4)

        self._preview = tk.Text(cp, height=5, bg=T["log_bg"], fg=T["log_fg"],
                                font=("Consolas", 8), relief="flat", wrap="word",
                                state="disabled", padx=6, pady=4)
        self._preview.pack(fill="x")

        tk.Button(cp, text="Refresh Preview", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=self._refresh_preview).pack(anchor="w", pady=(4, 0))

        # ── GPU / CUDA ─────────────────────────────────────────────────────────
        sep(sf, T)
        section(sf, "GPU / CUDA", T)
        gf = tk.Frame(sf, bg=sf.cget("bg"))
        gf.pack(fill="x", padx=12, pady=4)
        cbk(gf, "Swap GPU order  (CUDA_VISIBLE_DEVICES=1,0)",
            self._state.cuda_swap_var,
            "Makes GPU 1 the primary CUDA device (device 0).\n"
            "Use when GPU 1 has more VRAM and should handle the larger share.\n"
            "Also update tensor-split and main-gpu in your profile to match.")

        # ── Proxy bypass ───────────────────────────────────────────────────────
        sep(sf, T)
        section(sf, "DEBUG", T)
        dbf = tk.Frame(sf, bg=sf.cget("bg"))
        dbf.pack(fill="x", padx=12, pady=4)
        cbk(dbf, "Bypass proxy (connect clients directly to :8089)",
            self._state.proxy_bypass_var,
            "Skip tool-proxy.py. Use when debugging direct API calls on :8089.")

    def _apply_wsl_memory(self) -> None:
        from core.wsl import write_wsl_memory
        mem = self._state.wsl_memory_var.get().strip()
        write_wsl_memory(mem)
        self._log(f"[WSL] Memory limit set to {mem}. Restart WSL for it to take effect.", "info")

    def _refresh_preview(self) -> None:
        try:
            cmd = self._state.build_cmd()
        except Exception as e:
            cmd = f"<error building command: {e}>"
        self._preview.config(state="normal")
        self._preview.delete("1.0", tk.END)
        self._preview.insert(tk.END, cmd)
        self._preview.config(state="disabled")

    def _scrollable_frame(self) -> tk.Frame:
        T      = self._T
        canvas = tk.Canvas(self._frame, bg=T["bg2"], highlightthickness=0)
        vsb    = ttk.Scrollbar(self._frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        sf  = tk.Frame(canvas, bg=T["bg2"])
        win = canvas.create_window((0, 0), window=sf, anchor="nw")
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.bind("<Enter>",
            lambda e: self._frame.winfo_toplevel().bind_all(
                "<MouseWheel>", lambda ev: canvas.yview_scroll(-1 if ev.delta > 0 else 1, "units")))
        canvas.bind("<Leave>",
            lambda e: self._frame.winfo_toplevel().unbind_all("<MouseWheel>"))
        return sf
