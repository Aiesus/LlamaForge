"""Sampling tab — temp, top-k/p, min-p, penalties, predict, seed, mirostat."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Callable

from gui.widgets import section, sep, spinbox, slider_spin, flag_row, grid_frame

LogFn = Callable[[str, str | None], None]

MIROSTAT_LABELS = {0: "Off", 1: "v1", 2: "v2"}


class SamplingTab:

    def __init__(self, frame: tk.Frame, state, T: dict, log_fn: LogFn):
        self._frame = frame
        self._state = state
        self._T     = T
        self._log   = log_fn

    def build(self) -> None:
        T  = self._T
        sf = self._scrollable_frame()

        section(sf, "SAMPLING", T)
        g = grid_frame(sf)
        r = 0
        slider_spin(g, r, "Temperature (--temp)",     self._state.temp_var,
                    0.0, 2.0, 0.01,
                    "Randomness. 0 = deterministic. 1 = typical. 2 = chaotic."); r += 1
        slider_spin(g, r, "Top-K (--top-k)",          self._state.top_k_var,
                    0, 200, 1,
                    "Keep top K tokens. 0 = disabled."); r += 1
        slider_spin(g, r, "Top-P (--top-p)",          self._state.top_p_var,
                    0.0, 1.0, 0.01,
                    "Nucleus sampling. Keep tokens summing to this probability mass."); r += 1
        slider_spin(g, r, "Min-P (--min-p)",          self._state.min_p_var,
                    0.0, 1.0, 0.01,
                    "Minimum probability relative to top token. Clips low-prob tokens."); r += 1

        sep(sf, T)
        section(sf, "PENALTIES", T)
        g2 = grid_frame(sf)
        r  = 0
        slider_spin(g2, r, "Repeat penalty (--repeat-penalty)", self._state.repeat_penalty_var,
                    1.0, 2.0, 0.01,
                    "Penalize recently used tokens. 1.0 = no penalty."); r += 1
        spinbox(g2, r, "Repeat last N (--repeat-last-n)", self._state.repeat_last_n_var,
                -1, 2048, tip="Window of tokens considered for repeat penalty."); r += 1
        slider_spin(g2, r, "Presence penalty",  self._state.presence_penalty_var,
                    -2.0, 2.0, 0.01,
                    "Penalize tokens that have appeared at all. OpenAI-compatible."); r += 1
        slider_spin(g2, r, "Frequency penalty", self._state.frequency_penalty_var,
                    -2.0, 2.0, 0.01,
                    "Penalize tokens proportional to frequency. OpenAI-compatible."); r += 1

        sep(sf, T)
        section(sf, "GENERATION", T)
        g3 = grid_frame(sf)
        r  = 0
        spinbox(g3, r, "Predict / max tokens (-n)", self._state.predict_var, -1, 32768,
                tip="-1 = unlimited. Controls max tokens in a single generation."); r += 1
        spinbox(g3, r, "Seed (--seed)", self._state.seed_var, -1, 2147483647,
                tip="-1 = random. Fixed seed for reproducible outputs."); r += 1

        sep(sf, T)
        section(sf, "MIROSTAT", T)
        g4 = grid_frame(sf)
        r  = 0

        tk.Label(g4, text="Mode (--mirostat)", bg=g4.cget("bg"), fg=T["fg2"],
                 font=("Segoe UI", 9)).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        mf = tk.Frame(g4, bg=g4.cget("bg"))
        mf.grid(row=r, column=1, sticky="w", pady=2)
        for val, lbl in [(0, "Off"), (1, "v1"), (2, "v2")]:
            tk.Radiobutton(mf, text=lbl, variable=self._state.mirostat_var, value=val,
                           bg=g4.cget("bg"), fg=T["fg"], selectcolor=T["bg3"],
                           activebackground=g4.cget("bg"), activeforeground=T["accent"],
                           font=("Segoe UI", 9), cursor="hand2").pack(side="left", padx=4)
        r += 1

        slider_spin(g4, r, "Learning rate (--mirostat-lr)", self._state.mirostat_lr_var,
                    0.0, 1.0, 0.01,
                    "Mirostat learning rate (eta). 0.1 default."); r += 1
        slider_spin(g4, r, "Entropy (--mirostat-ent)",      self._state.mirostat_ent_var,
                    1.0, 10.0, 0.1,
                    "Target entropy (tau). 5.0 default."); r += 1

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
