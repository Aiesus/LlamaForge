"""
Header bar — server status, GPU/RAM/CPU bars, tokens/sec.
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Callable

from core.monitor  import MonitorSnapshot

LogFn = Callable[[str, str | None], None]


class Header:

    def __init__(self, root: tk.Tk, state, T: dict, log_fn: LogFn):
        self._root    = root
        self._state   = state
        self._T       = T
        self._log     = log_fn
        self._frame: tk.Frame | None = None
        self._bars: dict = {}
        self._build(root)

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self, parent: tk.Widget) -> None:
        T = self._T
        self._frame = tk.Frame(parent, bg=T["bg2"], pady=8)
        self._frame.pack(fill="x", padx=10, pady=(10, 5))

        # ── Status pills ──────────────────────────────────────────────────────
        self._server_pill = tk.Label(
            self._frame, text="  STOPPED  ",
            bg=T["red"], fg=T["bg"], font=("Consolas", 10, "bold"),
            padx=6, pady=2
        )
        self._server_pill.pack(side="left", padx=(10, 8))

        # Tokens/sec readout
        self._tps_label = tk.Label(
            self._frame, text="",
            bg=T["bg2"], fg=T["fg2"], font=("Consolas", 9),
        )
        self._tps_label.pack(side="left", padx=(0, 12))

        # ── GPU bars ──────────────────────────────────────────────────────────
        self._gpu_vram_bars:       list[dict]     = []
        self._gpu_compute_bars:    list[dict]     = []
        self._gpu_temp_labels:     list[tk.Label] = []
        self._gpu_temp_hdr_labels: list[tk.Label] = []

        for i in range(2):
            vram_bar = self._make_bar(self._frame, f"GPU {i} VRAM", T["bar_fg"])
            vram_bar["outer"].pack(side="left", padx=8)
            self._gpu_vram_bars.append(vram_bar)

            compute_bar = self._make_bar(self._frame, f"GPU {i} Load", T["orange"])
            compute_bar["outer"].pack(side="left", padx=8)
            compute_bar["text"].config(text="—")
            self._gpu_compute_bars.append(compute_bar)

            temp_outer = tk.Frame(self._frame, bg=T["bg2"])
            temp_outer.pack(side="left", padx=(0, 8))
            temp_hdr = tk.Label(temp_outer, text=f"GPU {i} Temp", bg=T["bg2"], fg=T["fg2"],
                                font=("Consolas", 9))
            temp_hdr.pack(anchor="w")
            self._gpu_temp_hdr_labels.append(temp_hdr)
            temp_lbl = tk.Label(temp_outer, text="—°C",
                                bg=T["bg2"], fg=T["fg2"], font=("Consolas", 10, "bold"))
            temp_lbl.pack(anchor="w")
            self._gpu_temp_labels.append(temp_lbl)

            if i > 0:
                vram_bar["text"].config(text="not detected")
                compute_bar["text"].config(text="not detected")
                temp_lbl.config(text="—°C")

        # ── WSL RAM ───────────────────────────────────────────────────────────
        wsl_bar = self._make_bar(self._frame, "WSL RAM", T["green"])
        wsl_bar["outer"].pack(side="left", padx=8)
        self._wsl_ram_bar = wsl_bar

        # ── Windows RAM ───────────────────────────────────────────────────────
        win_ram_bar = self._make_bar(self._frame, "Win RAM", T["accent"])
        win_ram_bar["outer"].pack(side="left", padx=8)
        self._win_ram_bar = win_ram_bar

        # ── Windows CPU ───────────────────────────────────────────────────────
        cpu_bar = self._make_bar(self._frame, "Win CPU", T["yellow"])
        cpu_bar["outer"].pack(side="left", padx=8)
        cpu_bar["text"].config(text="— %")
        self._cpu_bar = cpu_bar


    # ── Update methods (called from main thread via _safe_after) ──────────────

    def update_server_status(self, state: str, model: str = "") -> None:
        T = self._T
        try:
            if state == "stopped":
                self._server_pill.config(
                    text="  STOPPED  ", bg=T["red"], fg=T["bg"])
                self._tps_label.config(text="")
            elif state == "loading":
                short = (model[:26] + "…") if len(model) > 28 else model
                self._server_pill.config(
                    text=f"  LOADING: {short}  ", bg=T["orange"], fg=T["bg"])
            elif state == "running":
                short = (model[:26] + "…") if len(model) > 28 else model
                self._server_pill.config(
                    text=f"  RUNNING: {short}  ", bg=T["green"], fg=T["bg"])
        except Exception:
            pass

    def update_tps(self, tps: float) -> None:
        """Update the tokens/sec readout. Called when log parser detects gen stats."""
        try:
            self._tps_label.config(text="" if tps == 0.0 else f"{tps:.1f} t/s")
        except Exception:
            pass

    def update_stats(self, snap: MonitorSnapshot) -> None:
        T = self._T
        try:
            swap = self._state.cuda_swap_var.get()
            # When swapped, display slot 0 = CUDA device 0 = physical GPU 1 (and vice versa)
            phys_order = [1, 0] if swap else [0, 1]
            prefix     = "CUDA" if swap else "GPU"

            for slot in range(len(self._gpu_vram_bars)):
                phys = phys_order[slot] if slot < len(phys_order) else slot
                self._gpu_vram_bars[slot]["label"].config(text=f"{prefix} {slot} VRAM")
                self._gpu_compute_bars[slot]["label"].config(text=f"{prefix} {slot} Load")
                self._gpu_temp_hdr_labels[slot].config(text=f"{prefix} {slot} Temp")

                if phys < len(snap.gpus):
                    gpu = snap.gpus[phys]
                    self._update_bar(self._gpu_vram_bars[slot],
                                     gpu.used_mb, gpu.total_mb, gpu.pct, "MiB")
                    self._update_bar(self._gpu_compute_bars[slot],
                                     gpu.util_pct, 100, gpu.util_pct / 100, "%")
                    self._gpu_temp_labels[slot].config(text=f"{gpu.temp_c}°C",
                                                       fg=self._temp_color(gpu.temp_c))
                else:
                    self._gpu_vram_bars[slot]["text"].config(text="not detected")
                    self._gpu_compute_bars[slot]["text"].config(text="not detected")
                    self._gpu_temp_labels[slot].config(text="—°C")

            # WSL RAM
            if snap.wsl_total_gb:
                self._update_bar(self._wsl_ram_bar,
                                 snap.wsl_used_gb, snap.wsl_total_gb,
                                 snap.wsl_pct, "GiB")

            # Windows RAM
            if snap.win_ram_total_gb:
                self._update_bar(self._win_ram_bar,
                                 snap.win_ram_used_gb, snap.win_ram_total_gb,
                                 snap.win_ram_pct, "GiB")

            # Windows CPU
            if snap.win_cpu_pct >= 0:
                pct = snap.win_cpu_pct / 100
                self._update_bar(self._cpu_bar,
                                 snap.win_cpu_pct, 100, pct, "%")

        except Exception:
            pass

    # ── Bar helpers ───────────────────────────────────────────────────────────

    def _make_bar(self, parent: tk.Widget, label: str, color: str) -> dict:
        T = self._T
        outer = tk.Frame(parent, bg=T["bg2"])
        hdr = tk.Label(outer, text=label, bg=T["bg2"], fg=T["fg2"], font=("Consolas", 9))
        hdr.pack(anchor="w")
        bar_bg = tk.Frame(outer, bg=T["bar_bg"], width=140, height=14)
        bar_bg.pack_propagate(False)
        bar_bg.pack()
        bar_fill = tk.Frame(bar_bg, bg=color, width=0, height=14)
        bar_fill.place(x=0, y=0, height=14)
        text_lbl = tk.Label(outer, text="— / —", bg=T["bg2"], fg=T["fg"],
                            font=("Consolas", 9))
        text_lbl.pack()
        return {"outer": outer, "bg": bar_bg, "fill": bar_fill,
                "text": text_lbl, "default_color": color, "label": hdr}

    def _update_bar(self, bar: dict, used, total, pct: float, unit: str) -> None:
        try:
            T = self._T
            w = int(140 * pct)
            color = (T["red"]    if pct > 0.9 else
                     T["orange"] if pct > 0.7 else
                     bar["default_color"])
            bar["fill"].place(x=0, y=0, width=w, height=14)
            bar["fill"].config(bg=color)
            if unit == "%":
                bar["text"].config(text=f"{pct * 100:.0f}%")
            elif unit in ("MiB", "GiB"):
                bar["text"].config(
                    text=f"{used:.1f}/{total:.1f} {unit} ({pct*100:.0f}%)"
                    if unit == "GiB"
                    else f"{int(used):,}/{int(total):,} {unit} ({pct*100:.0f}%)"
                )
        except Exception:
            pass

    def _temp_color(self, temp: int) -> str:
        T = self._T
        if temp >= 85: return T["red"]
        if temp >= 70: return T["orange"]
        if temp >= 50: return T["yellow"]
        return T["green"]

