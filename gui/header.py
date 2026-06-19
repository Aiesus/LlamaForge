"""
Header bar — server status, GPU/RAM/CPU bars, tokens/sec.
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gui.app import AppState

from core.monitor  import MonitorSnapshot
from core.metrics  import TokenStats

LogFn = Callable[[str, str | None], None]


class Header:

    def __init__(self, root: tk.Tk, state: AppState, T: dict, log_fn: LogFn):
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
        # Status is conveyed by color; text is just the loaded model (or ● when idle).
        self._server_pill = tk.Label(
            self._frame, text="  ●  ",
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

        # ── GPU bars — grouped by metric, GPU0 stacked over GPU1 ──────────────
        self._gpu_vram_bars:       list[dict]     = []
        self._gpu_compute_bars:    list[dict]     = []
        self._gpu_temp_labels:     list[tk.Label] = []
        self._gpu_temp_hdr_labels: list[tk.Label] = []

        vram_group = tk.Frame(self._frame, bg=T["bg2"]); vram_group.pack(side="left", padx=8)
        load_group = tk.Frame(self._frame, bg=T["bg2"]); load_group.pack(side="left", padx=8)
        temp_group = tk.Frame(self._frame, bg=T["bg2"]); temp_group.pack(side="left", padx=8)

        for i in range(2):
            vram_bar = self._make_bar(vram_group, f"G{i} VRAM", T["bar_fg"])
            vram_bar["outer"].pack(side="top", anchor="w", pady=(0, 3))
            self._gpu_vram_bars.append(vram_bar)

            compute_bar = self._make_bar(load_group, f"G{i} Load", T["orange"])
            compute_bar["outer"].pack(side="top", anchor="w", pady=(0, 3))
            compute_bar["text"].config(text="—")
            self._gpu_compute_bars.append(compute_bar)

            temp_outer = tk.Frame(temp_group, bg=T["bg2"])
            temp_outer.pack(side="top", anchor="w", pady=(0, 3))
            temp_hdr = tk.Label(temp_outer, text=f"G{i} Temp", bg=T["bg2"], fg=T["fg2"],
                                font=("Consolas", 9), width=8, anchor="w")
            temp_hdr.pack(side="left")
            self._gpu_temp_hdr_labels.append(temp_hdr)
            temp_lbl = tk.Label(temp_outer, text="—°C",
                                bg=T["bg2"], fg=T["fg2"], font=("Consolas", 10, "bold"))
            temp_lbl.pack(side="left", padx=(2, 0))
            self._gpu_temp_labels.append(temp_lbl)

            if i > 0:
                vram_bar["text"].config(text="not detected")
                compute_bar["text"].config(text="not detected")
                temp_lbl.config(text="—°C")

        # ── RAM — WSL stacked over Windows ────────────────────────────────────
        ram_group = tk.Frame(self._frame, bg=T["bg2"]); ram_group.pack(side="left", padx=8)
        wsl_bar = self._make_bar(ram_group, "WSL RAM", T["green"])
        wsl_bar["outer"].pack(side="top", anchor="w", pady=(0, 3))
        self._wsl_ram_bar = wsl_bar
        win_ram_bar = self._make_bar(ram_group, "Win RAM", T["accent"])
        win_ram_bar["outer"].pack(side="top", anchor="w", pady=(0, 3))
        self._win_ram_bar = win_ram_bar

        # ── Windows CPU ───────────────────────────────────────────────────────
        cpu_bar = self._make_bar(self._frame, "Win CPU", T["yellow"])
        cpu_bar["outer"].pack(side="left", padx=8)
        cpu_bar["text"].config(text="— %")
        self._cpu_bar = cpu_bar

        # ── Token / context monitor strip ─────────────────────────────────────
        tok_outer = tk.Frame(self._frame, bg=T["bg2"])
        tok_outer.pack(side="left", padx=(12, 8))
        tk.Label(tok_outer, text="Context (tokens)", bg=T["bg2"], fg=T["fg2"],
                 font=("Consolas", 9)).pack(anchor="w")
        self._tok_ctx_lbl = tk.Label(tok_outer, text="—", bg=T["bg2"], fg=T["fg2"],
                                     font=("Consolas", 10, "bold"))
        self._tok_ctx_lbl.pack(anchor="w")
        self._tok_req_lbl = tk.Label(tok_outer, text="", bg=T["bg2"], fg=T["fg2"],
                                     font=("Consolas", 8))
        self._tok_req_lbl.pack(anchor="w")


    # ── Update methods (called from main thread via _safe_after) ──────────────

    def update_server_status(self, state: str, model: str = "") -> None:
        T = self._T
        try:
            short = (model[:24] + "…") if len(model) > 25 else model
            if state == "stopped":
                self._server_pill.config(text="  ●  ", bg=T["red"], fg=T["bg"])
                self._tps_label.config(text="")
            elif state == "loading":
                self._server_pill.config(text=f"  {short or '●'}  ",
                                         bg=T["orange"], fg=T["bg"])
            elif state == "running":
                self._server_pill.config(text=f"  {short or '●'}  ",
                                         bg=T["green"], fg=T["bg"])
        except Exception:
            pass

    def update_tps(self, tps: float) -> None:
        """Update the tokens/sec readout. Called when log parser detects gen stats."""
        try:
            self._tps_label.config(text="" if tps == 0.0 else f"{tps:.1f} t/s")
        except Exception:
            pass

    def update_tokens(self, st: TokenStats) -> None:
        """Update the context/token strip from a TokenStats poll."""
        T = self._T
        try:
            if not st.ok:
                self._tok_ctx_lbl.config(text="—", fg=T["fg2"])
                self._tok_req_lbl.config(text="server offline")
                return

            # Context fill comes from /slots, so it works even when /metrics is off.
            if st.n_ctx and st.ctx_used:
                pct = st.ctx_ratio * 100
                color = (T["red"]    if pct > 90 else
                         T["orange"] if pct > 70 else
                         T["yellow"] if pct > 50 else
                         T["green"])
                self._tok_ctx_lbl.config(
                    text=f"{pct:.0f}%  {self._fmt_k(st.ctx_used)}/{self._fmt_k(st.n_ctx)}",
                    fg=color)
            else:
                self._tok_ctx_lbl.config(text="—", fg=T["fg2"])

            parts = []
            if st.metrics_on and st.last_prompt:   # last-request size needs /metrics
                parts.append(f"last req {self._fmt_k(st.last_prompt)}")
            parts.append(f"● gen {self._fmt_k(st.n_decoded)}" if st.processing else "idle")
            self._tok_req_lbl.config(text="  ·  ".join(parts))
        except Exception:
            pass

    @staticmethod
    def _fmt_k(n: int) -> str:
        try:
            n = int(n)
        except (ValueError, TypeError):
            return "—"
        if n >= 100_000:
            return f"{n/1000:.0f}k"
        if n >= 1_000:
            return f"{n/1000:.1f}k"
        return str(n)

    def update_stats(self, snap: MonitorSnapshot) -> None:
        T = self._T
        try:
            swap = self._state.cuda_swap_var.get()
            # When swapped, display slot 0 = CUDA device 0 = physical GPU 1 (and vice versa)
            phys_order = [1, 0] if swap else [0, 1]
            prefix     = "C" if swap else "G"   # short: G0/G1 (or C0/C1 when swapped)

            for slot in range(len(self._gpu_vram_bars)):
                phys = phys_order[slot] if slot < len(phys_order) else slot
                self._gpu_vram_bars[slot]["label"].config(text=f"{prefix}{slot} VRAM")
                self._gpu_compute_bars[slot]["label"].config(text=f"{prefix}{slot} Load")
                self._gpu_temp_hdr_labels[slot].config(text=f"{prefix}{slot} Temp")

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
        # Single row: label │ bar │ value — keeps the header short when stacked.
        outer = tk.Frame(parent, bg=T["bg2"])
        hdr = tk.Label(outer, text=label, bg=T["bg2"], fg=T["fg2"],
                       font=("Consolas", 9), width=8, anchor="w")
        hdr.pack(side="left")
        canvas = tk.Canvas(outer, bg=T["bar_bg"], width=104, height=14,
                           highlightthickness=0, bd=0)
        canvas.pack(side="left", padx=(2, 4))
        rect = canvas.create_rectangle(0, 0, 0, 14, fill=color, width=0)
        text_lbl = tk.Label(outer, text="— / —", bg=T["bg2"], fg=T["fg"],
                            font=("Consolas", 9), anchor="w")
        text_lbl.pack(side="left")
        return {"outer": outer, "canvas": canvas, "rect": rect,
                "text": text_lbl, "default_color": color, "label": hdr}

    def _update_bar(self, bar: dict, used, total, pct: float, unit: str) -> None:
        try:
            T = self._T
            w = int(104 * pct)
            color = (T["red"]    if pct > 0.9 else
                     T["orange"] if pct > 0.7 else
                     bar["default_color"])
            bar["canvas"].coords(bar["rect"], 0, 0, w, 14)
            bar["canvas"].itemconfig(bar["rect"], fill=color)
            if unit == "%":
                bar["text"].config(text=f"{pct * 100:.0f}%")
            elif unit in ("MiB", "GiB"):
                used_g  = used / 1024 if unit == "MiB" else used
                total_g = total / 1024 if unit == "MiB" else total
                bar["text"].config(text=f"{pct*100:.0f}%  {used_g:.1f}/{total_g:.1f}G")
        except Exception:
            pass

    def _temp_color(self, temp: int) -> str:
        T = self._T
        if temp >= 85: return T["red"]
        if temp >= 70: return T["orange"]
        if temp >= 50: return T["yellow"]
        return T["green"]

