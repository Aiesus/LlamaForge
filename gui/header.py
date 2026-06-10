"""
Header bar — server status, GPU/RAM/CPU bars, tokens/sec, action buttons.
"""
from __future__ import annotations
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable

from core.monitor  import MonitorSnapshot
from core.server   import run_diagnostics
from core.wsl      import restart_proxy
from gui.themes    import THEME_LABELS, get as get_theme

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
        self._gpu_vram_bars:    list[dict] = []
        self._gpu_compute_bars: list[dict] = []
        self._gpu_temp_labels:  list[tk.Label] = []

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
            tk.Label(temp_outer, text=f"GPU {i} Temp", bg=T["bg2"], fg=T["fg2"],
                     font=("Consolas", 9)).pack(anchor="w")
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

        # ── Right-side buttons ────────────────────────────────────────────────
        tk.Button(
            self._frame, text="Theme",
            bg=T["btn"], fg=T["btn_fg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9), command=self._open_theme_picker
        ).pack(side="right", padx=4)

        tk.Button(
            self._frame, text="🌐 llama UI",
            bg=T["accent"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9, "bold"),
            command=lambda: webbrowser.open(
                f"http://localhost:{self._state.port_var.get()}"
            )
        ).pack(side="right", padx=4)

        tk.Button(
            self._frame, text="Restart Proxy",
            bg=T["btn"], fg=T["btn_fg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9), command=self._restart_proxy
        ).pack(side="right", padx=4)

        tk.Button(
            self._frame, text="Diagnose",
            bg=T["yellow"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9, "bold"), command=self._diagnose
        ).pack(side="right", padx=4)

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
            for i, gpu in enumerate(snap.gpus):
                if i >= len(self._gpu_vram_bars):
                    break
                self._update_bar(self._gpu_vram_bars[i],
                                 gpu.used_mb, gpu.total_mb, gpu.pct, "MiB")
                self._update_bar(self._gpu_compute_bars[i],
                                 gpu.util_pct, 100, gpu.util_pct / 100, "%")
                temp_lbl = self._gpu_temp_labels[i]
                temp_lbl.config(text=f"{gpu.temp_c}°C",
                                fg=self._temp_color(gpu.temp_c))

            # Mark undetected GPUs
            for i in range(len(snap.gpus), len(self._gpu_vram_bars)):
                self._gpu_vram_bars[i]["text"].config(text="not detected")
                self._gpu_compute_bars[i]["text"].config(text="not detected")
                self._gpu_temp_labels[i].config(text="—°C")

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
        tk.Label(outer, text=label, bg=T["bg2"], fg=T["fg2"],
                 font=("Consolas", 9)).pack(anchor="w")
        bar_bg = tk.Frame(outer, bg=T["bar_bg"], width=140, height=14)
        bar_bg.pack_propagate(False)
        bar_bg.pack()
        bar_fill = tk.Frame(bar_bg, bg=color, width=0, height=14)
        bar_fill.place(x=0, y=0, height=14)
        text_lbl = tk.Label(outer, text="— / —", bg=T["bg2"], fg=T["fg"],
                            font=("Consolas", 9))
        text_lbl.pack()
        return {"outer": outer, "bg": bar_bg, "fill": bar_fill,
                "text": text_lbl, "default_color": color}

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

    # ── Button actions ────────────────────────────────────────────────────────

    def _restart_proxy(self) -> None:
        s = self._state.settings
        restart_proxy(s.wsl_distro, s.wsl_user, self._log)

    def _diagnose(self) -> None:
        s = self._state.settings
        run_diagnostics(
            s.wsl_distro, s.wsl_user,
            self._state.port_var.get(),
            self._log,
        )

    def _open_theme_picker(self) -> None:
        T = self._T
        win = tk.Toplevel(self._root)
        win.title("Choose Theme")
        win.geometry("320x260")
        win.configure(bg=T["bg"])
        win.resizable(False, False)

        tk.Label(win, text="Select theme:", bg=T["bg"], fg=T["fg"],
                 font=("Segoe UI", 10)).pack(pady=(16, 8))

        var = tk.StringVar(value=self._state.settings.theme)
        for key, label in THEME_LABELS.items():
            tk.Radiobutton(
                win, text=label, variable=var, value=key,
                bg=T["bg"], fg=T["fg"], selectcolor=T["bg3"],
                activebackground=T["bg"], activeforeground=T["accent"],
                font=("Segoe UI", 9), cursor="hand2"
            ).pack(anchor="w", padx=24, pady=2)

        def _apply():
            chosen = var.get()
            self._state.settings.theme = chosen
            from core.settings import save_settings
            save_settings(self._state.settings)
            messagebox.showinfo(
                "Theme saved",
                f"Theme set to '{THEME_LABELS[chosen]}'.\n"
                "Restart the app to apply.",
                parent=win
            )
            win.destroy()

        btn_row = tk.Frame(win, bg=T["bg"])
        btn_row.pack(pady=12)
        tk.Button(btn_row, text="Apply", bg=T["green"], fg=T["bg"],
                  relief="flat", font=("Segoe UI", 10, "bold"), cursor="hand2",
                  command=_apply).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", font=("Segoe UI", 10), cursor="hand2",
                  command=win.destroy).pack(side="left")
