"""Optimizer tab — guided and sweep modes for llama-bench."""
from __future__ import annotations
import csv
import io
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gui.app import AppState

from core.settings import load_bench_results, save_bench_result, DEFAULT_FORKS
import core.optimizer as optimizer

LogFn = Callable[[str, str | None], None]

SCENARIOS = [
    "Full GPU",
    "Full Multi-GPU",
    "Partial Offload",
    "MoE CPU Offload",
    "Long Context",
    "Max Throughput",
]

SWEEP_PARAMS = [
    ("ngl",          "ngl",           "GPU Layers"),
    ("ctx",          "ctx",           "Context"),
    ("batch",        "batch",         "Batch"),
    ("ubatch",       "ubatch",        "µBatch"),
    ("flash_attn",   "flash_attn",    "Flash Attn"),
    ("cache_type_k", "cache_type_k",  "KV-K type"),
    ("cache_type_v", "cache_type_v",  "KV-V type"),
]

RESULT_COLS = ("pp t/s", "tg t/s", "ngl", "ctx", "batch", "cache_k", "cache_v", "flash_attn")


class OptimizerTab:

    def __init__(self, frame: tk.Frame, state: AppState, T: dict, log_fn: LogFn):
        self._frame  = frame
        self._state  = state
        self._T      = T
        self._log    = log_fn
        self._running = False

    def build(self) -> None:
        T = self._T

        nb = ttk.Notebook(self._frame)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        # --- Guided tab
        gf = tk.Frame(nb, bg=T["bg2"])
        nb.add(gf, text="Guided")
        self._build_guided(gf)

        # --- Sweep tab
        sf = tk.Frame(nb, bg=T["bg2"])
        nb.add(sf, text="Sweep")
        self._build_sweep(sf)

        # --- Results tab
        rf = tk.Frame(nb, bg=T["bg2"])
        nb.add(rf, text="Results")
        self._build_results(rf)

    # ── Guided ────────────────────────────────────────────────────────────────

    def _build_guided(self, parent: tk.Frame) -> None:
        T = self._T
        self._cancel_evt = threading.Event()

        # Hardware summary
        hw_frame = tk.Frame(parent, bg=T["bg3"])
        hw_frame.pack(fill="x", padx=8, pady=(8, 4))
        self._hw_label = tk.Label(
            hw_frame, text="Detecting hardware…",
            bg=T["bg3"], fg=T["fg2"], font=("Consolas", 8),
            justify="left", padx=8, pady=6
        )
        self._hw_label.pack(anchor="w")
        tk.Button(hw_frame, text="Refresh", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 8),
                  command=self._refresh_hw).pack(anchor="e", padx=8, pady=(0, 4))

        # Scenario selector + auto-detect
        sc_frame = tk.Frame(parent, bg=T["bg2"])
        sc_frame.pack(fill="x", padx=8, pady=4)
        tk.Label(sc_frame, text="Scenario:", bg=T["bg2"], fg=T["fg2"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._scenario_var = tk.StringVar(value=SCENARIOS[0])
        sc_combo = ttk.Combobox(sc_frame, textvariable=self._scenario_var,
                                values=SCENARIOS, state="readonly",
                                font=("Segoe UI", 9), width=20)
        sc_combo.pack(side="left", padx=8)
        sc_combo.bind("<<ComboboxSelected>>", lambda e: self._update_guided_matrix())
        tk.Button(sc_frame, text="Auto-detect", bg=T["btn"], fg=T["accent"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 8),
                  command=self._auto_detect_scenario).pack(side="left", padx=4)
        self._detect_reason_lbl = tk.Label(
            sc_frame, text="", bg=T["bg2"], fg=T["fg2"], font=("Consolas", 7)
        )
        self._detect_reason_lbl.pack(side="left", padx=8)

        # Test matrix display
        self._matrix_frame = tk.Frame(parent, bg=T["bg3"])
        self._matrix_frame.pack(fill="x", padx=8, pady=4)
        self._matrix_lbl = tk.Label(
            self._matrix_frame,
            text="Select a scenario to see the test matrix.",
            bg=T["bg3"], fg=T["fg2"], font=("Consolas", 8), padx=8, pady=4, justify="left"
        )
        self._matrix_lbl.pack(anchor="w")

        # Fork selector
        fk_frame = tk.Frame(parent, bg=T["bg2"])
        fk_frame.pack(fill="x", padx=8, pady=2)
        tk.Label(fk_frame, text="Bench binary:", bg=T["bg2"], fg=T["fg2"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._guided_fork_var = tk.StringVar(value=DEFAULT_FORKS[0]["label"])
        for f in DEFAULT_FORKS:
            tk.Radiobutton(fk_frame, text=f["label"],
                           variable=self._guided_fork_var, value=f["label"],
                           bg=T["bg2"], fg=T["fg"], selectcolor=T["bg3"],
                           activebackground=T["bg2"], activeforeground=T["accent"],
                           font=("Segoe UI", 9), cursor="hand2"
                           ).pack(side="left", padx=4)

        # Run button + cancel + progress
        run_frame = tk.Frame(parent, bg=T["bg2"])
        run_frame.pack(fill="x", padx=8, pady=6)
        self._guided_run_btn = tk.Button(
            run_frame, text="▶  Run Guided Benchmark",
            bg=T["green"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 10, "bold"), pady=5,
            command=self._run_guided
        )
        self._guided_run_btn.pack(side="left")
        self._guided_cancel_btn = tk.Button(
            run_frame, text="■  Cancel",
            bg=T["red"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9), pady=5, state="disabled",
            command=self._cancel_bench
        )
        self._guided_cancel_btn.pack(side="left", padx=6)
        self._guided_status = tk.Label(
            run_frame, text="", bg=T["bg2"], fg=T["fg2"], font=("Consolas", 9)
        )
        self._guided_status.pack(side="left", padx=8)

        # Log area
        self._guided_log = tk.Text(
            parent, height=8, bg=T["log_bg"], fg=T["log_fg"],
            font=("Consolas", 8), relief="flat", state="disabled", padx=6, pady=4
        )
        self._guided_log.pack(fill="x", padx=8, pady=(0, 8))

        self._refresh_hw()
        self._auto_detect_scenario()

    def _refresh_hw(self) -> None:
        hw = self._state.hardware
        if hw and hw.detected:
            from core.hardware import summary_lines
            lines = summary_lines(hw)
            self._hw_label.config(text="\n".join(lines))
        else:
            self._hw_label.config(text="Hardware not yet detected — start the app and wait a moment.")

    def _auto_detect_scenario(self) -> None:
        hw         = self._state.hardware
        model_name = self._state.model_var.get()
        size_gb    = self._state.model_size_gb
        scenario   = optimizer.detect_scenario(hw, model_name, size_gb)
        reason     = optimizer.detect_scenario_reason(hw, model_name, size_gb)
        self._scenario_var.set(scenario)
        self._detect_reason_lbl.config(text=reason)
        self._update_guided_matrix()

    def _update_guided_matrix(self) -> None:
        hw       = self._state.hardware
        scenario = self._scenario_var.get()
        combos   = optimizer.guided_matrix(scenario, hw)
        est      = optimizer.estimated_minutes(len(combos))
        lines    = [f"  {i+1}. {optimizer.combo_summary(c)}" for i, c in enumerate(combos)]
        self._matrix_lbl.config(
            text=f"Test matrix — {len(combos)} runs, {est}:\n" + "\n".join(lines)
        )
        self._guided_combos = combos

    def _run_guided(self) -> None:
        if self._running:
            messagebox.showwarning("Busy", "A benchmark is already running.")
            return
        combos = getattr(self, "_guided_combos", [])
        if not combos:
            messagebox.showwarning("No matrix", "No test matrix. Select a scenario first.")
            return
        self._run_bench(combos, label="guided")

    def _cancel_bench(self) -> None:
        self._cancel_evt.set()

    # ── Sweep ──────────────────────────────────────────────────────────────────

    def _build_sweep(self, parent: tk.Frame) -> None:
        T = self._T
        tk.Label(parent, text="Check a parameter to include it. Edit values (comma-separated).",
                 bg=T["bg2"], fg=T["fg2"], font=("Segoe UI", 8)).pack(anchor="w", padx=12, pady=(8, 2))

        check_frame = tk.Frame(parent, bg=T["bg2"])
        check_frame.pack(fill="x", padx=12, pady=4)
        self._sweep_vars:     dict[str, tk.BooleanVar] = {}
        self._sweep_val_vars: dict[str, tk.StringVar]  = {}

        from core.optimizer import _SWEEP_RANGES
        for key, _, label in SWEEP_PARAMS:
            row = tk.Frame(check_frame, bg=T["bg2"])
            row.pack(fill="x", pady=1)

            bv = tk.BooleanVar(value=False)
            self._sweep_vars[key] = bv
            tk.Checkbutton(row, text=f"{label:<16}", variable=bv,
                           bg=T["bg2"], fg=T["fg"], selectcolor=T["bg3"],
                           activebackground=T["bg2"], activeforeground=T["accent"],
                           font=("Consolas", 9), cursor="hand2", width=16,
                           ).pack(side="left")

            defaults = ", ".join(str(x) for x in _SWEEP_RANGES.get(key, []))
            sv = tk.StringVar(value=defaults)
            self._sweep_val_vars[key] = sv
            ent = tk.Entry(row, textvariable=sv, width=38,
                           bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                           font=("Consolas", 8), insertbackground=T["fg"])
            ent.pack(side="left", padx=4)
            bv.trace_add("write", lambda *_: self._update_sweep_count())
            sv.trace_add("write", lambda *_: self._update_sweep_count())

        self._combo_count_lbl = tk.Label(
            parent, text="Combinations: 0", bg=T["bg2"], fg=T["accent"],
            font=("Consolas", 9)
        )
        self._combo_count_lbl.pack(anchor="w", padx=12, pady=2)

        run_row = tk.Frame(parent, bg=T["bg2"])
        run_row.pack(anchor="w", padx=12, pady=6)
        tk.Button(run_row, text="▶  Run Sweep", bg=T["green"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 10, "bold"), pady=5,
                  command=self._run_sweep).pack(side="left")
        tk.Button(run_row, text="■  Cancel", bg=T["red"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9), pady=5,
                  command=self._cancel_bench).pack(side="left", padx=6)

        self._sweep_log = tk.Text(
            parent, height=8, bg=T["log_bg"], fg=T["log_fg"],
            font=("Consolas", 8), relief="flat", state="disabled", padx=6, pady=4
        )
        self._sweep_log.pack(fill="x", padx=8, pady=(0, 8))

    def _parse_sweep_ranges(self) -> dict[str, list]:
        """Read user-entered comma-separated values per sweep parameter."""
        out: dict[str, list] = {}
        for key, bv in self._sweep_vars.items():
            if not bv.get():
                continue
            raw = self._sweep_val_vars[key].get().strip()
            parsed = []
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    parsed.append(int(part))
                except ValueError:
                    try:
                        parsed.append(float(part))
                    except ValueError:
                        parsed.append(part)
            if parsed:
                out[key] = parsed
        return out

    def _update_sweep_count(self) -> None:
        custom_ranges = self._parse_sweep_ranges()
        hw     = self._state.hardware
        combos = optimizer.sweep_combos_custom(
            custom_ranges, self._state.get_profile_dict(), hw)
        est    = optimizer.estimated_minutes(len(combos))
        self._combo_count_lbl.config(
            text=f"Combinations: {len(combos)}  ({est})")
        self._sweep_combos = combos

    def _run_sweep(self) -> None:
        if self._running:
            messagebox.showwarning("Busy", "A benchmark is already running.")
            return
        custom_ranges = self._parse_sweep_ranges()
        if not custom_ranges:
            messagebox.showwarning("Nothing selected", "Select at least one sweep parameter.")
            return
        hw     = self._state.hardware
        combos = optimizer.sweep_combos_custom(
            custom_ranges, self._state.get_profile_dict(), hw)
        self._run_bench(combos, label="sweep")

    # ── Results ────────────────────────────────────────────────────────────────

    def _build_results(self, parent: tk.Frame) -> None:
        T = self._T

        ctrl = tk.Frame(parent, bg=T["bg2"])
        ctrl.pack(fill="x", padx=8, pady=(8, 4))
        tk.Button(ctrl, text="Refresh", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=self._refresh_results).pack(side="left", padx=2)
        tk.Button(ctrl, text="Export CSV", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=self._export_csv).pack(side="left", padx=2)
        tk.Button(ctrl, text="Apply Best to Profile", bg=T["accent"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9, "bold"),
                  command=self._apply_best).pack(side="right", padx=2)

        # Treeview
        cols = ("timestamp", "pp", "tg", "ngl", "ctx", "batch", "k", "v", "fa")
        self._tree = ttk.Treeview(parent, columns=cols, show="headings", height=15)
        headings = ("Time", "pp t/s", "tg t/s", "ngl", "ctx", "batch", "KV-K", "KV-V", "FA")
        widths   = (130,     70,       70,       50,    60,    60,      70,     70,      40)
        for col, hdr, w in zip(cols, headings, widths):
            self._tree.heading(col, text=hdr,
                               command=lambda c=col: self._sort_by(c))
            self._tree.column(col, width=w, anchor="center")
        self._tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._tree.configure(
            style="App.Treeview"
        )
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")

        self._sort_col    = "tg"
        self._sort_rev    = True
        self._refresh_results()

    def _refresh_results(self) -> None:
        for row in self._tree.get_children():
            self._tree.delete(row)
        results = load_bench_results()
        results.sort(key=lambda r: float(r.get(self._sort_col, 0) or 0),
                     reverse=self._sort_rev)
        for r in results:
            self._tree.insert("", tk.END, values=(
                r.get("timestamp", ""),
                f"{r.get('pp', 0):.1f}",
                f"{r.get('tg', 0):.1f}",
                r.get("ngl", ""),
                r.get("ctx", ""),
                r.get("batch", ""),
                r.get("cache_type_k", ""),
                r.get("cache_type_v", ""),
                "Y" if r.get("flash_attn") else "N",
            ))

    def _sort_by(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = True
        self._refresh_results()

    def _export_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            title="Export results"
        )
        if not path:
            return
        results = load_bench_results()
        if not results:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        self._log(f"[OPT] Exported {len(results)} results to {path}", "success")

    def _apply_best(self) -> None:
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a result row to apply.")
            return
        vals = self._tree.item(sel[0], "values")
        # columns: timestamp, pp, tg, ngl, ctx, batch, k, v, fa
        try:
            self._state.ngl_var.set(int(vals[3]))
            self._state.ctx_var.set(vals[4])
            self._state.batch_var.set(int(vals[5]))
            self._state.cache_type_k_var.set(vals[6])
            self._state.cache_type_v_var.set(vals[7])
            self._state.flash_attn_var.set(vals[8] == "Y")
            self._log("[OPT] Best result applied to current profile vars.", "success")
        except Exception as e:
            self._log(f"[OPT] Apply failed: {e}", "error")

    # ── Benchmark runner (shared by guided + sweep) ────────────────────────────

    def _run_bench(self, combos: list[dict], label: str) -> None:
        s = self._state.settings
        fork_label = self._guided_fork_var.get()
        bench_bin  = s.fork_bench(
            next((f for f in DEFAULT_FORKS if f["label"] == fork_label), DEFAULT_FORKS[0])
        )
        model_wsl  = self._state.settings.models_wsl
        model_name = self._state.model_var.get()
        if not model_name:
            messagebox.showwarning("No Model", "Select a model first.")
            return

        log_widget  = self._guided_log if label == "guided" else self._sweep_log
        status_lbl  = self._guided_status
        cancel_btn  = self._guided_cancel_btn
        run_btn     = self._guided_run_btn
        n_total     = len(combos)

        self._running = True
        self._cancel_evt.clear()
        run_btn.config(state="disabled")
        cancel_btn.config(state="normal")
        status_lbl.config(text=f"0 / {n_total}")

        def _progress(i: int) -> None:
            self._frame.after(0, lambda: status_lbl.config(text=f"{i} / {n_total}"))

        def _done(results: list[dict]) -> None:
            self._running = False
            for r in results:
                save_bench_result(r)
            self._refresh_results()
            cancelled = self._cancel_evt.is_set()
            self._frame.after(0, lambda: (
                run_btn.config(state="normal"),
                cancel_btn.config(state="disabled"),
                status_lbl.config(
                    text=f"Cancelled — {len(results)} saved." if cancelled
                    else f"Done — {len(results)} result(s) saved."
                ),
            ))
            self._log(f"[OPT] {label} benchmark complete: {len(results)} rows.", "success")

        def _log_bench(text: str, tag=None) -> None:
            def _do():
                log_widget.config(state="normal")
                log_widget.insert(tk.END, text + "\n")
                log_widget.see(tk.END)
                log_widget.config(state="disabled")
            self._frame.after(0, _do)

        threading.Thread(
            target=optimizer.run_bench_combos,
            args=(s.wsl_distro, s.wsl_user, bench_bin,
                  model_wsl, model_name, combos, _log_bench, _done,
                  self._cancel_evt, _progress),
            daemon=True,
        ).start()
