"""Optimizer tab — guided goal-based and sweep modes for llama-bench."""
from __future__ import annotations
import csv
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gui.app import AppState

from core.settings import load_bench_results, save_bench_result, DEFAULT_FORKS
import core.optimizer as optimizer

LogFn = Callable[[str, str | None], None]

SWEEP_PARAMS = [
    ("ngl",          "GPU Layers"),
    ("ctx",          "Context"),
    ("batch",        "Batch"),
    ("ubatch",       "µBatch"),
    ("flash_attn",   "Flash Attn"),
    ("cache_type_k", "KV-K type"),
    ("cache_type_v", "KV-V type"),
]


class OptimizerTab:

    def __init__(self, frame: tk.Frame, state: AppState, T: dict, log_fn: LogFn):
        self._frame   = frame
        self._state   = state
        self._T       = T
        self._log     = log_fn
        self._running = False

    def build(self) -> None:
        T = self._T
        nb = ttk.Notebook(self._frame)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        gf = tk.Frame(nb, bg=T["bg2"])
        nb.add(gf, text="Guided")
        self._build_guided(gf)

        sf = tk.Frame(nb, bg=T["bg2"])
        nb.add(sf, text="Sweep")
        self._build_sweep(sf)

        rf = tk.Frame(nb, bg=T["bg2"])
        nb.add(rf, text="Results")
        self._build_results(rf)

    # ── Guided tab ────────────────────────────────────────────────────────────

    def _build_guided(self, parent: tk.Frame) -> None:
        T = self._T
        self._cancel_evt  = threading.Event()
        self._guided_best: dict | None = None

        # Hardware summary
        hw_frame = tk.Frame(parent, bg=T["bg3"])
        hw_frame.pack(fill="x", padx=8, pady=(8, 4))
        self._hw_label = tk.Label(
            hw_frame, text="Detecting hardware…",
            bg=T["bg3"], fg=T["fg2"], font=("Consolas", 8),
            justify="left", padx=8, pady=6
        )
        self._hw_label.pack(side="left", anchor="w", fill="x", expand=True)
        tk.Button(hw_frame, text="Refresh", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 8),
                  command=self._refresh_hw).pack(side="right", padx=8, pady=4)

        # ── Goal selector ──────────────────────────────────────────────────
        tk.Label(parent, text="Goal", bg=T["bg2"], fg=T["fg2"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(8, 0))

        goal_frame = tk.Frame(parent, bg=T["bg2"])
        goal_frame.pack(fill="x", padx=12, pady=(2, 0))
        self._goal_var = tk.StringVar(value=optimizer.GOALS[0])
        for g in optimizer.GOALS:
            tk.Radiobutton(
                goal_frame, text=g, variable=self._goal_var, value=g,
                bg=T["bg2"], fg=T["fg"], selectcolor=T["bg3"],
                activebackground=T["bg2"], activeforeground=T["accent"],
                font=("Segoe UI", 9), cursor="hand2",
                command=self._update_guided_info,
            ).pack(side="left", padx=(0, 14))

        self._goal_desc_lbl = tk.Label(
            parent, text=optimizer.GOAL_DESCRIPTIONS[optimizer.GOALS[0]],
            bg=T["bg2"], fg=T["fg2"], font=("Segoe UI", 8), justify="left",
            wraplength=600
        )
        self._goal_desc_lbl.pack(anchor="w", padx=16, pady=(2, 6))

        # ── Depth + Binary + Estimated time ───────────────────────────────
        opts_frame = tk.Frame(parent, bg=T["bg2"])
        opts_frame.pack(fill="x", padx=12, pady=(0, 4))

        tk.Label(opts_frame, text="Depth:", bg=T["bg2"], fg=T["fg2"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._depth_var = tk.StringVar(value="Quick")
        for d in optimizer.DEPTHS:
            tk.Radiobutton(
                opts_frame, text=d, variable=self._depth_var, value=d,
                bg=T["bg2"], fg=T["fg"], selectcolor=T["bg3"],
                activebackground=T["bg2"], activeforeground=T["accent"],
                font=("Segoe UI", 9), cursor="hand2",
                command=self._update_guided_info,
            ).pack(side="left", padx=(4, 10))

        ttk.Separator(opts_frame, orient="vertical").pack(
            side="left", fill="y", padx=(0, 10), pady=2)

        tk.Label(opts_frame, text="Binary:", bg=T["bg2"], fg=T["fg2"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._guided_fork_var = tk.StringVar(value=DEFAULT_FORKS[0]["label"])
        for f in DEFAULT_FORKS:
            tk.Radiobutton(
                opts_frame, text=f["label"],
                variable=self._guided_fork_var, value=f["label"],
                bg=T["bg2"], fg=T["fg"], selectcolor=T["bg3"],
                activebackground=T["bg2"], activeforeground=T["accent"],
                font=("Segoe UI", 9), cursor="hand2",
                command=self._update_guided_info,
            ).pack(side="left", padx=(4, 10))

        self._est_label = tk.Label(
            opts_frame, text="", bg=T["bg2"], fg=T["accent"],
            font=("Consolas", 8)
        )
        self._est_label.pack(side="right", padx=8)

        # ── Run / Cancel ───────────────────────────────────────────────────
        run_frame = tk.Frame(parent, bg=T["bg2"])
        run_frame.pack(fill="x", padx=12, pady=6)
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

        # ── Probe results (hidden until probe completes) ───────────────────
        self._probe_outer = tk.Frame(parent, bg=T["bg3"])
        # Packed dynamically after probe completes
        self._probe_lbl = tk.Label(
            self._probe_outer, text="", bg=T["bg3"], fg=T["fg2"],
            font=("Consolas", 8), justify="left", padx=10, pady=4
        )
        self._probe_lbl.pack(anchor="w")

        # ── Log area ───────────────────────────────────────────────────────
        log_frame = tk.Frame(parent, bg=T["bg2"])
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self._guided_log = tk.Text(
            log_frame, bg=T["log_bg"], fg=T["log_fg"],
            font=("Consolas", 8), relief="flat", state="disabled",
            padx=6, pady=4
        )
        log_vsb = ttk.Scrollbar(log_frame, orient="vertical",
                                 command=self._guided_log.yview)
        self._guided_log.configure(yscrollcommand=log_vsb.set)
        log_vsb.pack(side="right", fill="y")
        self._guided_log.pack(side="left", fill="both", expand=True)

        # Tag colours for structured log lines
        for tag, colour in [("success", T.get("green", "#4ec9b0")),
                             ("error",   T.get("red",   "#f14c4c")),
                             ("warn",    T.get("accent","#ce9178")),
                             ("info",    T.get("fg2",   "#9cdcfe"))]:
            self._guided_log.tag_configure(tag, foreground=colour)

        # ── Best result banner (hidden until run completes) ────────────────
        self._best_outer = tk.Frame(parent, bg=T["bg3"])
        # Packed dynamically after run completes
        best_inner = tk.Frame(self._best_outer, bg=T["bg3"])
        best_inner.pack(fill="x", padx=8, pady=6)
        self._best_lbl = tk.Label(
            best_inner, text="", bg=T["bg3"], fg=T.get("green", "#4ec9b0"),
            font=("Consolas", 9, "bold"), justify="left"
        )
        self._best_lbl.pack(side="left", anchor="w")
        self._apply_best_btn = tk.Button(
            best_inner, text="Apply to Current Settings",
            bg=T["accent"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9, "bold"),
            command=self._apply_best_guided
        )
        self._apply_best_btn.pack(side="right", padx=4)

        self._refresh_hw()
        self._update_guided_info()

    def _refresh_hw(self) -> None:
        hw = self._state.hardware
        if hw and hw.detected:
            from core.hardware import summary_lines
            self._hw_label.config(text="  ".join(summary_lines(hw)))
        else:
            self._hw_label.config(
                text="Hardware not yet detected — start the server once to populate this.")

    def _update_guided_info(self) -> None:
        goal       = self._goal_var.get()
        depth      = self._depth_var.get()
        fork_label = self._guided_fork_var.get()
        kv_pairs   = optimizer.kv_pairs_for_binary(fork_label)

        self._goal_desc_lbl.config(
            text=optimizer.GOAL_DESCRIPTIONS.get(goal, ""))

        n_combos = optimizer.guided_combo_count(goal, depth, kv_pairs)
        n_probes = len(kv_pairs) if optimizer.needs_probe(goal) else 0
        est      = optimizer.estimated_minutes(n_combos, n_probes)
        probe_note = f" + {n_probes} probe(s)" if n_probes else ""
        self._est_label.config(
            text=f"{n_combos} run(s){probe_note}  ·  {est}")

    def _run_guided(self) -> None:
        if self._running:
            messagebox.showwarning("Busy", "A benchmark is already running.")
            return

        s          = self._state.settings
        fork_label = self._guided_fork_var.get()
        bench_bin  = s.fork_bench(
            next((f for f in DEFAULT_FORKS if f["label"] == fork_label),
                 DEFAULT_FORKS[0])
        )

        model_full = self._state.model_var.get().strip()
        if not model_full:
            messagebox.showwarning("No Model", "Select a model first.")
            return
        model_path = model_full.replace("~", "$HOME")

        goal     = self._goal_var.get()
        depth    = self._depth_var.get()
        kv_pairs = optimizer.kv_pairs_for_binary(fork_label)
        ngl      = int(self._state.ngl_var.get())

        threads_raw = self._state.threads_var.get()
        try:
            threads = int(str(threads_raw))
        except (ValueError, TypeError):
            threads = 4

        n_combos = optimizer.guided_combo_count(goal, depth, kv_pairs)
        n_probes = len(kv_pairs) if optimizer.needs_probe(goal) else 0

        self._running = True
        self._guided_best = None
        self._cancel_evt.clear()

        # Reset UI
        self._best_outer.pack_forget()
        self._probe_outer.pack_forget()
        self._probe_lbl.config(text="")
        self._guided_run_btn.config(state="disabled")
        self._guided_cancel_btn.config(state="normal")
        self._guided_status.config(text=f"Starting…  (0 / {n_combos})")

        # Clear log
        self._guided_log.config(state="normal")
        self._guided_log.delete("1.0", tk.END)
        self._guided_log.config(state="disabled")

        def _log_bench(text: str, tag=None) -> None:
            def _do():
                self._guided_log.config(state="normal")
                if tag:
                    self._guided_log.insert(tk.END, text + "\n", tag)
                else:
                    self._guided_log.insert(tk.END, text + "\n")
                self._guided_log.see(tk.END)
                self._guided_log.config(state="disabled")
            self._frame.after(0, _do)

        def _probe_done(probe_results: dict) -> None:
            lines = ["VRAM probe results:"]
            for (k, v), ctx in probe_results.items():
                if ctx:
                    lines.append(f"  {k}/{v}: max {ctx:,} tokens")
                else:
                    lines.append(f"  {k}/{v}: OOM even at minimum context — excluded")
            text = "\n".join(lines)
            def _do():
                self._probe_lbl.config(text=text)
                self._probe_outer.pack(fill="x", padx=8, pady=(0, 4),
                                       before=self._guided_log.master)
            self._frame.after(0, _do)

        def _progress(i: int) -> None:
            self._frame.after(0, lambda: self._guided_status.config(
                text=f"Run {i} / {n_combos}"))

        def _done(results: list[dict]) -> None:
            self._running = False
            for r in results:
                save_bench_result(r)
            self._refresh_results()

            scored = optimizer.score_results(results, goal) if results else []
            self._guided_best = scored[0] if scored else None

            def _finish():
                self._guided_run_btn.config(state="normal")
                self._guided_cancel_btn.config(state="disabled")
                cancelled = self._cancel_evt.is_set()

                if self._guided_best:
                    best   = self._guided_best
                    metric = optimizer.GOAL_METRIC.get(goal, "tg")
                    if metric == "ctx":
                        score_str = f"ctx={best.get('ctx'):,}"
                    else:
                        score_str = f"{metric}={best.get(metric, 0):.1f} t/s"
                    summary = (
                        f"Best ({goal}):  {score_str}  │  "
                        f"KV {best.get('cache_type_k')}/{best.get('cache_type_v')}  │  "
                        f"FA {'on' if best.get('flash_attn') else 'off'}  │  "
                        f"batch={best.get('batch')}  │  "
                        f"ctx={best.get('ctx')}"
                    )
                    self._best_lbl.config(text=summary)
                    self._best_outer.pack(fill="x", padx=8, pady=4,
                                          before=self._guided_log.master)
                    status = (f"Cancelled — {len(results)} saved." if cancelled
                              else f"Done — {len(results)} result(s)")
                else:
                    status = "Done — no successful runs"
                self._guided_status.config(text=status)

            self._frame.after(0, _finish)
            if results:
                self._log(f"[OPT] Guided benchmark ({goal}) complete: "
                          f"{len(results)} result(s).", "success")

        threading.Thread(
            target=optimizer.run_guided_bench,
            args=(s.wsl_distro, s.wsl_user, bench_bin, model_path,
                  goal, depth, kv_pairs, ngl, threads,
                  _log_bench, _done, self._cancel_evt, _progress, _probe_done),
            daemon=True,
        ).start()

    def _apply_best_guided(self) -> None:
        best = self._guided_best
        if not best:
            return
        try:
            if best.get("cache_type_k"):
                self._state.cache_type_k_var.set(best["cache_type_k"])
            if best.get("cache_type_v"):
                self._state.cache_type_v_var.set(best["cache_type_v"])
            self._state.flash_attn_var.set(bool(best.get("flash_attn", False)))
            if best.get("batch"):
                self._state.batch_var.set(int(best["batch"]))
            if best.get("ubatch"):
                self._state.ubatch_var.set(int(best["ubatch"]))
            self._log("[OPT] Best settings applied to current vars. "
                      "Restart server to use them.", "success")
        except Exception as e:
            self._log(f"[OPT] Apply failed: {e}", "error")

    def _cancel_bench(self) -> None:
        self._cancel_evt.set()

    # ── Sweep tab ─────────────────────────────────────────────────────────────

    def _build_sweep(self, parent: tk.Frame) -> None:
        T = self._T
        tk.Label(parent,
                 text="Check a parameter to include it. Edit values (comma-separated).",
                 bg=T["bg2"], fg=T["fg2"], font=("Segoe UI", 8),
                 ).pack(anchor="w", padx=12, pady=(8, 2))

        check_frame = tk.Frame(parent, bg=T["bg2"])
        check_frame.pack(fill="x", padx=12, pady=4)
        self._sweep_vars:     dict[str, tk.BooleanVar] = {}
        self._sweep_val_vars: dict[str, tk.StringVar]  = {}

        from core.optimizer import _SWEEP_RANGES
        for key, label in SWEEP_PARAMS:
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
            tk.Entry(row, textvariable=sv, width=38,
                     bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                     font=("Consolas", 8), insertbackground=T["fg"]
                     ).pack(side="left", padx=4)
            bv.trace_add("write", lambda *_: self._update_sweep_count())
            sv.trace_add("write", lambda *_: self._update_sweep_count())

        self._combo_count_lbl = tk.Label(
            parent, text="Combinations: 0", bg=T["bg2"], fg=T["accent"],
            font=("Consolas", 9)
        )
        self._combo_count_lbl.pack(anchor="w", padx=12, pady=2)

        # Fork selector
        sfk_frame = tk.Frame(parent, bg=T["bg2"])
        sfk_frame.pack(fill="x", padx=12, pady=2)
        tk.Label(sfk_frame, text="Bench binary:", bg=T["bg2"], fg=T["fg2"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._sweep_fork_var = tk.StringVar(value=DEFAULT_FORKS[0]["label"])
        for f in DEFAULT_FORKS:
            tk.Radiobutton(sfk_frame, text=f["label"],
                           variable=self._sweep_fork_var, value=f["label"],
                           bg=T["bg2"], fg=T["fg"], selectcolor=T["bg3"],
                           activebackground=T["bg2"], activeforeground=T["accent"],
                           font=("Segoe UI", 9), cursor="hand2",
                           ).pack(side="left", padx=4)

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
        out: dict[str, list] = {}
        for key, bv in self._sweep_vars.items():
            if not bv.get():
                continue
            raw    = self._sweep_val_vars[key].get().strip()
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
            messagebox.showwarning("Nothing selected",
                                   "Select at least one sweep parameter.")
            return
        hw     = self._state.hardware
        combos = optimizer.sweep_combos_custom(
            custom_ranges, self._state.get_profile_dict(), hw)
        self._run_sweep_bench(combos)

    def _run_sweep_bench(self, combos: list[dict]) -> None:
        s          = self._state.settings
        fork_label = self._sweep_fork_var.get()
        bench_bin  = s.fork_bench(
            next((f for f in DEFAULT_FORKS if f["label"] == fork_label),
                 DEFAULT_FORKS[0])
        )
        model_full = self._state.model_var.get().strip()
        if not model_full:
            messagebox.showwarning("No Model", "Select a model first.")
            return
        model_path = model_full.replace("~", "$HOME")

        n_total = len(combos)
        self._running = True
        self._cancel_evt.clear()

        def _log_sweep(text: str, tag=None) -> None:
            def _do():
                self._sweep_log.config(state="normal")
                self._sweep_log.insert(tk.END, text + "\n")
                self._sweep_log.see(tk.END)
                self._sweep_log.config(state="disabled")
            self._frame.after(0, _do)

        def _done(results: list[dict]) -> None:
            self._running = False
            for r in results:
                save_bench_result(r)
            self._refresh_results()
            self._log(f"[OPT] Sweep complete: {len(results)} result(s).", "success")

        threading.Thread(
            target=optimizer.run_bench_combos,
            args=(s.wsl_distro, s.wsl_user, bench_bin, model_path,
                  combos, _log_sweep, _done, self._cancel_evt, None),
            daemon=True,
        ).start()

    # ── Results tab ───────────────────────────────────────────────────────────

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
        tk.Button(ctrl, text="Apply Selected to Profile Vars", bg=T["accent"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9, "bold"),
                  command=self._apply_selected).pack(side="right", padx=2)

        cols     = ("timestamp", "goal", "pp", "tg", "ngl", "ctx", "batch", "k", "v", "fa")
        headings = ("Time",      "Goal", "pp t/s", "tg t/s", "ngl", "ctx", "batch",
                    "KV-K", "KV-V", "FA")
        widths   = (120,          80,    70,       70,       50,    60,    55,
                    70,     70,    40)

        frame = tk.Frame(parent, bg=T["bg2"])
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._tree = ttk.Treeview(frame, columns=cols, show="headings", height=15)
        for col, hdr, w in zip(cols, headings, widths):
            self._tree.heading(col, text=hdr,
                               command=lambda c=col: self._sort_by(c))
            self._tree.column(col, width=w, anchor="center")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._tree.pack(side="left", fill="both", expand=True)

        self._sort_col = "tg"
        self._sort_rev = True
        self._refresh_results()

    def _refresh_results(self) -> None:
        if not hasattr(self, "_tree"):
            return
        for row in self._tree.get_children():
            self._tree.delete(row)
        results = load_bench_results()
        results.sort(key=lambda r: float(r.get(self._sort_col, 0) or 0),
                     reverse=self._sort_rev)
        for r in results:
            self._tree.insert("", tk.END, values=(
                r.get("timestamp", ""),
                r.get("goal", ""),
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
        import csv as _csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        self._log(f"[OPT] Exported {len(results)} results to {path}", "success")

    def _apply_selected(self) -> None:
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a result row first.")
            return
        vals = self._tree.item(sel[0], "values")
        # cols: timestamp, goal, pp, tg, ngl, ctx, batch, k, v, fa
        try:
            self._state.ngl_var.set(int(vals[4]))
            self._state.ctx_var.set(vals[5])
            self._state.batch_var.set(int(vals[6]))
            self._state.cache_type_k_var.set(vals[7])
            self._state.cache_type_v_var.set(vals[8])
            self._state.flash_attn_var.set(vals[9] == "Y")
            self._log("[OPT] Selected result applied to current profile vars. "
                      "Restart server to use.", "success")
        except Exception as e:
            self._log(f"[OPT] Apply failed: {e}", "error")
