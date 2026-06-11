"""Model tab — ngl, ctx, batch, KV cache types, mlock, moe, flash-attn."""
from __future__ import annotations
import re
import tkinter as tk
from tkinter import ttk
from typing import Callable

from gui.widgets import (ToolTip, DynamicToolTip, section, sep, spinbox, combo,
                         cbk, flag_row, grid_frame)

LogFn = Callable[[str, str | None], None]

KV_TYPES_OFFICIAL = ["f16","bf16","q8_0","q5_1","q5_0","q4_1","q4_0","iq4_nl","f32"]
KV_TYPES_TURBO    = ["turbo4","turbo3","turbo2"] + KV_TYPES_OFFICIAL
CTX_VALUES = ["512","1024","2048","4096","8192","16384","32768","65536","131072","262144"]

_TC_QUANT_RE = re.compile(
    r'(?:[-_.])(IQ[1-5]_[A-Za-z0-9_]+|Q[2-9]_[A-Za-z0-9_]+|BF16|F16|F32)(?=[-_.]|$)',
    re.IGNORECASE,
)


class ModelTab:

    def __init__(self, frame: tk.Frame, state, T: dict, log_fn: LogFn):
        self._frame   = frame
        self._state   = state
        self._T       = T
        self._log     = log_fn
        self._tc_auto = False   # True = current tokenizer path was auto-detected

    def build(self) -> None:
        T  = self._T
        sf = self._scrollable_frame()

        section(sf, "MODEL LOADING", T)
        g = grid_frame(sf)
        r = 0

        # NGL slider + spinbox
        tk.Label(g, text="GPU Layers (-ngl)", bg=g.cget("bg"), fg=T["fg2"],
                 font=("Segoe UI", 9)).grid(row=r, column=0, sticky="w", padx=(0,8), pady=2)
        nf = tk.Frame(g, bg=g.cget("bg"))
        nf.grid(row=r, column=1, sticky="ew", pady=2)
        ns = tk.Scale(nf, from_=0, to=100, orient="horizontal",
                      variable=self._state.ngl_var,
                      bg=g.cget("bg"), fg=T["fg"], troughcolor=T["bar_bg"],
                      highlightthickness=0, activebackground=T["accent"],
                      length=110, showvalue=False)
        ns.pack(side="left")
        tk.Spinbox(nf, from_=0, to=999, textvariable=self._state.ngl_var, width=5,
                   bg=T["entry_bg"], fg=T["entry_fg"], buttonbackground=T["btn"],
                   relief="flat", font=("Consolas", 10)).pack(side="left", padx=3)
        ToolTip(ns, "Layers offloaded to GPU. 99 = all layers.")
        r += 1

        combo(g, r, "Context size (-c)", self._state.ctx_var, CTX_VALUES,
              tip="Max context window in tokens. Must be ≤ model's trained max."); r += 1
        spinbox(g, r, "Batch size (-b)",     self._state.batch_var,   64, 4096, 64,
                tip="Tokens per batch during prompt ingestion."); r += 1
        spinbox(g, r, "Micro-batch (-ub)",   self._state.ubatch_var,  64, 4096, 64,
                tip="Physical micro-batch size. Must be ≤ batch size."); r += 1
        spinbox(g, r, "Threads (-t)",        self._state.threads_var, 1,  64,
                tip="CPU threads for generation."); r += 1
        spinbox(g, r, "Batch threads (-tb)", self._state.threads_batch_var, -1, 64,
                tip="CPU threads for batch processing. -1 = same as -t."); r += 1
        self._k_combo = combo(g, r, "KV cache K (--cache-type-k)", self._state.cache_type_k_var,
                              KV_TYPES_TURBO,
                              tip="Key cache type. turbo3 recommended for K with TurboQuant binary."); r += 1
        self._v_combo = combo(g, r, "KV cache V (--cache-type-v)", self._state.cache_type_v_var,
                              KV_TYPES_TURBO,
                              tip="Value cache type. turbo4 recommended for V with TurboQuant binary."); r += 1

        # Keep combo lists in sync with the selected binary
        self._state.llama_bin_var.trace_add("write", lambda *_: self._on_bin_change())
        self._on_bin_change()  # apply initial state

        sep(sf, T)
        section(sf, "MODEL LOADING FLAGS", T)
        mf = tk.Frame(sf, bg=sf.cget("bg"))
        mf.pack(fill="x", padx=12, pady=4)

        cbk(mf, "--flash-attn  (-fa)", self._state.flash_attn_var,
            "Flash Attention: reduces VRAM for long contexts. Recommended for ctx > 8192.")
        cbk(mf, "--mlock", self._state.mlock_var,
            "Pin model in RAM/VRAM to prevent OS swapping.")

        # mlock fix button
        fix_row = tk.Frame(mf, bg=mf.cget("bg"))
        fix_row.pack(anchor="w", padx=22, pady=(0, 3))
        from core.wsl import fix_mlock
        s = self._state.settings
        tk.Button(fix_row, text="Fix mlock limits (run once)",
                  bg=T["btn"], fg=T["btn_fg"], relief="flat", cursor="hand2",
                  font=("Segoe UI", 8),
                  command=lambda: fix_mlock(s.wsl_distro, self._log)
                  ).pack(side="left")

        cbk(mf, "--no-mmap", self._state.no_mmap_var,
            "Force full model load into RAM. Required when using --cpu-moe.")
        cbk(mf, "--cpu-moe  (MoE experts → CPU)", self._state.cpu_moe_var,
            "For MoE models: keep expert tensors in CPU RAM. Always pair with --no-mmap.")
        flag_row(mf, "--n-cpu-moe", self._state.n_cpu_moe_en_var,
                 self._state.n_cpu_moe_var, "entry",
                 "Keep first N layers' MoE experts on CPU. Requires --no-mmap.",
                 val_width=5)
        cbk(mf, "--jinja  (tool calling)", self._state.jinja_var,
            "Use model's Jinja2 chat template. Required for native tool calling.")
        cbk(mf, "--no-warmup", self._state.no_warmup_var,
            "Skip startup inference pass. Server ready faster; first request slightly slower.")

        sep(sf, T)
        section(sf, "TOKENIZER CONFIG", T)
        tc_outer = tk.Frame(sf, bg=sf.cget("bg"))
        tc_outer.pack(fill="x", padx=12, pady=4)

        tc_row = tk.Frame(tc_outer, bg=tc_outer.cget("bg"))
        tc_row.pack(fill="x", anchor="w")

        tc_cb = tk.Checkbutton(
            tc_row, text="--chat-template-file",
            variable=self._state.tokenizer_config_en_var,
            bg=tc_row.cget("bg"), fg=T["fg"],
            activebackground=tc_row.cget("bg"), activeforeground=T["accent"],
            selectcolor=T["bg3"], relief="flat",
            font=("Consolas", 9), cursor="hand2",
        )
        tc_cb.pack(side="left")
        ToolTip(tc_cb,
            "Load Jinja2 chat template from an external .jinja file.\n"
            "Essential when the GGUF has no embedded template (common in quantized releases).\n"
            "Hover the indicator dot for the active file path.")

        self._tc_indicator = tk.Label(
            tc_row, text="· not set",
            bg=tc_row.cget("bg"), fg=T["fg2"],
            font=("Consolas", 9), cursor="hand2",
        )
        self._tc_indicator.pack(side="left", padx=(12, 0))
        DynamicToolTip(self._tc_indicator,
            lambda: self._state.tokenizer_config_var.get() or "No tokenizer config file set")

        self._tc_entry = tk.Entry(
            tc_outer, textvariable=self._state.tokenizer_config_var,
            bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
            font=("Consolas", 9), insertbackground=T["fg"],
        )
        self._tc_entry.pack(fill="x", pady=(4, 0))
        self._tc_entry.bind("<FocusIn>", lambda e: setattr(self, "_tc_auto", False))

        self._state.tokenizer_config_en_var.trace_add("write", lambda *_: self._on_tc_change())
        self._state.tokenizer_config_var.trace_add("write",    lambda *_: self._on_tc_change())
        self._state.model_var.trace_add("write", lambda *_: self._auto_detect_tokenizer())
        self._on_tc_change()
        self._auto_detect_tokenizer()

        sep(sf, T)
        section(sf, "MULTI-GPU", T)
        g2 = grid_frame(sf)

        tk.Label(g2, text="Main GPU (--main-gpu)", bg=g2.cget("bg"), fg=T["fg2"],
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=(0,8), pady=2)
        mgf = tk.Frame(g2, bg=g2.cget("bg"))
        mgf.grid(row=0, column=1, sticky="w", pady=2)
        for idx, lbl in enumerate(["GPU 0", "GPU 1"]):
            tk.Radiobutton(mgf, text=lbl, variable=self._state.main_gpu_var, value=idx,
                           bg=g2.cget("bg"), fg=T["fg"], selectcolor=T["bg3"],
                           activebackground=g2.cget("bg"), activeforeground=T["accent"],
                           font=("Segoe UI", 9), cursor="hand2").pack(side="left", padx=4)

        # Tensor split — percentage slider spanning both columns
        split_outer = tk.Frame(g2, bg=g2.cget("bg"))
        split_outer.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 2))

        en_cb = tk.Checkbutton(split_outer, text="--tensor-split",
                               variable=self._state.tensor_split_en_var,
                               bg=split_outer.cget("bg"), fg=T["fg"],
                               activebackground=split_outer.cget("bg"),
                               activeforeground=T["accent"],
                               selectcolor=T["bg3"], relief="flat",
                               font=("Consolas", 9), cursor="hand2")
        en_cb.pack(side="left")
        ToolTip(en_cb, "Enable multi-GPU tensor split. Distributes model layers across GPUs.")

        bg = split_outer.cget("bg")
        self._ts_lbl0 = tk.Label(split_outer, text="GPU 0: 60%",
                                 bg=bg, fg=T["accent"], font=("Consolas", 9), width=10, anchor="e")
        self._ts_lbl0.pack(side="left", padx=(10, 2))

        self._ts_slider = tk.Scale(
            split_outer, from_=5, to=95, resolution=5, orient="horizontal",
            variable=self._state.tensor_split_pct_var,
            bg=bg, fg=T["fg"], troughcolor=T["bar_bg"],
            highlightthickness=0, activebackground=T["accent"],
            length=160, showvalue=False,
        )
        self._ts_slider.pack(side="left", padx=2)
        ToolTip(self._ts_slider,
                "Proportion of model offloaded to each GPU.\n"
                "GPU 0 gets X%, GPU 1 gets (100−X)%.\n"
                "With CUDA swap on, GPU 0 here = CUDA device 0 (RTX 3060 12GB).")

        self._ts_lbl1 = tk.Label(split_outer, text="GPU 1: 40%",
                                 bg=bg, fg=T["accent"], font=("Consolas", 9), width=10, anchor="w")
        self._ts_lbl1.pack(side="left", padx=(2, 0))

        def _on_pct(*_):
            pct = self._state.tensor_split_pct_var.get()
            self._ts_lbl0.config(text=f"GPU 0: {pct}%")
            self._ts_lbl1.config(text=f"GPU 1: {100-pct}%")
            if self._state.tensor_split_en_var.get():
                self._state.tensor_split_var.set(f"{pct},{100-pct}")

        def _on_enable(*_):
            enabled = self._state.tensor_split_en_var.get()
            self._ts_slider.config(state="normal" if enabled else "disabled")
            if enabled:
                pct = self._state.tensor_split_pct_var.get()
                self._state.tensor_split_var.set(f"{pct},{100-pct}")
            else:
                self._state.tensor_split_var.set("")

        self._state.tensor_split_pct_var.trace_add("write", _on_pct)
        self._state.tensor_split_en_var.trace_add("write", _on_enable)
        _on_pct()
        _on_enable()

    def _auto_detect_tokenizer(self) -> None:
        model = self._state.model_var.get()
        if not model or not model.lower().endswith(".gguf"):
            return
        s   = self._state.settings
        unc = s.models_unc
        if not unc:
            return
        from pathlib import Path
        stem = model[:-5]
        quant_pos = len(stem)
        for m in _TC_QUANT_RE.finditer(stem):
            quant_pos = m.start()
        clean = stem[:quant_pos].rstrip("-_.")
        found_wsl = None
        candidates = [
            f"{stem}-chat-template.jinja",
            f"{clean}-chat-template.jinja",
            f"{stem}-tokenizer_config.json",
            f"{clean}-tokenizer_config.json",
        ]
        for name in candidates:
            try:
                if (Path(unc) / name).exists():
                    found_wsl = f"{s.models_wsl}/{name}"
                    break
            except Exception:
                pass
        if found_wsl:
            self._tc_auto = True
            self._state.tokenizer_config_var.set(found_wsl)
            self._state.tokenizer_config_en_var.set(True)
        elif self._tc_auto:
            self._tc_auto = False
            self._state.tokenizer_config_var.set("")
            self._state.tokenizer_config_en_var.set(False)
        self._update_tc_indicator()

    def _on_tc_change(self) -> None:
        en = self._state.tokenizer_config_en_var.get()
        try:
            self._tc_entry.config(state="normal" if en else "disabled")
        except Exception:
            pass
        self._update_tc_indicator()

    def _update_tc_indicator(self) -> None:
        T    = self._T
        en   = self._state.tokenizer_config_en_var.get()
        path = self._state.tokenizer_config_var.get().strip()
        try:
            if not en or not path:
                self._tc_indicator.config(text="· not set", fg=T["fg2"])
            elif self._tc_auto:
                self._tc_indicator.config(text="● auto", fg=T["green"])
            else:
                self._tc_indicator.config(text="● set",  fg=T["accent"])
        except Exception:
            pass

    def _on_bin_change(self) -> None:
        """Swap KV cache type lists based on whether TurboQuant binary is selected."""
        # llama_bin_var stores the fork label ("TurboQuant"), not the path
        label    = self._state.llama_bin_var.get().lower()
        is_turbo = "turbo" in label
        types = KV_TYPES_TURBO if is_turbo else KV_TYPES_OFFICIAL
        try:
            self._k_combo["values"] = types
            self._v_combo["values"] = types
        except AttributeError:
            return  # called before combos exist
        # If a turbo type is currently selected but turbo binary isn't active, reset
        if not is_turbo:
            for var in (self._state.cache_type_k_var, self._state.cache_type_v_var):
                if var.get() not in KV_TYPES_OFFICIAL:
                    var.set("f16")

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
