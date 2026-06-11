"""
LlamaApp — root window, AppState, tab notebook.
AppState holds all tk.Var instances and live process state.
All other GUI modules receive a reference to AppState.
"""
from __future__ import annotations
import subprocess
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Callable

from core.settings import (
    AppSettings, DEFAULT_PROFILE, DEFAULT_FORKS,
    load_settings, save_settings,
    load_profiles, save_profiles,
    load_agents, save_agents,
)
from core.hardware import HardwareProfile, detect as detect_hardware
from core.monitor  import Monitor, MonitorSnapshot
from core.server   import ServerController, HealthChecker, ServerState, build_command
from gui.themes    import get as get_theme, THEME_LABELS, DEFAULT_THEME

LogFn = Callable[[str, str | None], None]

_MAX_LOG_BUFFER = 3000


def _ts_to_pct(ts: str) -> int:
    """Parse any --tensor-split value (e.g. '3,2' or '60,40') → GPU 0 integer %."""
    parts = [p.strip() for p in ts.replace(";", ",").split(",") if p.strip()]
    if len(parts) >= 2:
        try:
            v0, v1 = float(parts[0]), float(parts[1])
            if v0 + v1 > 0:
                return max(5, min(95, round(v0 / (v0 + v1) * 100)))
        except ValueError:
            pass
    return 60


# ── AppState ──────────────────────────────────────────────────────────────────

class AppState:
    """
    Central shared state. Holds all tk.Var instances and live process handles.
    Constructed after the tk.Tk root exists. Passed by reference to every panel/tab.
    """

    def __init__(self, root: tk.Tk, settings: AppSettings,
                 profiles: dict, agents: list, hardware: HardwareProfile):
        self.root     = root
        self.settings = settings
        self.profiles = profiles
        self.agents   = agents
        self.hardware = hardware

        # ── Live process state ─────────────────────────────────────────────────
        self.server_ctrl: ServerController | None = None
        self.hermes_process: subprocess.Popen | None = None
        self._scroll_paused  = False
        self._rebuilding     = False

        # ── Server / Model vars ────────────────────────────────────────────────
        self.model_var              = tk.StringVar()
        self.profile_var            = tk.StringVar()
        self.llama_bin_var          = tk.StringVar(value=settings.llama_bin or DEFAULT_FORKS[0]["bin_rel"])
        self.port_var               = tk.StringVar(value=str(DEFAULT_PROFILE["port"]))
        self.api_key_server_var     = tk.StringVar()
        self.parallel_var           = tk.IntVar(value=DEFAULT_PROFILE["parallel"])
        self.cont_batching_var      = tk.BooleanVar(value=DEFAULT_PROFILE["cont_batching"])
        self.embeddings_var         = tk.BooleanVar(value=DEFAULT_PROFILE["embeddings"])
        self.metrics_var            = tk.BooleanVar(value=DEFAULT_PROFILE["metrics"])
        self.props_endpoint_var     = tk.BooleanVar(value=DEFAULT_PROFILE["props_endpoint"])
        self.slots_endpoint_var     = tk.BooleanVar(value=DEFAULT_PROFILE["slots_endpoint"])
        self.threads_http_var       = tk.IntVar(value=DEFAULT_PROFILE["threads_http"])
        self.alias_var              = tk.StringVar()
        self.server_timeout_var     = tk.StringVar(value=str(DEFAULT_PROFILE["server_timeout"]))

        # ── Model loading vars ─────────────────────────────────────────────────
        self.ngl_var                = tk.IntVar(value=DEFAULT_PROFILE["ngl"])
        self.ctx_var                = tk.StringVar(value=str(DEFAULT_PROFILE["ctx"]))
        self.batch_var              = tk.IntVar(value=DEFAULT_PROFILE["batch"])
        self.ubatch_var             = tk.IntVar(value=DEFAULT_PROFILE["ubatch"])
        self.threads_var            = tk.IntVar(value=DEFAULT_PROFILE["threads"])
        self.threads_batch_var      = tk.IntVar(value=DEFAULT_PROFILE["threads_batch"])
        self.cache_type_k_var       = tk.StringVar(value=DEFAULT_PROFILE["cache_type_k"])
        self.cache_type_v_var       = tk.StringVar(value=DEFAULT_PROFILE["cache_type_v"])
        self.mlock_var              = tk.BooleanVar(value=DEFAULT_PROFILE["mlock"])
        self.no_mmap_var            = tk.BooleanVar(value=DEFAULT_PROFILE["no_mmap"])
        self.flash_attn_var         = tk.BooleanVar(value=DEFAULT_PROFILE["flash_attn"])
        self.cpu_moe_var            = tk.BooleanVar(value=DEFAULT_PROFILE["cpu_moe"])
        self.n_cpu_moe_en_var       = tk.BooleanVar(value=DEFAULT_PROFILE["n_cpu_moe"])
        self.n_cpu_moe_var          = tk.StringVar(value=str(DEFAULT_PROFILE["n_cpu_moe_n"]))
        self.no_warmup_var          = tk.BooleanVar(value=DEFAULT_PROFILE["no_warmup"])
        self.tokenizer_config_en_var = tk.BooleanVar(value=False)
        self.tokenizer_config_var    = tk.StringVar(value="")
        self.tensor_split_var       = tk.StringVar(value=DEFAULT_PROFILE["tensor_split"])
        self.tensor_split_pct_var   = tk.IntVar(value=60)   # GPU 0 share 0-100
        self.tensor_split_en_var    = tk.BooleanVar(value=False)
        self.main_gpu_var           = tk.IntVar(value=DEFAULT_PROFILE["main_gpu"])
        self.no_display_prompt_var  = tk.BooleanVar(value=DEFAULT_PROFILE["no_display_prompt"])
        self.jinja_var              = tk.BooleanVar(value=DEFAULT_PROFILE["jinja"])
        self.extra_flags_var        = tk.StringVar()

        # ── Sampling vars ──────────────────────────────────────────────────────
        self.temp_var               = tk.DoubleVar(value=DEFAULT_PROFILE["temperature"])
        self.top_k_var              = tk.IntVar(value=DEFAULT_PROFILE["top_k"])
        self.top_p_var              = tk.DoubleVar(value=DEFAULT_PROFILE["top_p"])
        self.min_p_var              = tk.DoubleVar(value=DEFAULT_PROFILE["min_p"])
        self.repeat_penalty_var     = tk.DoubleVar(value=DEFAULT_PROFILE["repeat_penalty"])
        self.repeat_last_n_var      = tk.IntVar(value=DEFAULT_PROFILE["repeat_last_n"])
        self.presence_penalty_var   = tk.DoubleVar(value=DEFAULT_PROFILE["presence_penalty"])
        self.frequency_penalty_var  = tk.DoubleVar(value=DEFAULT_PROFILE["frequency_penalty"])
        self.seed_var               = tk.IntVar(value=DEFAULT_PROFILE["seed"])
        self.mirostat_var           = tk.IntVar(value=DEFAULT_PROFILE["mirostat"])
        self.mirostat_lr_var        = tk.DoubleVar(value=DEFAULT_PROFILE["mirostat_lr"])
        self.mirostat_ent_var       = tk.DoubleVar(value=DEFAULT_PROFILE["mirostat_ent"])
        self.predict_var            = tk.IntVar(value=DEFAULT_PROFILE["predict"])

        # ── Context / RoPE vars ────────────────────────────────────────────────
        self.rope_freq_base_var     = tk.DoubleVar(value=DEFAULT_PROFILE["rope_freq_base"])
        self.rope_scaling_var       = tk.StringVar(value=DEFAULT_PROFILE["rope_scaling"])

        # ── Performance flag vars ──────────────────────────────────────────────
        self.prio_en_var            = tk.BooleanVar(value=DEFAULT_PROFILE["flag_prio"])
        self.prio_level_var         = tk.StringVar(value=str(DEFAULT_PROFILE["flag_prio_level"]))
        self.prio_batch_en_var      = tk.BooleanVar(value=DEFAULT_PROFILE["flag_prio_batch"])
        self.prio_batch_level_var   = tk.StringVar(value=str(DEFAULT_PROFILE["flag_prio_batch_level"]))
        self.cache_reuse_en_var     = tk.BooleanVar(value=DEFAULT_PROFILE["flag_cache_reuse"])
        self.cache_reuse_n_var      = tk.StringVar(value=str(DEFAULT_PROFILE["flag_cache_reuse_n"]))

        # ── Speculative decoding vars ──────────────────────────────────────────
        self.spec_mtp_var           = tk.BooleanVar(value=DEFAULT_PROFILE["flag_spec_mtp"])
        self.spec_draft_n_en_var    = tk.BooleanVar(value=DEFAULT_PROFILE["flag_spec_draft_n"])
        self.spec_draft_n_var       = tk.StringVar(value=str(DEFAULT_PROFILE["flag_spec_draft_n_max"]))
        self.prio_draft_en_var      = tk.BooleanVar(value=DEFAULT_PROFILE["flag_prio_draft"])
        self.prio_draft_level_var   = tk.StringVar(value=str(DEFAULT_PROFILE["flag_prio_draft_level"]))

        # ── WSL / misc vars ────────────────────────────────────────────────────
        self.wsl_memory_var         = tk.StringVar(value=settings.wsl_memory)
        self.proxy_bypass_var       = tk.BooleanVar(value=settings.proxy_bypass)
        self.cuda_swap_var          = tk.BooleanVar(value=settings.cuda_swap)

    # ── Profile helpers ────────────────────────────────────────────────────────

    def get_profile_dict(self) -> dict:
        """Snapshot all current parameter vars into a profile dict."""
        return {
            "model":                self.model_var.get(),
            "llama_bin":            self.llama_bin_var.get(),
            "ngl":                  self.ngl_var.get(),
            "ctx":                  self.ctx_var.get(),
            "batch":                self.batch_var.get(),
            "ubatch":               self.ubatch_var.get(),
            "threads":              self.threads_var.get(),
            "threads_batch":        self.threads_batch_var.get(),
            "cache_type_k":         self.cache_type_k_var.get(),
            "cache_type_v":         self.cache_type_v_var.get(),
            "temperature":          round(self.temp_var.get(), 3),
            "top_k":                self.top_k_var.get(),
            "top_p":                round(self.top_p_var.get(), 3),
            "min_p":                round(self.min_p_var.get(), 3),
            "repeat_penalty":       round(self.repeat_penalty_var.get(), 3),
            "repeat_last_n":        self.repeat_last_n_var.get(),
            "presence_penalty":     round(self.presence_penalty_var.get(), 3),
            "frequency_penalty":    round(self.frequency_penalty_var.get(), 3),
            "predict":              self.predict_var.get(),
            "seed":                 self.seed_var.get(),
            "mirostat":             self.mirostat_var.get(),
            "mirostat_lr":          round(self.mirostat_lr_var.get(), 3),
            "mirostat_ent":         round(self.mirostat_ent_var.get(), 2),
            "rope_freq_base":       self.rope_freq_base_var.get(),
            "rope_scaling":         self.rope_scaling_var.get(),
            "port":                 self.port_var.get(),
            "parallel":             self.parallel_var.get(),
            "threads_http":         self.threads_http_var.get(),
            "server_timeout":       self.server_timeout_var.get(),
            "api_key_server":       self.api_key_server_var.get(),
            "cont_batching":        self.cont_batching_var.get(),
            "embeddings":           self.embeddings_var.get(),
            "metrics":              self.metrics_var.get(),
            "props_endpoint":       self.props_endpoint_var.get(),
            "slots_endpoint":       self.slots_endpoint_var.get(),
            "flash_attn":           self.flash_attn_var.get(),
            "mlock":                self.mlock_var.get(),
            "no_mmap":              self.no_mmap_var.get(),
            "cpu_moe":              self.cpu_moe_var.get(),
            "n_cpu_moe":            self.n_cpu_moe_en_var.get(),
            "n_cpu_moe_n":          self.n_cpu_moe_var.get(),
            "no_warmup":            self.no_warmup_var.get(),
            "tokenizer_config":     self.tokenizer_config_var.get() if self.tokenizer_config_en_var.get() else "",
            "no_display_prompt":    self.no_display_prompt_var.get(),
            "jinja":                self.jinja_var.get(),
            "tensor_split":         self.tensor_split_var.get(),
            "main_gpu":             self.main_gpu_var.get(),
            "extra_flags":          self.extra_flags_var.get(),
            "alias":                self.alias_var.get(),
            "flag_prio":            self.prio_en_var.get(),
            "flag_prio_level":      self.prio_level_var.get(),
            "flag_prio_batch":      self.prio_batch_en_var.get(),
            "flag_prio_batch_level":self.prio_batch_level_var.get(),
            "flag_cache_reuse":     self.cache_reuse_en_var.get(),
            "flag_cache_reuse_n":   self.cache_reuse_n_var.get(),
            "flag_spec_mtp":        self.spec_mtp_var.get(),
            "flag_spec_draft_n":    self.spec_draft_n_en_var.get(),
            "flag_spec_draft_n_max":self.spec_draft_n_var.get(),
            "flag_prio_draft":      self.prio_draft_en_var.get(),
            "flag_prio_draft_level":self.prio_draft_level_var.get(),
        }

    def apply_profile_dict(self, p: dict) -> None:
        """Apply a profile dict to all vars."""
        if p.get("model"):
            self.model_var.set(p["model"])
        if p.get("llama_bin"):
            self.llama_bin_var.set(p["llama_bin"])
        self.ngl_var.set(p.get("ngl", 60))
        self.ctx_var.set(str(p.get("ctx", "16384")))
        self.batch_var.set(p.get("batch", 512))
        self.ubatch_var.set(p.get("ubatch", 512))
        self.threads_var.set(p.get("threads", 4))
        self.threads_batch_var.set(p.get("threads_batch", -1))
        self.cache_type_k_var.set(p.get("cache_type_k", "f16"))
        self.cache_type_v_var.set(p.get("cache_type_v", "f16"))
        self.temp_var.set(p.get("temperature", 0.8))
        self.top_k_var.set(p.get("top_k", 40))
        self.top_p_var.set(p.get("top_p", 0.95))
        self.min_p_var.set(p.get("min_p", 0.05))
        self.repeat_penalty_var.set(p.get("repeat_penalty", 1.0))
        self.repeat_last_n_var.set(p.get("repeat_last_n", 64))
        self.presence_penalty_var.set(p.get("presence_penalty", 0.0))
        self.frequency_penalty_var.set(p.get("frequency_penalty", 0.0))
        self.predict_var.set(p.get("predict", -1))
        self.seed_var.set(p.get("seed", -1))
        self.mirostat_var.set(p.get("mirostat", 0))
        self.mirostat_lr_var.set(p.get("mirostat_lr", 0.1))
        self.mirostat_ent_var.set(p.get("mirostat_ent", 5.0))
        self.rope_freq_base_var.set(p.get("rope_freq_base", 0.0))
        self.rope_scaling_var.set(p.get("rope_scaling", "auto"))
        self.port_var.set(str(p.get("port", 8089)))
        self.parallel_var.set(p.get("parallel", 1))
        self.threads_http_var.set(p.get("threads_http", -1))
        self.server_timeout_var.set(str(p.get("server_timeout", "0")))
        self.api_key_server_var.set(p.get("api_key_server", ""))
        self.cont_batching_var.set(p.get("cont_batching", True))
        self.embeddings_var.set(p.get("embeddings", False))
        self.metrics_var.set(p.get("metrics", False))
        self.props_endpoint_var.set(p.get("props_endpoint", True))
        self.slots_endpoint_var.set(p.get("slots_endpoint", True))
        self.flash_attn_var.set(p.get("flash_attn", False))
        self.mlock_var.set(p.get("mlock", False))
        self.no_mmap_var.set(p.get("no_mmap", False))
        self.cpu_moe_var.set(p.get("cpu_moe", False))
        self.n_cpu_moe_en_var.set(p.get("n_cpu_moe", False))
        self.n_cpu_moe_var.set(str(p.get("n_cpu_moe_n", "4")))
        self.no_warmup_var.set(p.get("no_warmup", False))
        tc = p.get("tokenizer_config", "")
        self.tokenizer_config_en_var.set(bool(tc.strip()))
        self.tokenizer_config_var.set(tc)
        self.no_display_prompt_var.set(p.get("no_display_prompt", False))
        self.jinja_var.set(p.get("jinja", False))
        ts = p.get("tensor_split", "")
        # Update pct_var BEFORE en_var so the trace-on-enable sees the correct pct
        self.tensor_split_en_var.set(False)
        if ts.strip():
            self.tensor_split_pct_var.set(_ts_to_pct(ts))
            self.tensor_split_en_var.set(True)
        else:
            self.tensor_split_var.set("")
        self.main_gpu_var.set(p.get("main_gpu", 0))
        self.extra_flags_var.set(p.get("extra_flags", ""))
        self.alias_var.set(p.get("alias", ""))
        self.prio_en_var.set(p.get("flag_prio", False))
        self.prio_level_var.set(str(p.get("flag_prio_level", "2")))
        self.prio_batch_en_var.set(p.get("flag_prio_batch", False))
        self.prio_batch_level_var.set(str(p.get("flag_prio_batch_level", "2")))
        self.cache_reuse_en_var.set(p.get("flag_cache_reuse", False))
        self.cache_reuse_n_var.set(str(p.get("flag_cache_reuse_n", "256")))
        self.spec_mtp_var.set(p.get("flag_spec_mtp", False))
        self.spec_draft_n_en_var.set(p.get("flag_spec_draft_n", False))
        self.spec_draft_n_var.set(str(p.get("flag_spec_draft_n_max", "2")))
        self.prio_draft_en_var.set(p.get("flag_prio_draft", False))
        self.prio_draft_level_var.set(str(p.get("flag_prio_draft_level", "2")))

    def build_cmd(self) -> str:
        label = self.llama_bin_var.get()
        bin_path = label  # fallback: use as-is if no matching fork found
        for fork in DEFAULT_FORKS:
            if fork["label"] == label:
                bin_path = self.settings.fork_bin(fork)
                break
        return build_command(self.settings, self.get_profile_dict(), bin_path)


# ── LlamaApp ──────────────────────────────────────────────────────────────────

class LlamaApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LlamaForge")
        self.root.minsize(1000, 640)

        settings = load_settings()
        self.root.geometry(settings.geometry or "1380x860")

        self.state = AppState(
            root     = self.root,
            settings = settings,
            profiles = load_profiles(),
            agents   = load_agents(),
            hardware = HardwareProfile(),
        )

        self.T = get_theme(settings.theme)
        self.root._T = self.T   # allows widgets.py _T() to find theme by walking up
        self._apply_ttk_style()
        self._theme_titlebar()
        self._build_menubar()

        if not settings.setup_done or not settings.wsl_user:
            self._run_setup_wizard()
        else:
            self._build_main_ui()
            self._start_services()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menubar(self) -> None:
        T = self.T
        menubar = tk.Menu(self.root,
                          bg=T["bg2"], fg=T["fg"],
                          activebackground=T["accent"], activeforeground=T["bg"],
                          relief="flat", bd=0)
        self.root.config(menu=menubar)

        help_menu = tk.Menu(menubar, tearoff=0,
                            bg=T["bg2"], fg=T["fg"],
                            activebackground=T["accent"], activeforeground=T["bg"])
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Quick Reference",
                              command=self._open_help)
        help_menu.add_command(label="Re-run Setup Wizard",
                              command=self._rerun_setup_wizard)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)

    def _open_help(self) -> None:
        from gui.help_window import HelpWindow
        HelpWindow(self.root, self.T)

    def _rerun_setup_wizard(self) -> None:
        self._run_setup_wizard()

    def _show_about(self) -> None:
        from tkinter import messagebox
        messagebox.showinfo(
            "About LlamaForge",
            "LlamaForge v2\n\n"
            "A GUI manager for llama.cpp running in WSL2.\n"
            "Supports model download, server control,\n"
            "inline chat, and Hermes Agent integration.\n\n"
            "github.com/Aiesus/LlamaForge",
            parent=self.root,
        )

    # ── Setup wizard ──────────────────────────────────────────────────────────

    def _run_setup_wizard(self):
        from gui.setup_wizard import SetupWizard
        wizard = SetupWizard(self.root, self.state, self.T, self._on_setup_complete)
        wizard.show()

    def _on_setup_complete(self):
        save_settings(self.state.settings)
        self._build_main_ui()
        self._start_services()

    # ── Main UI ───────────────────────────────────────────────────────────────

    def _build_main_ui(self):
        self.root.configure(bg=self.T["bg"])

        from gui.header import Header
        self.header = Header(self.root, self.state, self.T, log_fn=self._log)

        T = self.T
        self.main_frame = tk.Frame(self.root, bg=T["bg"])
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._paned = tk.PanedWindow(
            self.main_frame, orient="horizontal",
            bg=T["bg3"], sashwidth=5, sashpad=1,
            sashrelief="flat", showhandle=False,
            relief="flat", bd=0,
        )
        self._paned.pack(fill="both", expand=True)

        self._saved_log_width: int | None = None
        self._build_left_pane()
        self._build_center_tabs()
        self._build_log_panel()
        self._build_log_stub()

        # Restore saved sash positions.
        # winfo_width() can return a non-zero "requested" width before the widget
        # is actually painted.  <Configure> fires with the true rendered size, so
        # we bind once and unregister immediately after setting positions.
        s = self.state.settings
        _done = [False]
        def _restore_sashes(event=None):
            if _done[0]:
                return
            _done[0] = True
            self._paned.unbind("<Configure>")
            try:
                if s.pane_sash0 > 0:
                    self._paned.sash_place(0, s.pane_sash0, 0)
                if s.pane_sash1 > 0:
                    # Small delay so sash-0's layout change propagates first
                    self.root.after(20, lambda: self._paned.sash_place(1, s.pane_sash1, 0))
            except Exception:
                pass
        self._paned.bind("<Configure>", _restore_sashes)

    def _build_left_pane(self):
        from gui.left_panel import LeftPanel
        T = self.T
        left = tk.Frame(self._paned, bg=T["bg2"])
        self._paned.add(left, minsize=300, width=460)
        self.left_panel = LeftPanel(self.root, self.state, self.T, log_fn=self._log)
        self.left_panel.build(left)

    def _build_center_tabs(self):
        T = self.T
        center = tk.Frame(self._paned, bg=T["bg2"])
        self._paned.add(center, minsize=350)

        style = ttk.Style()
        style.configure("App.TNotebook",
                        background=T["bg2"], borderwidth=0)
        style.configure("App.TNotebook.Tab",
                        background=T["btn"], foreground=T["btn_fg"],
                        padding=[10, 4], font=("Segoe UI", 9))
        style.map("App.TNotebook.Tab",
                  background=[("selected", T["bg3"])],
                  foreground=[("selected", T["accent"])])

        self.notebook = ttk.Notebook(center, style="App.TNotebook")
        self.notebook.pack(fill="both", expand=True)

        self._build_tabs()

    def _build_tabs(self):
        from gui.tabs.server_tab    import ServerTab
        from gui.tabs.model_tab     import ModelTab
        from gui.tabs.sampling_tab  import SamplingTab
        from gui.tabs.advanced_tab  import AdvancedTab
        from gui.tabs.agents_tab    import AgentsTab
        from gui.tabs.optimizer_tab import OptimizerTab

        tabs = [
            ("Server",    ServerTab),
            ("Load",      ModelTab),
            ("Sampling",  SamplingTab),
            ("Advanced",  AdvancedTab),
            ("Agents",    AgentsTab),
            ("Optimizer", OptimizerTab),
        ]
        for label, TabClass in tabs:
            frame = tk.Frame(self.notebook, bg=self.T["bg2"])
            self.notebook.add(frame, text=label)
            TabClass(frame, self.state, self.T, log_fn=self._log).build()

    def _build_log_panel(self):
        T = self.T
        self.right = tk.Frame(self._paned, bg=T["bg2"])
        self._paned.add(self.right, minsize=180, width=400)

        # Vertical split: log on top (40%), chat on bottom (60%)
        self._right_paned = tk.PanedWindow(
            self.right, orient="vertical",
            bg=T["bg3"], sashwidth=5, sashpad=1,
            sashrelief="flat", showhandle=False,
            relief="flat", bd=0,
        )
        self._right_paned.pack(fill="both", expand=True)

        # ── Log pane ──────────────────────────────────────────────────────────
        self._log_pane = tk.Frame(self._right_paned, bg=T["bg2"])
        self._right_paned.add(self._log_pane, minsize=80)

        hrow = tk.Frame(self._log_pane, bg=T["bg2"])
        hrow.pack(fill="x")
        _section_label(hrow, "LIVE LOG", T)

        ctrl = tk.Frame(hrow, bg=T["bg2"])
        ctrl.pack(side="right", padx=8)

        tk.Button(
            ctrl, text="◀ Hide", bg=T["btn"], fg=T["btn_fg"],
            relief="flat", cursor="hand2", font=("Segoe UI", 8),
            command=self._toggle_log,
        ).pack(side="left", padx=2)

        self._pause_btn = tk.Button(
            ctrl, text="⏸ Pause", bg=T["btn"], fg=T["btn_fg"],
            relief="flat", cursor="hand2", font=("Segoe UI", 8),
            command=self._toggle_scroll_pause,
        )
        self._pause_btn.pack(side="left", padx=2)

        tk.Button(ctrl, text="Clear", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 8),
                  command=self._clear_log).pack(side="left", padx=2)

        tk.Button(ctrl, text="Copy", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 8),
                  command=self._copy_log).pack(side="left", padx=2)

        self._chat_toggle_btn = tk.Button(
            ctrl, text="▼ Chat", bg=T["btn"], fg=T["accent"],
            relief="flat", cursor="hand2", font=("Segoe UI", 8),
            command=self._toggle_chat,
        )
        # Shown only after first server-ready; packed then

        _log_wrap = tk.Frame(self._log_pane, bg=T["log_bg"])
        _log_wrap.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        _log_vsb = ttk.Scrollbar(_log_wrap, orient="vertical")
        _log_vsb.pack(side="right", fill="y")
        self.log_box = tk.Text(
            _log_wrap, bg=T["log_bg"], fg=T["log_fg"],
            font=("Consolas", 9), relief="flat", wrap="word",
            state="disabled", padx=8, pady=6,
            yscrollcommand=_log_vsb.set,
        )
        self.log_box.pack(side="left", fill="both", expand=True)
        _log_vsb.config(command=self.log_box.yview)
        self.log_box.tag_config("error",   foreground=T["red"])
        self.log_box.tag_config("success", foreground=T["green"])
        self.log_box.tag_config("warn",    foreground=T["orange"])
        self.log_box.tag_config("info",    foreground=T["accent"])
        self.log_box.tag_config("hermes",  foreground=T["yellow"])

        # ── Log buffer (for filter replay) ───────────────────────────────────
        self._log_buffer: list[tuple[str, str | None]] = []

        # ── Log filter entry ──────────────────────────────────────────────────
        filter_row = tk.Frame(self._log_pane, bg=T["bg2"])
        filter_row.pack(fill="x", padx=8, pady=(0, 2))
        tk.Label(filter_row, text="Filter:", bg=T["bg2"], fg=T["fg2"],
                 font=("Segoe UI", 8)).pack(side="left")
        self._log_filter_var = tk.StringVar()
        self._log_filter_var.trace_add("write", self._apply_log_filter)
        tk.Entry(
            filter_row, textvariable=self._log_filter_var,
            bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
            font=("Consolas", 8), insertbackground=T["fg"],
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        # ── Chat panel (built now, added to paned on first server-ready) ──────
        from gui.chat_panel import ChatPanel
        self.chat_panel = ChatPanel(
            self._right_paned, self.state, T,
            log_fn=self._log,
            hide_fn=self._toggle_chat,
        )
        self._chat_ever_shown = False

    def _build_log_stub(self):
        T = self.T
        self._log_stub = tk.Frame(self._paned, bg=T["bg2"], width=24)
        tk.Button(
            self._log_stub, text="▶",
            bg=T["bg2"], fg=T["fg2"],
            relief="flat", cursor="hand2",
            font=("Segoe UI", 9), bd=0,
            command=self._toggle_log,
        ).pack(fill="both", expand=True)

    # ── Services ──────────────────────────────────────────────────────────────

    def _start_services(self):
        s = self.state.settings

        # Hardware detection (async — doesn't block UI)
        def _hw():
            hw = detect_hardware(s.wsl_distro, s.wsl_user)
            self.state.hardware = hw

        threading.Thread(target=_hw, daemon=True).start()

        # Monitor
        if s.wsl_distro and s.wsl_user:
            self._monitor = Monitor(s.wsl_distro, s.wsl_user)
            self._monitor.register(self._on_monitor_snapshot)
            self._monitor.start()

        # Server controller
        self.state.server_ctrl = ServerController(
            distro    = s.wsl_distro,
            user      = s.wsl_user,
            log_fn    = self._log,
            status_fn = self._on_server_status,
            ready_fn  = self._on_server_ready,
            tps_fn    = self._on_tps,
        )

        # Health checker
        self._health = HealthChecker(
            port_fn    = lambda: self.state.port_var.get(),
            api_key_fn = lambda: self.state.api_key_server_var.get(),
            status_fn  = self._on_server_status,
            model_fn   = lambda: self.state.model_var.get(),
            log_fn     = self._log,
            skip_fn    = lambda: (
                self.state.server_ctrl is not None and
                self.state.server_ctrl.state == ServerState.STOPPED
            ),
        )
        self._health.start()

        # Restore last profile
        last = s.last_profile
        if last and last in self.state.profiles:
            self.state.apply_profile_dict(self.state.profiles[last])

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_tps(self, tps: float) -> None:
        if hasattr(self, "header"):
            self._safe_after(lambda t=tps: self.header.update_tps(t))

    def _on_monitor_snapshot(self, snap: MonitorSnapshot) -> None:
        if hasattr(self, "header"):
            self._safe_after(lambda s=snap: self.header.update_stats(s))

    def _on_server_status(self, state: str, model: str) -> None:
        if hasattr(self, "header"):
            self._safe_after(lambda: self.header.update_server_status(state, model))
        if hasattr(self, "left_panel"):
            self._safe_after(lambda: self.left_panel.update_server_status(state))
        if hasattr(self, "chat_panel") and state in ("stopped", "error", "crashed"):
            self._safe_after(lambda: self.chat_panel.set_connected(False))

    def _on_server_ready(self) -> None:
        from core import agents as agents_core
        # Show chat panel and mark it connected on first (and subsequent) loads
        if hasattr(self, "chat_panel"):
            def _activate_chat():
                self.chat_panel.set_connected(True)
                if str(self.chat_panel.frame) not in self._right_paned.panes():
                    self._show_chat()
            self._safe_after(_activate_chat)

        s = self.state.settings
        for agent in self.state.agents:
            if agent.get("auto_sync_model") and agent.get("enabled"):
                agents_core.sync_model(
                    agent,
                    model_name = self.state.alias_var.get() or self.state.model_var.get(),
                    base_url   = f"http://localhost:{self.state.port_var.get()}/v1"
                        if self.state.proxy_bypass_var.get()
                        else "http://localhost:8088/v1",
                    api_key    = self.state.api_key_server_var.get(),
                    log_fn     = self._log,
                )
        if s.proxy_enabled and not self.state.proxy_bypass_var.get():
            from core.wsl import start_proxy
            start_proxy(s.wsl_distro, s.wsl_user, self._log)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, text: str, tag: str | None = None) -> None:
        def _do():
            try:
                self._log_buffer.append((text, tag))
                if len(self._log_buffer) > _MAX_LOG_BUFFER:
                    del self._log_buffer[:-_MAX_LOG_BUFFER]
                q = self._log_filter_var.get().strip().lower()
                if q and q not in text.lower():
                    return
                self.log_box.config(state="normal")
                self.log_box.insert(tk.END, text + "\n", tag or "")
                if not self.state._scroll_paused:
                    self.log_box.see(tk.END)
                self.log_box.config(state="disabled")
            except Exception:
                pass
        self._safe_after(_do)

    def _clear_log(self) -> None:
        self._log_buffer.clear()
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", tk.END)
        self.log_box.config(state="disabled")

    def _copy_log(self) -> None:
        try:
            text = self.log_box.get("1.0", tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except Exception:
            pass

    def _apply_log_filter(self, *_) -> None:
        try:
            q = self._log_filter_var.get().strip().lower()
            self.log_box.config(state="normal")
            self.log_box.delete("1.0", tk.END)
            for text, tag in self._log_buffer:
                if not q or q in text.lower():
                    self.log_box.insert(tk.END, text + "\n", tag or "")
            if not self.state._scroll_paused:
                self.log_box.see(tk.END)
            self.log_box.config(state="disabled")
        except Exception:
            pass

    def _toggle_log(self):
        panes = self._paned.panes()
        if str(self.right) in panes:
            self._saved_log_width = self.right.winfo_width()
            self._paned.forget(self.right)
            self._paned.add(self._log_stub, minsize=24, width=24)
        else:
            if str(self._log_stub) in self._paned.panes():
                self._paned.forget(self._log_stub)
            w = self._saved_log_width or 400
            self._paned.add(self.right, minsize=180, width=w)

    def _toggle_chat(self) -> None:
        visible = str(self.chat_panel.frame) in self._right_paned.panes()
        if visible:
            self._right_paned.forget(self.chat_panel.frame)
            self._chat_toggle_btn.config(text="▶ Chat")
        else:
            self._show_chat()
            self._chat_toggle_btn.config(text="▼ Chat")

    def _show_chat(self) -> None:
        """Add chat to right_paned at 60 % of the panel height."""
        h = self._right_paned.winfo_height()
        chat_h = max(200, int(h * 0.6)) if h > 10 else 300
        self.chat_panel.show(height=chat_h)
        if not self._chat_ever_shown:
            self._chat_ever_shown = True
            self._chat_toggle_btn.pack(side="left", padx=2)

    def _toggle_scroll_pause(self):
        self.state._scroll_paused = not self.state._scroll_paused
        self._pause_btn.config(
            text="▶ Resume" if self.state._scroll_paused else "⏸ Pause"
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _safe_after(self, fn: Callable) -> None:
        if self.state._rebuilding:
            return
        try:
            self.root.after(0, fn)
        except Exception:
            pass

    def _apply_ttk_style(self):
        T = self.T
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TCombobox",
                        fieldbackground=T["entry_bg"],
                        background=T["btn"],
                        foreground=T["entry_fg"],
                        selectbackground=T["entry_bg"],
                        selectforeground=T["entry_fg"],
                        arrowcolor=T["fg2"])
        # readonly state must be mapped explicitly — configure() alone doesn't
        # override the OS selection highlight that hides the displayed value
        style.map("TCombobox",
                  fieldbackground=[("readonly", T["entry_bg"]),
                                   ("disabled", T["bg3"])],
                  foreground=[("readonly", T["entry_fg"]),
                               ("disabled", T["fg2"])],
                  selectbackground=[("readonly", T["entry_bg"])],
                  selectforeground=[("readonly", T["entry_fg"])],
                  background=[("readonly", T["btn"])])
        style.configure("TScrollbar",
                        background=T["bg3"],
                        troughcolor=T["bg"],
                        arrowcolor=T["fg2"],
                        bordercolor=T["bg"],
                        darkcolor=T["bg3"],
                        lightcolor=T["bg3"],
                        relief="flat")
        style.map("TScrollbar",
                  background=[("active", T["accent"]), ("pressed", T["accent"])],
                  arrowcolor=[("active", T["fg"]), ("pressed", T["fg"])])
        style.configure("Vertical.TScrollbar",   width=12)
        style.configure("Horizontal.TScrollbar", width=12)

    def _theme_titlebar(self):
        import sys, ctypes
        if sys.platform != "win32":
            return
        try:
            self.root.update()  # realize the window so frame() returns a valid HWND
            def _colorref(h: str) -> int:
                return int(h[1:3], 16) | (int(h[3:5], 16) << 8) | (int(h[5:7], 16) << 16)
            T    = self.T
            # root.frame() returns the hex HWND of the WM frame window (has the title bar)
            hwnd = int(self.root.frame(), 16)
            # Force dark caption icons/text (white minimize/restore/close icons)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4)
            # Caption background color
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 35, ctypes.byref(ctypes.c_int(_colorref(T["bg2"]))), 4)
            # Caption text/icon color
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 36, ctypes.byref(ctypes.c_int(_colorref(T["fg"]))), 4)
        except Exception:
            pass

    def _save_state(self):
        s = self.state.settings
        s.geometry     = self.root.geometry()
        s.last_model   = self.state.model_var.get()
        s.last_profile = self.state.profile_var.get()
        s.proxy_bypass = self.state.proxy_bypass_var.get()
        s.wsl_memory   = self.state.wsl_memory_var.get()
        s.cuda_swap    = self.state.cuda_swap_var.get()
        try:
            s.pane_sash0 = self._paned.sash_coord(0)[0]
            if str(self.right) in self._paned.panes():
                s.pane_sash1 = self._paned.sash_coord(1)[0]
        except Exception:
            pass
        save_settings(s)
        save_profiles(self.state.profiles)
        save_agents(self.state.agents)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_close(self):
        self._save_state()
        from tkinter import messagebox
        if (self.state.server_ctrl
                and self.state.server_ctrl.state != ServerState.STOPPED):
            if messagebox.askyesno("Running Server",
                                   "A model is loaded. Unload before closing?"):
                self.state.server_ctrl.stop(silent=True)
        self.root.destroy()

    def run(self):
        self.root.lift()
        self.root.focus_force()
        self.root.mainloop()


# ── Widget helpers (module-level, used by panels/tabs) ────────────────────────

def _section_label(parent: tk.Widget, text: str, T: dict) -> tk.Label:
    lbl = tk.Label(parent, text=text, bg=parent.cget("bg"),
                   fg=T["accent"], font=("Consolas", 8, "bold"))
    lbl.pack(anchor="w", padx=8, pady=(8, 2))
    return lbl


def separator(parent: tk.Widget, T: dict) -> None:
    tk.Frame(parent, bg=T["bg3"], height=1).pack(fill="x", padx=8, pady=4)
