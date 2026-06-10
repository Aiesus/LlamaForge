"""
LlamaApp — root window, AppState, tab notebook.
AppState holds all tk.Var instances and live process state.
All other GUI modules receive a reference to AppState.
"""
from __future__ import annotations
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
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
        self.tensor_split_var       = tk.StringVar(value=DEFAULT_PROFILE["tensor_split"])
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
        self.no_display_prompt_var.set(p.get("no_display_prompt", False))
        self.jinja_var.set(p.get("jinja", False))
        self.tensor_split_var.set(p.get("tensor_split", ""))
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
        self.root.title("llama-gui")
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

        if not settings.setup_done or not settings.wsl_user:
            self._run_setup_wizard()
        else:
            self._build_main_ui()
            self._start_services()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

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

        from gui.header     import Header
        from gui.left_panel import LeftPanel

        self.header     = Header(self.root, self.state, self.T, log_fn=self._log)
        self.left_panel = LeftPanel(self.root, self.state, self.T, log_fn=self._log)

        self.main_frame = tk.Frame(self.root, bg=self.T["bg"])
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.columnconfigure(2, weight=2)
        self.main_frame.rowconfigure(0, weight=1)

        self.left_panel.build(self.main_frame)
        self._build_center_tabs()
        self._build_log_panel()

    def _build_center_tabs(self):
        T = self.T
        center = tk.Frame(self.main_frame, bg=T["bg2"])
        center.grid(row=0, column=1, sticky="nsew", padx=5)

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
            ("Model",     ModelTab),
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
        self.right = tk.Frame(self.main_frame, bg=T["bg2"])
        self.right.grid(row=0, column=2, sticky="nsew", padx=(5, 0))

        hrow = tk.Frame(self.right, bg=T["bg2"])
        hrow.pack(fill="x")
        _section_label(hrow, "LIVE LOG", T)

        ctrl = tk.Frame(hrow, bg=T["bg2"])
        ctrl.pack(side="right", padx=8)

        self._pause_btn = tk.Button(
            ctrl, text="⏸ Pause", bg=T["btn"], fg=T["btn_fg"],
            relief="flat", cursor="hand2", font=("Segoe UI", 8),
            command=self._toggle_scroll_pause
        )
        self._pause_btn.pack(side="left", padx=2)

        tk.Button(ctrl, text="Clear", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 8),
                  command=self._clear_log).pack(side="left", padx=2)

        self._log_toggle_btn = tk.Button(
            ctrl, text="Hide", bg=T["btn"], fg=T["btn_fg"],
            relief="flat", cursor="hand2", font=("Segoe UI", 8),
            command=self._toggle_log
        )
        self._log_toggle_btn.pack(side="left", padx=2)

        self.log_box = scrolledtext.ScrolledText(
            self.right, bg=T["log_bg"], fg=T["log_fg"],
            font=("Consolas", 9), relief="flat", wrap="word",
            state="disabled", padx=8, pady=6
        )
        self.log_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_box.tag_config("error",   foreground=T["red"])
        self.log_box.tag_config("success", foreground=T["green"])
        self.log_box.tag_config("warn",    foreground=T["orange"])
        self.log_box.tag_config("info",    foreground=T["accent"])
        self.log_box.tag_config("hermes",  foreground=T["yellow"])

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

    def _on_server_ready(self) -> None:
        from core import agents as agents_core
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
                self.log_box.config(state="normal")
                self.log_box.insert(tk.END, text + "\n", tag or "")
                if not self.state._scroll_paused:
                    self.log_box.see(tk.END)
                self.log_box.config(state="disabled")
            except Exception:
                pass
        self._safe_after(_do)

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", tk.END)
        self.log_box.config(state="disabled")

    def _toggle_log(self):
        if self.log_box.winfo_ismapped():
            self.log_box.pack_forget()
            self._log_toggle_btn.config(text="Show")
        else:
            self.log_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self._log_toggle_btn.config(text="Hide")

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
                        troughcolor=T["bg2"],
                        arrowcolor=T["fg2"])

    def _save_state(self):
        s = self.state.settings
        s.geometry     = self.root.geometry()
        s.last_model   = self.state.model_var.get()
        s.last_profile = self.state.profile_var.get()
        s.proxy_bypass = self.state.proxy_bypass_var.get()
        s.wsl_memory   = self.state.wsl_memory_var.get()
        s.cuda_swap    = self.state.cuda_swap_var.get()
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
