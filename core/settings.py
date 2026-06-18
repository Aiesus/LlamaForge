"""
Persistent settings and profiles.
All paths/usernames come from here — nothing hardcoded elsewhere.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────

APP_DIR      = Path(__file__).parent.parent
SETTINGS_FILE = APP_DIR / "settings.json"
PROFILES_FILE = APP_DIR / "profiles.json"
AGENTS_FILE   = APP_DIR / "agents.json"
BENCH_FILE    = APP_DIR / "bench_results.json"
CRASH_LOG     = APP_DIR / "crash.log"

# ── Default profile ────────────────────────────────────────────────────────────

DEFAULT_PROFILE: dict[str, Any] = {
    "model":               "",
    "ngl":                 60,
    "ctx":                 "16384",
    "batch":               512,
    "ubatch":              512,
    "threads":             4,
    "threads_batch":       -1,
    "cache_type_k":        "f16",
    "cache_type_v":        "f16",
    "temperature":         0.8,
    "top_k":               40,
    "top_p":               0.95,
    "min_p":               0.05,
    "repeat_penalty":      1.0,
    "repeat_last_n":       64,
    "presence_penalty":    0.0,
    "frequency_penalty":   0.0,
    "predict":             -1,
    "seed":                -1,
    "mirostat":            0,
    "mirostat_lr":         0.1,
    "mirostat_ent":        5.0,
    "rope_freq_base":      0.0,
    "rope_scaling":        "auto",
    "port":                8089,
    "parallel":            1,
    "threads_http":        -1,
    "api_key_server":      "",
    "cont_batching":       True,
    "embeddings":          False,
    "metrics":             True,
    "props_endpoint":      True,
    "slots_endpoint":      True,
    "flash_attn":          False,
    "mlock":               False,
    "no_mmap":             False,
    "no_display_prompt":   False,
    "jinja":               False,
    "reasoning_off":       False,
    "disable_cuda_graphs": False,
    "tensor_split":        "",
    "main_gpu":            0,
    "extra_flags":         "",
    "alias":               "",
    "server_timeout":      "0",
    "cpu_moe":             False,
    "n_cpu_moe":           False,
    "n_cpu_moe_n":         "4",
    "no_warmup":           False,
    "tokenizer_config":    "",
    "flag_prio":           False,
    "flag_prio_level":     "2",
    "flag_prio_batch":     False,
    "flag_prio_batch_level": "2",
    "flag_cache_reuse":    False,
    "flag_cache_reuse_n":  "256",
    "override_expert_count":   False,
    "override_expert_count_n": "2",
    "flag_spec_mtp":       False,
    "flag_spec_draft_n":   False,
    "flag_spec_draft_n_max": "2",
    "flag_prio_draft":     False,
    "flag_prio_draft_level": "2",
    "verbose_log":         False,
}

BUILTIN_PROFILES: dict[str, dict] = {
    "Coding":       {**DEFAULT_PROFILE, "ngl": 60, "ctx": "8192",  "temperature": 0.2, "flash_attn": True},
    "Creative":     {**DEFAULT_PROFILE, "ngl": 60, "ctx": "8192",  "temperature": 1.1},
    "Long Context": {**DEFAULT_PROFILE, "ngl": 40, "ctx": "32768", "temperature": 0.7, "flash_attn": True},
    "Dual GPU":     {**DEFAULT_PROFILE, "ngl": 99, "ctx": "40960", "temperature": 0.8,
                     "tensor_split": "8,12", "main_gpu": 0, "flash_attn": True},
}

# ── Fork definitions ───────────────────────────────────────────────────────────

DEFAULT_FORKS = [
    {
        "label":       "Official",
        "bin_rel":     "build/bin/llama-server",
        "bench_rel":   "build/bin/llama-bench",
        "root_key":    "llama_root",
        "description": "Upstream ggml-org/llama.cpp",
        "turbo_kv":    False,
    },
    {
        "label":       "TurboQuant",
        "bin_rel":     "build/bin/llama-server",
        "bench_rel":   "build/bin/llama-bench",
        "root_key":    "turbo_root",
        "description": "TheTom fork — turbo2/3/4 KV cache (Walsh-Hadamard). Not in upstream.",
        "turbo_kv":    True,
    },
]

# ── AppSettings dataclass ──────────────────────────────────────────────────────

@dataclass
class AppSettings:
    # WSL
    wsl_distro:     str = ""
    wsl_user:       str = ""
    llama_root:     str = "~/llama.cpp"
    turbo_root:     str = "~/llama-turbo"
    models_subdir:  str = "models"

    # Model libraries — list of WSL paths scanned for .gguf files.
    # First entry is the default download destination.
    model_libraries: list = field(default_factory=list)

    # Active binary (path inside WSL, absolute or ~-relative)
    llama_bin:      str = ""

    # Proxy
    proxy_enabled:  bool = True
    proxy_bypass:   bool = False    # Advanced/debug: skip proxy, connect direct to :8089

    # UI
    theme:          str  = "catppuccin-mocha"
    geometry:       str  = "1380x860"

    # Last-used
    last_model:     str  = ""
    last_profile:   str  = ""

    # WSL memory cap (written to ~/.wslconfig); empty = not yet configured
    wsl_memory:     str  = ""

    # Swap CUDA device order: sets CUDA_VISIBLE_DEVICES=1,0 before llama-server.
    # Useful when GPU 1 is larger and should be the primary compute device.
    cuda_swap:      bool = False

    # Saved PanedWindow sash positions (pixels from left edge)
    pane_sash0:     int  = 460   # between left panel and center tabs
    pane_sash1:     int  = 940   # between center tabs and log

    # Setup completed flag
    setup_done:     bool = False

    # ── Derived properties ─────────────────────────────────────────────────────

    def wsl_path_to_unc(self, wsl_path: str) -> str:
        """Convert any WSL path to a Windows-accessible path."""
        if not self.wsl_distro or not self.wsl_user:
            return ""
        path = wsl_path.replace("~", f"/home/{self.wsl_user}")
        # /mnt/x/ → Windows drive (X:\...)
        mnt = re.match(r"^/mnt/([a-zA-Z])(?:/|$)(.*)", path)
        if mnt:
            drive = mnt.group(1).upper()
            rest  = mnt.group(2).replace("/", "\\")
            return f"{drive}:\\{rest}" if rest else f"{drive}:\\"
        # Native WSL filesystem
        win_path = path.lstrip("/").replace("/", "\\")
        return rf"\\wsl.localhost\{self.wsl_distro}\{win_path}"

    @property
    def all_library_uncs(self) -> list:
        """Return [(wsl_path, unc_path), ...] for every configured library."""
        return [(p, self.wsl_path_to_unc(p)) for p in self.model_libraries
                if self.wsl_path_to_unc(p)]

    @property
    def models_unc(self) -> str:
        """UNC path to the primary (first) models folder."""
        if self.model_libraries:
            return self.wsl_path_to_unc(self.model_libraries[0])
        if not self.wsl_distro or not self.wsl_user:
            return ""
        root = self.llama_root.replace("~", f"/home/{self.wsl_user}")
        win_root = root.replace("/", "\\").lstrip("\\")
        return rf"\\wsl.localhost\{self.wsl_distro}\{win_root}\{self.models_subdir}"

    @property
    def models_wsl(self) -> str:
        """WSL path to the primary (first) models folder."""
        if self.model_libraries:
            return self.model_libraries[0]
        return f"{self.llama_root}/{self.models_subdir}"

    def fork_bin(self, fork: dict) -> str:
        """Return the WSL binary path for a given fork dict."""
        root = self.turbo_root if fork["root_key"] == "turbo_root" else self.llama_root
        return f"{root}/{fork['bin_rel']}"

    def fork_bench(self, fork: dict) -> str:
        """Return the WSL llama-bench path for a given fork dict."""
        root = self.turbo_root if fork["root_key"] == "turbo_root" else self.llama_root
        return f"{root}/{fork['bench_rel']}"

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AppSettings":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Load / Save ────────────────────────────────────────────────────────────────

def load_settings() -> AppSettings:
    try:
        if SETTINGS_FILE.exists():
            s = AppSettings.from_dict(
                json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            )
            # Migration: populate model_libraries from legacy llama_root + models_subdir
            if not s.model_libraries and s.llama_root:
                s.model_libraries = [f"{s.llama_root}/{s.models_subdir}"]
            return s
    except Exception:
        pass
    return AppSettings()


def save_settings(s: AppSettings) -> None:
    try:
        SETTINGS_FILE.write_text(
            json.dumps(s.to_dict(), indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def load_profiles() -> dict[str, dict]:
    profiles = dict(BUILTIN_PROFILES)
    try:
        if PROFILES_FILE.exists():
            profiles.update(
                json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
            )
    except Exception:
        pass
    return profiles


def save_profiles(profiles: dict[str, dict]) -> None:
    try:
        user = {k: v for k, v in profiles.items() if k not in BUILTIN_PROFILES}
        PROFILES_FILE.write_text(json.dumps(user, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_agents() -> list[dict]:
    try:
        if AGENTS_FILE.exists():
            return json.loads(AGENTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return [
        {
            "name":            "Hermes",
            "type":            "hermes",
            "exe":             "",
            "config":          "",
            "url":             "http://localhost:8088",
            "auto_sync_model": True,
            "enabled":         True,
        }
    ]


def save_agents(agents: list[dict]) -> None:
    try:
        AGENTS_FILE.write_text(json.dumps(agents, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_bench_results() -> list[dict]:
    try:
        if BENCH_FILE.exists():
            return json.loads(BENCH_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def save_bench_result(result: dict) -> None:
    """Append a single bench run result to the history file."""
    try:
        results = load_bench_results()
        results.append(result)
        BENCH_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    except Exception:
        pass
