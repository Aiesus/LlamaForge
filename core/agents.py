"""
Agent lifecycle and model-config sync.
Supports any OpenAI-compatible agent frontend.
Hermes is type "hermes" — syncs config.yaml.
"""
from __future__ import annotations
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

LogFn = Callable[[str, str | None], None]


def start(agent: dict, log_fn: LogFn) -> subprocess.Popen | None:
    """Launch an agent process. Returns the Popen handle or None on failure."""
    exe = agent.get("exe", "").strip()
    if not exe or not Path(exe).exists():
        log_fn(f"[AGENT:{agent['name']}] Executable not found: {exe}", "error")
        return None
    agent_type = agent.get("type", "")
    if agent_type == "hermes":
        return _start_hermes(agent, exe, log_fn)
    try:
        proc = subprocess.Popen(
            [exe], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(Path(exe).parent)
        )
        log_fn(f"[AGENT:{agent['name']}] Started (pid {proc.pid})", "info")
        threading.Thread(
            target=_stream_log, args=(proc, agent["name"], log_fn), daemon=True
        ).start()
        return proc
    except Exception as e:
        log_fn(f"[AGENT:{agent['name']}] Launch failed: {e}", "error")
        return None


def stop(proc: subprocess.Popen | None, name: str, log_fn: LogFn) -> None:
    if proc and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
    log_fn(f"[AGENT:{name}] Stopped.", "warn")


def is_running(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def sync_model(agent: dict, model_name: str, base_url: str,
               api_key: str, log_fn: LogFn) -> None:
    """Update agent's config file to point at the current model."""
    agent_type = agent.get("type", "")
    if agent_type == "hermes":
        # Strip .gguf suffix — Hermes expects a bare model name
        if model_name.lower().endswith(".gguf"):
            model_name = model_name[:-5]
        threading.Thread(
            target=_sync_hermes, args=(agent, model_name, base_url, api_key, log_fn),
            daemon=True
        ).start()
    # Future types: add elif blocks here


# ── Hermes launch ─────────────────────────────────────────────────────────────

def _find_hermes_cli(electron_exe: str) -> str | None:
    """Walk up from the Electron app path to find venv/Scripts/hermes.exe."""
    p = Path(electron_exe).parent
    for _ in range(8):
        cli = p / "venv" / "Scripts" / "hermes.exe"
        if cli.exists():
            return str(cli)
        if p.parent == p:
            break
        p = p.parent
    return None


def _start_hermes(agent: dict, exe: str, log_fn: LogFn) -> subprocess.Popen | None:
    name = agent["name"]
    # Detect whether user pointed at the Electron UI or the CLI backend.
    is_electron = "win-unpacked" in exe.lower() or "release" in exe.lower()
    cli_path = None
    if is_electron:
        cli_path = _find_hermes_cli(exe)
        if cli_path:
            log_fn(f"[AGENT:{name}] Electron app detected — launching CLI backend: {cli_path}", "hermes")
            launch_exe = cli_path
        else:
            log_fn(f"[AGENT:{name}] Warning: CLI backend not found — launching Electron directly", "warn")
            launch_exe = exe
    else:
        launch_exe = exe

    log_fn(f"[AGENT:{name}] Starting: {launch_exe}", "hermes")
    try:
        proc = subprocess.Popen(
            [launch_exe],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(Path(launch_exe).parent),
        )
        log_fn(f"[AGENT:{name}] Started (pid {proc.pid})", "hermes")
        threading.Thread(
            target=_stream_log, args=(proc, name, log_fn), daemon=True
        ).start()
        # If CLI backend started, open the Electron UI after a short delay
        if is_electron and cli_path:
            def _open_ui():
                time.sleep(2)
                log_fn(f"[AGENT:{name}] Opening Electron UI: {exe}", "hermes")
                try:
                    subprocess.Popen([exe], cwd=str(Path(exe).parent))
                except Exception as e:
                    log_fn(f"[AGENT:{name}] Failed to open UI: {e}", "warn")
            threading.Thread(target=_open_ui, daemon=True).start()
        return proc
    except Exception as e:
        log_fn(f"[AGENT:{name}] Launch failed: {e}", "error")
        return None


# ── Hermes sync ───────────────────────────────────────────────────────────────

def _hermes_config_path(agent: dict) -> str:
    cfg = agent.get("config", "").strip()
    if not cfg:
        cfg = str(Path(os.environ.get("LOCALAPPDATA", "")) / "hermes" / "config.yaml")
    return cfg


def _create_hermes_config(cfg_path: str, model_name: str, base_url: str,
                          api_key: str, log_fn: LogFn, name: str) -> bool:
    """Create a minimal Hermes config.yaml pointing at our local server."""
    try:
        Path(cfg_path).parent.mkdir(parents=True, exist_ok=True)
        key_line = f"  api_key: {api_key}\n" if api_key else "  api_key: ''\n"
        content = (
            "model:\n"
            f"{key_line}"
            "  api_mode: chat_completions\n"
            f"  base_url: {base_url}\n"
            f"  default: {model_name}\n"
            f"  name: {model_name}\n"
            "  provider: custom\n"
            "providers: {}\n"
            "fallback_providers: []\n"
        )
        Path(cfg_path).write_text(content, encoding="utf-8")
        log_fn(f"[AGENT:{name}] Created config: {cfg_path}", "success")
        return True
    except Exception as e:
        log_fn(f"[AGENT:{name}] Failed to create config: {e}", "warn")
        return False


def _sync_hermes(agent: dict, model_name: str, base_url: str,
                 api_key: str, log_fn: LogFn) -> None:
    """
    Edit Hermes config.yaml to point at the current model.
    Updates: model.name, model.default, model.base_url, model.api_key, model.provider
    Falls back to %LOCALAPPDATA%\\hermes\\config.yaml if config field is empty.
    Creates the file if it doesn't exist.
    """
    name     = agent["name"]
    cfg_path = _hermes_config_path(agent)

    if not Path(cfg_path).exists():
        if not _create_hermes_config(cfg_path, model_name, base_url, api_key, log_fn, name):
            return
        return  # file is already fully written by _create_hermes_config

    try:
        with open(cfg_path, encoding="utf-8") as f:
            lines = f.readlines()

        in_model    = False
        found_keys  = set()
        new_lines   = []
        for line in lines:
            stripped = line.lstrip()
            indent   = len(line) - len(stripped)

            if line.rstrip() == "model:" or line.startswith("model:"):
                in_model = True
            elif in_model and indent == 0 and stripped and not stripped.startswith("#"):
                # Leaving the model section — inject any keys we never encountered
                if "provider" not in found_keys:
                    new_lines.append("  provider: custom\n")
                    found_keys.add("provider")
                in_model = False

            if in_model:
                if stripped.startswith("name:"):
                    line = " " * indent + f"name: {model_name}\n"
                    found_keys.add("name")
                elif stripped.startswith("default:"):
                    line = " " * indent + f"default: {model_name}\n"
                    found_keys.add("default")
                elif stripped.startswith("base_url:"):
                    line = " " * indent + f"base_url: {base_url}\n"
                    found_keys.add("base_url")
                elif stripped.startswith("api_key:"):
                    line = " " * indent + f"api_key: {api_key}\n"
                    found_keys.add("api_key")
                elif stripped.startswith("provider:"):
                    line = " " * indent + "provider: custom\n"
                    found_keys.add("provider")

            new_lines.append(line)

        with open(cfg_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        log_fn(
            f"[AGENT:{name}] config synced → model={model_name} url={base_url}",
            "success"
        )
    except Exception as e:
        log_fn(f"[AGENT:{name}] Config sync failed: {e}", "warn")


# ── Log streaming ─────────────────────────────────────────────────────────────

def _stream_log(proc: subprocess.Popen, name: str, log_fn: LogFn) -> None:
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log_fn(f"[AGENT:{name}] {line}", "hermes")
    except Exception:
        pass
