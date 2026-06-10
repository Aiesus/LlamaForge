"""
Agent lifecycle and model-config sync.
Supports any OpenAI-compatible agent frontend.
Hermes is type "hermes" — syncs config.yaml.
"""
from __future__ import annotations
import subprocess
import threading
from pathlib import Path
from typing import Callable

LogFn = Callable[[str, str | None], None]


def start(agent: dict, log_fn: LogFn) -> subprocess.Popen | None:
    """Launch an agent process. Returns the Popen handle or None on failure."""
    exe = agent.get("exe", "").strip()
    if not exe or not Path(exe).exists():
        log_fn(f"[AGENT:{agent['name']}] Executable not found: {exe}", "error")
        return None
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
        threading.Thread(
            target=_sync_hermes, args=(agent, model_name, base_url, api_key, log_fn),
            daemon=True
        ).start()
    # Future types: add elif blocks here


# ── Hermes sync ───────────────────────────────────────────────────────────────

def _sync_hermes(agent: dict, model_name: str, base_url: str,
                 api_key: str, log_fn: LogFn) -> None:
    """
    Edit Hermes config.yaml to point at the current model.
    Updates: model.name, model.default, model.base_url, model.api_key
    """
    cfg_path = agent.get("config", "").strip()
    if not cfg_path or not Path(cfg_path).exists():
        return
    try:
        with open(cfg_path, encoding="utf-8") as f:
            lines = f.readlines()

        in_model = False
        new_lines = []
        for line in lines:
            stripped = line.lstrip()
            indent   = len(line) - len(stripped)

            if line.rstrip() == "model:" or line.startswith("model:"):
                in_model = True
            elif in_model and indent == 0 and stripped and not stripped.startswith("#"):
                in_model = False

            if in_model:
                if stripped.startswith("name:"):
                    line = " " * indent + f"name: {model_name}\n"
                elif stripped.startswith("default:"):
                    line = " " * indent + f"default: {model_name}\n"
                elif stripped.startswith("base_url:"):
                    line = " " * indent + f"base_url: {base_url}\n"
                elif stripped.startswith("api_key:") and api_key:
                    line = " " * indent + f"api_key: {api_key}\n"

            new_lines.append(line)

        with open(cfg_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        log_fn(
            f"[AGENT:{agent['name']}] config synced → model={model_name} url={base_url}",
            "success"
        )
    except Exception as e:
        log_fn(f"[AGENT:{agent['name']}] Config sync failed: {e}", "warn")


# ── Log streaming ─────────────────────────────────────────────────────────────

def _stream_log(proc: subprocess.Popen, name: str, log_fn: LogFn) -> None:
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log_fn(f"[AGENT:{name}] {line}", "hermes")
    except Exception:
        pass
