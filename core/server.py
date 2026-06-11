"""
llama-server lifecycle: build command, start, stop, log streaming, health check.
No tkinter imports — all output via log_fn callbacks.
"""
from __future__ import annotations
import re
import subprocess
import threading
import time
import urllib.request
import urllib.error
from enum import Enum
from typing import Callable

_TPS_RE = re.compile(r"n_tokens_second\s*=\s*([\d.]+)", re.IGNORECASE)

from core.settings import AppSettings, DEFAULT_FORKS

LogFn    = Callable[[str, str | None], None]
StatusFn = Callable[[str, str], None]   # (state, model_name)  state: stopped|loading|running


class ServerState(Enum):
    STOPPED  = "stopped"
    LOADING  = "loading"
    RUNNING  = "running"


# ── Command builder ───────────────────────────────────────────────────────────

def build_command(s: AppSettings, p: dict, llama_bin: str) -> str:
    """
    Build the full llama-server bash command from settings + profile dict.
    Returns an empty string if no model is selected.
    """
    model = p.get("model", "")
    if not model:
        return ""

    if p.get("mlock") and s.cuda_swap:
        launch_prefix = "sudo prlimit --memlock=unlimited:unlimited env CUDA_VISIBLE_DEVICES=1,0 "
    elif p.get("mlock"):
        launch_prefix = "sudo prlimit --memlock=unlimited:unlimited "
    elif s.cuda_swap:
        launch_prefix = "CUDA_VISIBLE_DEVICES=1,0 "
    else:
        launch_prefix = ""
    alias = p.get("alias", "").strip() or (model[:-5] if model.lower().endswith(".gguf") else model)

    parts = [
        f"cd {s.llama_root} && {launch_prefix}{llama_bin}",
        f"-m {s.models_wsl}/{model}",
        f"--alias {alias}",
        f"-ngl {p.get('ngl', 60)}",
        f"-c {p.get('ctx', 16384)}",
        f"-b {p.get('batch', 512)}",
        f"-ub {p.get('ubatch', 512)}",
        f"-t {p.get('threads', 4)}",
    ]

    tb = p.get("threads_batch", -1)
    if tb > 0:
        parts.append(f"-tb {tb}")

    ck = p.get("cache_type_k", "f16")
    cv = p.get("cache_type_v", "f16")
    if ck != "f16":
        parts.append(f"--cache-type-k {ck}")
    if cv != "f16":
        parts.append(f"--cache-type-v {cv}")

    parts += [
        f"--port {p.get('port', 8089)}",
        "--host 0.0.0.0",
        f"-np {p.get('parallel', 1)}",
    ]

    th = p.get("threads_http", -1)
    if th > 0:
        parts.append(f"--threads-http {th}")

    to = str(p.get("server_timeout", "0")).strip()
    if to:
        parts.append(f"--timeout {to}")

    ak = p.get("api_key_server", "").strip()
    if ak:
        parts.append(f"--api-key {ak}")

    parts += [
        f"--temp {float(p.get('temperature', 0.8)):.3f}",
        f"--top-k {p.get('top_k', 40)}",
        f"--top-p {float(p.get('top_p', 0.95)):.3f}",
        f"--min-p {float(p.get('min_p', 0.05)):.3f}",
        f"--repeat-penalty {float(p.get('repeat_penalty', 1.0)):.3f}",
        f"--repeat-last-n {p.get('repeat_last_n', 64)}",
    ]

    pp = float(p.get("presence_penalty", 0.0))
    if pp != 0.0:
        parts.append(f"--presence-penalty {pp:.3f}")

    fp = float(p.get("frequency_penalty", 0.0))
    if fp != 0.0:
        parts.append(f"--frequency-penalty {fp:.3f}")

    pr = p.get("predict", -1)
    if pr != -1:
        parts.append(f"-n {pr}")

    sd = p.get("seed", -1)
    if sd != -1:
        parts.append(f"--seed {sd}")

    ms = p.get("mirostat", 0)
    if ms:
        parts += [
            f"--mirostat {ms}",
            f"--mirostat-lr {float(p.get('mirostat_lr', 0.1)):.3f}",
            f"--mirostat-ent {float(p.get('mirostat_ent', 5.0)):.2f}",
        ]

    rfb = str(p.get("rope_freq_base", 0.0)).strip()
    if rfb and rfb not in ("0", "0.0"):
        parts.append(f"--rope-freq-base {rfb}")

    rs = p.get("rope_scaling", "auto")
    if rs and rs != "auto":
        parts.append(f"--rope-scaling {rs}")

    ts = p.get("tensor_split", "").strip()
    if ts:
        parts.append(f"--tensor-split {ts}")
        parts.append(f"--main-gpu {p.get('main_gpu', 0)}")

    if p.get("cont_batching"):
        parts.append("--cont-batching")
    if p.get("embeddings"):
        parts.append("--embeddings")
    if p.get("metrics"):
        parts.append("--metrics")
    if p.get("props_endpoint"):
        parts.append("--props")
    if not p.get("slots_endpoint", True):
        parts.append("--no-slots")
    if p.get("flash_attn"):
        parts.append("--flash-attn on")
    if p.get("mlock"):
        parts.append("--mlock")
    if p.get("no_mmap"):
        parts.append("--no-mmap")
    if p.get("cpu_moe"):
        parts.append("--cpu-moe")
    if p.get("n_cpu_moe"):
        n = str(p.get("n_cpu_moe_n", "4")).strip()
        if n:
            parts.append(f"--n-cpu-moe {n}")
    if p.get("no_display_prompt"):
        parts.append("--no-display-prompt")
    if p.get("jinja"):
        parts.append("--jinja")
    tc = p.get("tokenizer_config", "").strip()
    if tc:
        parts.append(f"--chat-template-file {tc}")
    if p.get("no_warmup"):
        parts.append("--no-warmup")

    parts.append("-v")

    if p.get("flag_prio"):
        parts.append(f"--prio {p.get('flag_prio_level', '2')}")
    if p.get("flag_prio_batch"):
        parts.append(f"--prio-batch {p.get('flag_prio_batch_level', '2')}")
    if p.get("flag_cache_reuse"):
        n = str(p.get("flag_cache_reuse_n", "256")).strip()
        if n:
            parts.append(f"--cache-reuse {n}")
    if p.get("flag_spec_mtp"):
        parts.append("--spec-type draft-mtp")
    if p.get("flag_spec_draft_n"):
        n = str(p.get("flag_spec_draft_n_max", "2")).strip()
        if n:
            parts.append(f"--spec-draft-n-max {n}")
    if p.get("flag_prio_draft"):
        parts.append(f"--prio-draft {p.get('flag_prio_draft_level', '2')}")

    extra = p.get("extra_flags", "").strip()
    if extra:
        parts.append(extra)

    return " ".join(parts)


# ── Server controller ─────────────────────────────────────────────────────────

class ServerController:
    """
    Manages llama-server process lifecycle.
    All GUI callbacks are optional — safe to use from tests without a UI.
    """

    def __init__(self, distro: str, user: str,
                 log_fn: LogFn,
                 status_fn: StatusFn | None = None,
                 ready_fn: Callable[[], None] | None = None,
                 tps_fn: Callable[[float], None] | None = None):
        self._distro    = distro
        self._user      = user
        self._log       = log_fn
        self._status_fn = status_fn
        self._ready_fn  = ready_fn
        self._tps_fn    = tps_fn

        self.process: subprocess.Popen | None = None
        self.state   = ServerState.STOPPED
        self._log_thread: threading.Thread | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, inner_cmd: str, model_name: str) -> bool:
        """Launch llama-server. Returns False if already running or no command."""
        if not inner_cmd:
            self._log("[ERROR] No model selected.", "error")
            return False

        self.stop(silent=True)
        time.sleep(0.4)

        full_cmd = f'wsl -d {self._distro} -u {self._user} bash -c "{inner_cmd}"'
        self._log(f"\n[LOAD] Starting: {model_name}", "info")
        self._log(f"[CMD]  {inner_cmd}", "info")
        self._set_state(ServerState.LOADING, model_name)

        try:
            self.process = subprocess.Popen(
                full_cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except Exception as e:
            self._log(f"[ERROR] Launch failed: {e}", "error")
            self._set_state(ServerState.STOPPED)
            return False

        self._log_thread = threading.Thread(
            target=self._stream_log, args=(model_name,), daemon=True
        )
        self._log_thread.start()
        return True

    def stop(self, silent: bool = False) -> None:
        if not silent:
            self._log("[UNLOAD] Stopping llama-server...", "warn")
        try:
            subprocess.run(
                f"wsl -d {self._distro} -u {self._user} pkill -f llama-server",
                shell=True, capture_output=True, timeout=10
            )
        except Exception:
            pass
        if self.process:
            try:
                self.process.terminate()
            except Exception:
                pass
            self.process = None
        self._set_state(ServerState.STOPPED)
        if self._tps_fn:
            self._tps_fn(0.0)
        if not silent:
            self._log("[UNLOAD] Server stopped.", "warn")

    def is_running(self) -> bool:
        return self.state == ServerState.RUNNING

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_state(self, state: ServerState, model: str = "") -> None:
        self.state = state
        if self._status_fn:
            self._status_fn(state.value, model)

    def _stream_log(self, model_name: str) -> None:
        if not self.process:
            return
        _suppress = (
            "failed to mount", "see dmesg", "all tasks already finished",
            "stop: all tasks", "/api/v1/models", "/api/tags", "/v1/props",
            "log_server_r",
        )
        try:
            for line in self.process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if any(s in line for s in _suppress):
                    continue
                low = line.lower()
                if "error" in low or "failed" in low or "exception" in low:
                    tag = "error"
                elif "loaded" in low or "cuda" in low or "ggml_cuda" in low:
                    tag = "success"
                elif "warn" in low:
                    tag = "warn"
                elif any(k in low for k in ("llama_model_load", "llm_load", "ggml", "layer")):
                    tag = "info"
                else:
                    tag = None
                self._log(line, tag)

                # Parse tokens/sec from generation completion lines
                if self._tps_fn:
                    m = _TPS_RE.search(line)
                    if m:
                        try:
                            self._tps_fn(float(m.group(1)))
                        except Exception:
                            pass

                if any(k in low for k in (
                    "server listening", "all slots are idle",
                    "model loaded", "llama server listening",
                )):
                    self._set_state(ServerState.RUNNING, model_name)
                    if self._ready_fn:
                        threading.Thread(target=self._ready_fn, daemon=True).start()
        except Exception:
            pass


# ── Health check loop ─────────────────────────────────────────────────────────

class HealthChecker:
    """
    Polls the server's /health endpoint every `interval` seconds.
    Fires status_fn when state transitions (down→up or up→down).
    """

    def __init__(self, port_fn: Callable[[], str],
                 api_key_fn: Callable[[], str],
                 status_fn: StatusFn,
                 model_fn: Callable[[], str],
                 log_fn: LogFn,
                 interval: float = 3.0,
                 skip_fn: Callable[[], bool] | None = None):
        self._port_fn    = port_fn
        self._api_key_fn = api_key_fn
        self._status_fn  = status_fn
        self._model_fn   = model_fn
        self._log        = log_fn
        self._interval   = interval
        self._skip_fn    = skip_fn
        self._running    = False

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        was_ok = False
        while self._running:
            is_ok = False
            if not (self._skip_fn and self._skip_fn()):
                try:
                    port    = self._port_fn()
                    api_key = self._api_key_fn()
                    req = urllib.request.Request(f"http://localhost:{port}/health")
                    if api_key:
                        req.add_header("Authorization", f"Bearer {api_key}")
                    with urllib.request.urlopen(req, timeout=2) as r:
                        is_ok = (r.status == 200)
                except urllib.error.HTTPError:
                    is_ok = False
                except Exception:
                    is_ok = False

            model = self._model_fn()
            if is_ok and not was_ok:
                self._status_fn("running", model)
                self._log(f"[OK] Server healthy on port {self._port_fn()}", "success")
            elif not is_ok and was_ok:
                self._status_fn("stopped", "")

            was_ok = is_ok
            time.sleep(self._interval)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def run_diagnostics(distro: str, user: str, port: str, log_fn: LogFn) -> None:
    """Run all connectivity checks and log results."""
    log_fn("\n─── DIAGNOSTICS ─────────────────────────", "info")

    def _check():
        # 1. llama-server process in WSL
        r = subprocess.run(
            f"wsl -d {distro} pgrep -a llama-server",
            shell=True, capture_output=True, text=True
        )
        if r.stdout.strip():
            log_fn(f"[✓] llama-server process: {r.stdout.strip()}", "success")
        else:
            log_fn("[✗] llama-server: NOT running in WSL", "error")

        # 2. Port listening in WSL
        r = subprocess.run(
            f'wsl -d {distro} sh -c "ss -tlnp | grep {port}"',
            shell=True, capture_output=True, text=True
        )
        if r.stdout.strip():
            log_fn(f"[✓] WSL port {port} listening", "success")
        else:
            log_fn(f"[✗] WSL port {port}: not listening (still loading?)", "error")

        # 3. tool-proxy.py running
        r = subprocess.run(
            f"wsl -d {distro} pgrep -f tool-proxy.py",
            shell=True, capture_output=True, text=True
        )
        if r.stdout.strip():
            log_fn("[✓] tool-proxy.py running in WSL", "success")
        else:
            log_fn("[✗] tool-proxy.py: NOT running", "error")

        # 4. Proxy :8088 reachable
        try:
            with urllib.request.urlopen("http://localhost:8088/v1/models", timeout=3) as resp:
                log_fn(f"[✓] Proxy :8088 reachable (HTTP {resp.status})", "success")
        except Exception as e:
            log_fn(f"[✗] Proxy :8088 unreachable: {e}", "error")

        # 5. localhost:{port}/health
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=3
            ) as resp:
                body = resp.read().decode()
                log_fn(f"[✓] localhost:{port}/health → {body}", "success")
        except Exception as e:
            log_fn(f"[✗] localhost:{port}/health: {e}", "error")

        # 6. WSL IP direct
        try:
            wsl_ip_r = subprocess.run(
                f"wsl -d {distro} hostname -I",
                shell=True, capture_output=True, text=True, timeout=5
            )
            wsl_ip = wsl_ip_r.stdout.strip().split()[0]
            with urllib.request.urlopen(
                f"http://{wsl_ip}:{port}/health", timeout=3
            ) as resp:
                log_fn(f"[✓] {wsl_ip}:{port}/health → HTTP {resp.status}", "success")
        except Exception as e:
            log_fn(f"[✗] WSL direct unreachable: {e}", "error")

        log_fn("─────────────────────────────────────────", "info")

    threading.Thread(target=_check, daemon=True).start()
