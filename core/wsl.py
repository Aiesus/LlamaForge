"""
All WSL shell interactions.
No tkinter imports — communicates back via log_fn callback.
"""
from __future__ import annotations
import subprocess
import threading
from typing import Callable

LogFn = Callable[[str, str | None], None]

# ── Low-level runner ──────────────────────────────────────────────────────────

def run(distro: str, user: str, cmd: str,
        timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a bash command in WSL, return CompletedProcess."""
    return subprocess.run(
        ["wsl", "-d", distro, "-u", user, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout
    )


def run_root(distro: str, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a bash command in WSL as root."""
    return subprocess.run(
        ["wsl", "-d", distro, "-u", "root", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout
    )


def stream(distro: str, user: str, cmd: str,
           log_fn: LogFn, timeout: int = 600) -> int:
    """
    Run a bash command in WSL, streaming each output line to log_fn.
    Returns the process exit code.
    """
    proc = subprocess.Popen(
        ["wsl", "-d", distro, "-u", user, "bash", "-c", cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    try:
        for line in proc.stdout:
            log_fn(line.rstrip(), None)
    except Exception:
        pass
    proc.wait(timeout=timeout)
    return proc.returncode


def stream_async(distro: str, user: str, cmd: str,
                 log_fn: LogFn, done_fn: Callable[[int], None] | None = None,
                 timeout: int = 600) -> None:
    """Run stream() on a daemon thread. Calls done_fn(returncode) when finished."""
    def _run():
        rc = stream(distro, user, cmd, log_fn, timeout)
        if done_fn:
            done_fn(rc)
    threading.Thread(target=_run, daemon=True).start()


# ── Discovery ─────────────────────────────────────────────────────────────────

def list_distros() -> list[str]:
    """Return names of installed WSL distros."""
    try:
        result = subprocess.run(
            ["wsl", "--list", "--quiet"],
            capture_output=True, timeout=10
        )
        # wsl --list outputs UTF-16-LE on Windows
        raw = result.stdout
        try:
            text = raw.decode("utf-16-le")
        except Exception:
            text = raw.decode("utf-8", errors="replace")
        return [ln.strip().strip("\x00") for ln in text.splitlines()
                if ln.strip().strip("\x00")]
    except Exception:
        return []


def detect_user(distro: str) -> str:
    """Return the default non-root user in a distro."""
    try:
        result = subprocess.run(
            ["wsl", "-d", distro, "whoami"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return ""


def detect_cpu_cores(distro: str, user: str) -> int:
    """Return number of CPU cores visible to WSL."""
    try:
        r = run(distro, user, "nproc", timeout=5)
        return int(r.stdout.strip())
    except Exception:
        return 4


def get_wsl_ip(distro: str) -> str:
    """Return the WSL VM's IP address."""
    try:
        result = subprocess.run(
            f"wsl -d {distro} hostname -I",
            shell=True, capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip().split()[0]
    except Exception:
        return ""


# ── Dependency checks ─────────────────────────────────────────────────────────

DEPS: list[dict] = [
    {"name": "git",             "check": "git --version",                    "install": "apt-get install -y git"},
    {"name": "cmake",           "check": "cmake --version",                  "install": "apt-get install -y cmake"},
    {"name": "build-essential", "check": "gcc --version",                    "install": "apt-get install -y build-essential"},
    {"name": "CUDA toolkit",    "check": "nvcc --version",                   "install": None},   # manual — link to NVIDIA
    {"name": "python3",         "check": "python3 --version",               "install": "apt-get install -y python3"},
    {"name": "pip",             "check": "pip3 --version",                   "install": "apt-get install -y python3-pip"},
    {"name": "aiohttp",         "check": "python3 -c 'import aiohttp'",      "install": "pip3 install aiohttp"},
    {"name": "wget",            "check": "wget --version",                   "install": "apt-get install -y wget"},
]


def check_dep(distro: str, user: str, dep: dict) -> bool:
    """Return True if the dependency check command succeeds."""
    try:
        r = run(distro, user, dep["check"], timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def install_dep(distro: str, dep: dict,
                log_fn: LogFn, done_fn: Callable[[int], None] | None = None) -> None:
    """Run the dependency install command as root, streaming output."""
    if not dep.get("install"):
        log_fn(f"[SETUP] {dep['name']}: manual install required — see NVIDIA docs", "warn")
        return
    cmd = f"DEBIAN_FRONTEND=noninteractive {dep['install']}"
    log_fn(f"[SETUP] Installing {dep['name']}...", "info")
    stream_async(distro, "root", cmd, log_fn, done_fn)


# ── Binary checks ─────────────────────────────────────────────────────────────

def binary_exists(distro: str, user: str, wsl_path: str) -> bool:
    """Check if a file exists at the given WSL path."""
    try:
        r = run(distro, user, f"test -f {wsl_path} && echo yes", timeout=5)
        return "yes" in r.stdout
    except Exception:
        return False


def binary_version(distro: str, user: str, wsl_path: str) -> str:
    """Return version string from a llama-server binary, or empty string."""
    try:
        r = run(distro, user, f"{wsl_path} --version 2>&1 | head -1", timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""


# ── llama.cpp build ───────────────────────────────────────────────────────────

def clone_and_build(distro: str, user: str, repo_url: str,
                    dest: str, log_fn: LogFn,
                    done_fn: Callable[[int], None] | None = None) -> None:
    """Clone a llama.cpp repo and build it with CUDA support."""
    cmd = (
        f"git clone {repo_url} {dest} 2>&1 && "
        f"cd {dest} && "
        f"cmake -B build -DGGML_CUDA=ON 2>&1 && "
        f"cmake --build build --config Release -j$(nproc) 2>&1"
    )
    log_fn(f"[BUILD] Cloning {repo_url} → {dest} ...", "info")
    stream_async(distro, user, cmd, log_fn, done_fn, timeout=1200)


def update_build(distro: str, user: str, repo_dir: str,
                 label: str, log_fn: LogFn,
                 done_fn: Callable[[int], None] | None = None) -> None:
    """git pull + rebuild if there are new commits."""
    def _run():
        log_fn(f"[UPDATE] Checking {label}...", "info")
        try:
            r = run(distro, user, f"cd {repo_dir} && git pull 2>&1", timeout=120)
            for line in (r.stdout + r.stderr).strip().splitlines():
                log_fn(f"[UPDATE] {line}", "info")
            if "already up to date" in (r.stdout + r.stderr).lower():
                log_fn(f"[UPDATE] {label}: already up to date.", "success")
                if done_fn:
                    done_fn(0)
                return
            log_fn(f"[UPDATE] {label}: new commits — rebuilding...", "warn")
            rc = stream(distro, user,
                        f"cd {repo_dir} && cmake --build build --config Release "
                        f"-j$(nproc) 2>&1 | tail -20",
                        log_fn, timeout=600)
            if rc == 0:
                log_fn(f"[UPDATE] {label}: update complete.", "success")
            else:
                log_fn(f"[UPDATE] {label}: build FAILED.", "error")
            if done_fn:
                done_fn(rc)
        except Exception as e:
            log_fn(f"[UPDATE] {label}: error: {e}", "error")
            if done_fn:
                done_fn(1)
    threading.Thread(target=_run, daemon=True).start()


# ── Proxy management ──────────────────────────────────────────────────────────

def proxy_running(distro: str, user: str) -> bool:
    """Return True if tool-proxy.py is running in WSL."""
    try:
        r = run(distro, user, "pgrep -f tool-proxy.py", timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def start_proxy(distro: str, user: str, log_fn: LogFn) -> None:
    """Start tool-proxy.py in WSL if not already running."""
    if proxy_running(distro, user):
        log_fn("[PROXY] Tool proxy already running on :8088.", "info")
        return
    cmd = "nohup python3 ~/tool-proxy.py > /tmp/tool-proxy.log 2>&1 &"
    try:
        subprocess.Popen(
            ["wsl", "-d", distro, "-u", user, "bash", "-c", cmd]
        )
        log_fn("[PROXY] Starting tool-call proxy on :8088...", "info")
    except Exception as e:
        log_fn(f"[PROXY] Failed to start: {e}", "warn")


def restart_proxy(distro: str, user: str, log_fn: LogFn,
                  done_fn: Callable[[int], None] | None = None) -> None:
    """Kill and restart tool-proxy.py, then verify :8088 responds."""
    import urllib.request, time

    def _run():
        log_fn("[PROXY] Restarting tool proxy...", "info")
        try:
            run(distro, user, "pkill -f tool-proxy.py; sleep 1", timeout=10)
            start_proxy(distro, user, log_fn)
            time.sleep(2)
            try:
                urllib.request.urlopen("http://localhost:8088/v1/models", timeout=3)
                log_fn("[PROXY] Tool proxy restarted — :8088 is live", "success")
                if done_fn:
                    done_fn(0)
            except Exception as e:
                log_fn(f"[PROXY] Proxy not responding after restart: {e}", "error")
                if done_fn:
                    done_fn(1)
        except Exception as e:
            log_fn(f"[PROXY] Restart error: {e}", "error")
            if done_fn:
                done_fn(1)
    threading.Thread(target=_run, daemon=True).start()


def deploy_proxy(distro: str, user: str, proxy_src: str, log_fn: LogFn) -> bool:
    """
    Copy the bundled tool_proxy.py to ~/tool-proxy.py in WSL.
    proxy_src: Windows path to the bundled tool_proxy.py.
    Returns True on success.
    """
    import shutil
    from pathlib import Path
    try:
        # Derive the UNC destination path
        home_unc = rf"\\wsl.localhost\{distro}\home\{user}"
        dest = Path(home_unc) / "tool-proxy.py"
        shutil.copy2(proxy_src, dest)
        log_fn("[SETUP] tool-proxy.py deployed to WSL ~/tool-proxy.py", "success")
        return True
    except Exception as e:
        log_fn(f"[SETUP] Failed to deploy proxy: {e}", "error")
        return False


# ── mlock fix ─────────────────────────────────────────────────────────────────

def fix_mlock(distro: str, log_fn: LogFn) -> None:
    """Write unlimited memlock limits and sudoers rule to WSL."""
    script = (
        "sed -i '/memlock/d' /etc/security/limits.conf && "
        "printf '* soft memlock unlimited\\n* hard memlock unlimited\\n'"
        " >> /etc/security/limits.conf && "
        "mkdir -p /etc/sudoers.d && "
        "echo 'ALL ALL=(root) NOPASSWD: /usr/bin/prlimit' > /etc/sudoers.d/llama-mlock && "
        "chmod 440 /etc/sudoers.d/llama-mlock && "
        "echo DONE"
    )

    def _run():
        result = run_root(distro, script, timeout=15)
        if "DONE" in result.stdout or result.returncode == 0:
            log_fn("[MLOCK] Done. mlock ready immediately; permanent after WSL restart.", "success")
        else:
            log_fn(f"[MLOCK] Failed: {result.stderr or result.stdout}", "error")
    threading.Thread(target=_run, daemon=True).start()


# ── WSL memory (.wslconfig) ───────────────────────────────────────────────────

from pathlib import Path as _Path

WSLCONFIG = _Path.home() / ".wslconfig"


def read_wsl_memory() -> str:
    """Return the memory= value from ~/.wslconfig, or 'not set'."""
    try:
        if WSLCONFIG.exists():
            for line in WSLCONFIG.read_text().splitlines():
                if line.strip().lower().startswith("memory"):
                    return line.split("=", 1)[-1].strip()
    except Exception:
        pass
    return "not set"


def write_wsl_memory(mem: str) -> None:
    """Write memory={mem} to ~/.wslconfig [wsl2] section."""
    lines = []
    if WSLCONFIG.exists():
        lines = WSLCONFIG.read_text(encoding="utf-8").splitlines()

    in_wsl2 = False
    mem_written = False
    new_lines = []

    for line in lines:
        stripped = line.strip().lower()
        if stripped == "[wsl2]":
            in_wsl2 = True
            new_lines.append(line)
            continue
        if in_wsl2 and stripped.startswith("memory"):
            new_lines.append(f"memory={mem}")
            mem_written = True
            in_wsl2 = False
            continue
        if in_wsl2 and stripped.startswith("["):
            new_lines.append(f"memory={mem}")
            mem_written = True
            in_wsl2 = False
        new_lines.append(line)

    if not mem_written:
        if "[wsl2]" not in [l.strip().lower() for l in new_lines]:
            new_lines.append("[wsl2]")
        new_lines.append(f"memory={mem}")

    WSLCONFIG.write_text("\r\n".join(new_lines) + "\r\n", encoding="utf-8")
