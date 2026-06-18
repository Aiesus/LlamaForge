"""
Live hardware polling thread.
Polls GPU, WSL RAM, Windows CPU and RAM on a fixed interval.
Communicates back via registered callbacks — no tkinter imports.

GPU:         pynvml (in-process, zero subprocess overhead) with subprocess fallback
Windows:     ctypes GlobalMemoryStatusEx + GetSystemTimes (microsecond latency)
WSL RAM:     subprocess every `wsl_every` cycles (default every 3rd = 15 s at 5 s interval)
"""
from __future__ import annotations
import ctypes
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

# ── pynvml (preferred GPU backend) ───────────────────────────────────────────
try:
    import pynvml as _nvml
    _PYNVML = True
except Exception:
    _nvml = None   # type: ignore
    _PYNVML = False

# ── Windows ctypes for RAM + CPU ─────────────────────────────────────────────
if sys.platform == "win32":
    from ctypes import wintypes as _wt

    class _MEMSTATEX(ctypes.Structure):
        _fields_ = [
            ("dwLength",                _wt.DWORD),
            ("dwMemoryLoad",            _wt.DWORD),
            ("ullTotalPhys",            ctypes.c_uint64),
            ("ullAvailPhys",            ctypes.c_uint64),
            ("ullTotalPageFile",        ctypes.c_uint64),
            ("ullAvailPageFile",        ctypes.c_uint64),
            ("ullTotalVirtual",         ctypes.c_uint64),
            ("ullAvailVirtual",         ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    class _FILETIME(ctypes.Structure):
        _fields_ = [("dwLow", _wt.DWORD), ("dwHigh", _wt.DWORD)]

    def _ft(ft: "_FILETIME") -> int:
        return (ft.dwHigh << 32) | ft.dwLow

    _k32 = ctypes.windll.kernel32
    _WIN_CTYPES = True
else:
    _WIN_CTYPES = False


@dataclass
class GpuStats:
    index:    int
    used_mb:  int
    total_mb: int
    util_pct: int
    temp_c:   int

    @property
    def pct(self) -> float:
        return self.used_mb / self.total_mb if self.total_mb else 0.0


@dataclass
class MonitorSnapshot:
    gpus:             list[GpuStats] = field(default_factory=list)
    wsl_used_gb:      float = 0.0
    wsl_total_gb:     float = 0.0
    win_ram_used_gb:  float = 0.0
    win_ram_total_gb: float = 0.0
    win_cpu_pct:      float = 0.0

    @property
    def wsl_pct(self) -> float:
        return self.wsl_used_gb / self.wsl_total_gb if self.wsl_total_gb else 0.0

    @property
    def win_ram_pct(self) -> float:
        return self.win_ram_used_gb / self.win_ram_total_gb if self.win_ram_total_gb else 0.0


MonitorCallback = Callable[[MonitorSnapshot], None]


class Monitor:
    """
    Daemon thread that polls hardware every `interval` seconds and fires
    registered callbacks with a fresh MonitorSnapshot.

    WSL RAM is polled only every `wsl_every` cycles (default 3) because the
    cross-hypervisor subprocess is slow — RAM changes slowly anyway.
    """

    def __init__(self, distro: str, user: str,
                 interval: float = 5.0, wsl_every: int = 3):
        self._distro    = distro
        self._user      = user
        self._interval  = interval
        self._wsl_every = wsl_every
        self._callbacks: list[MonitorCallback] = []
        self._running   = False
        self._thread: threading.Thread | None = None

        # GPU state
        self._nvml_ok   = False

        # WSL poll throttle
        self._wsl_tick  = wsl_every   # start at limit → first cycle polls WSL
        self._cached_wsl: tuple[float, float] = (0.0, 0.0)
        self._wsl_paused = False       # set True during wsl --shutdown / restart

        # CPU delta tracking (GetSystemTimes returns cumulative values)
        self._prev_idle   = 0
        self._prev_kernel = 0
        self._prev_user   = 0

    def register(self, cb: MonitorCallback) -> None:
        self._callbacks.append(cb)

    def pause_wsl(self) -> None:
        """Stop WSL subprocess polling (call before wsl --shutdown)."""
        self._wsl_paused = True

    def resume_wsl(self) -> None:
        """Resume WSL subprocess polling and force a fresh read next cycle."""
        self._wsl_paused = False
        self._wsl_tick   = self._wsl_every  # trigger immediate poll on next cycle

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        # nvmlShutdown is called from inside _loop after the while-loop exits

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        # Init pynvml here (background thread) — nvmlInit on the main thread
        # blocks for ~100-500 ms while the NVML driver initialises.
        if _PYNVML:
            try:
                _nvml.nvmlInit()
                self._nvml_ok = True
            except Exception:
                self._nvml_ok = False

        while self._running:
            snap = self._collect()
            for cb in self._callbacks:
                try:
                    cb(snap)
                except Exception:
                    pass
            time.sleep(self._interval)

        if self._nvml_ok:
            try:
                _nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_ok = False

    def _collect(self) -> MonitorSnapshot:
        snap = MonitorSnapshot()
        self._collect_gpus(snap)
        self._collect_windows(snap)

        self._wsl_tick += 1
        if not self._wsl_paused and self._wsl_tick >= self._wsl_every:
            self._wsl_tick = 0
            self._collect_wsl(snap)
            if snap.wsl_total_gb > 0:
                self._cached_wsl = (snap.wsl_used_gb, snap.wsl_total_gb)
            else:
                snap.wsl_used_gb, snap.wsl_total_gb = self._cached_wsl
        else:
            snap.wsl_used_gb, snap.wsl_total_gb = self._cached_wsl

        return snap

    def _collect_gpus(self, snap: MonitorSnapshot) -> None:
        if self._nvml_ok:
            try:
                count = _nvml.nvmlDeviceGetCount()
                for i in range(count):
                    h    = _nvml.nvmlDeviceGetHandleByIndex(i)
                    mem  = _nvml.nvmlDeviceGetMemoryInfo(h)
                    util = _nvml.nvmlDeviceGetUtilizationRates(h)
                    temp = _nvml.nvmlDeviceGetTemperature(h, _nvml.NVML_TEMPERATURE_GPU)
                    snap.gpus.append(GpuStats(
                        index    = i,
                        used_mb  = int(mem.used  // (1024 * 1024)),
                        total_mb = int(mem.total // (1024 * 1024)),
                        util_pct = int(util.gpu),
                        temp_c   = int(temp),
                    ))
                return
            except Exception:
                pass
        # Fallback: subprocess nvidia-smi
        try:
            r = subprocess.run(
                "nvidia-smi "
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu "
                "--format=csv,noheader,nounits",
                shell=True, capture_output=True, text=True, timeout=5,
            )
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 5:
                    try:
                        snap.gpus.append(GpuStats(
                            index    = int(parts[0]),
                            used_mb  = int(parts[1]),
                            total_mb = int(parts[2]),
                            util_pct = int(parts[3]),
                            temp_c   = int(parts[4]),
                        ))
                    except ValueError:
                        pass
        except Exception:
            pass

    def _collect_windows(self, snap: MonitorSnapshot) -> None:
        if not _WIN_CTYPES:
            return
        # RAM (instant — single syscall)
        try:
            ms = _MEMSTATEX()
            ms.dwLength = ctypes.sizeof(ms)
            _k32.GlobalMemoryStatusEx(ctypes.byref(ms))
            total_gb = ms.ullTotalPhys / (1024 ** 3)
            avail_gb = ms.ullAvailPhys / (1024 ** 3)
            snap.win_ram_total_gb = total_gb
            snap.win_ram_used_gb  = total_gb - avail_gb
        except Exception:
            pass
        # CPU% via GetSystemTimes delta (two syscalls, no process spawn)
        try:
            idle = _FILETIME()
            kern = _FILETIME()
            user = _FILETIME()
            _k32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
            ci, ck, cu = _ft(idle), _ft(kern), _ft(user)
            dk = ck - self._prev_kernel
            du = cu - self._prev_user
            di = ci - self._prev_idle
            total = dk + du
            # Skip calculation on first call (prev values are 0 → delta = entire uptime)
            if total > 0 and (self._prev_kernel > 0 or self._prev_user > 0):
                snap.win_cpu_pct = max(0.0, min(100.0, (total - di) / total * 100.0))
            self._prev_idle   = ci
            self._prev_kernel = ck
            self._prev_user   = cu
        except Exception:
            pass

    def _collect_wsl(self, snap: MonitorSnapshot) -> None:
        try:
            r = subprocess.run(
                ["wsl", "-d", self._distro, "-u", self._user,
                 "bash", "-c", "free -b"],
                capture_output=True, text=True, timeout=5,
            )
            for line in r.stdout.splitlines():
                if line.startswith("Mem:"):
                    cols = line.split()
                    snap.wsl_total_gb = int(cols[1]) / (1024 ** 3)
                    snap.wsl_used_gb  = int(cols[2]) / (1024 ** 3)
                    break
        except Exception:
            pass
