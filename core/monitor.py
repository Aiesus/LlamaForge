"""
Live hardware polling thread.
Polls GPU, WSL RAM, Windows CPU and RAM on a fixed interval.
Communicates back via registered callbacks — no tkinter imports.
"""
from __future__ import annotations
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


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
    gpus:            list[GpuStats] = field(default_factory=list)
    wsl_used_gb:     float = 0.0
    wsl_total_gb:    float = 0.0
    win_ram_used_gb: float = 0.0
    win_ram_total_gb:float = 0.0
    win_cpu_pct:     float = 0.0

    @property
    def wsl_pct(self) -> float:
        return self.wsl_used_gb / self.wsl_total_gb if self.wsl_total_gb else 0.0

    @property
    def win_ram_pct(self) -> float:
        return self.win_ram_used_gb / self.win_ram_total_gb if self.win_ram_total_gb else 0.0


MonitorCallback = Callable[[MonitorSnapshot], None]


class Monitor:
    """
    Starts a single daemon thread that polls hardware and fires registered
    callbacks with a fresh MonitorSnapshot every `interval` seconds.
    """

    def __init__(self, distro: str, user: str, interval: float = 2.0):
        self._distro   = distro
        self._user     = user
        self._interval = interval
        self._callbacks: list[MonitorCallback] = []
        self._running  = False
        self._thread: threading.Thread | None = None
        self._win_ram_total_gb: float = 0.0  # cached — doesn't change at runtime

    def register(self, cb: MonitorCallback) -> None:
        self._callbacks.append(cb)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        self._win_ram_total_gb = _win_ram_total()
        while self._running:
            snap = self._collect()
            for cb in self._callbacks:
                try:
                    cb(snap)
                except Exception:
                    pass
            time.sleep(self._interval)

    def _collect(self) -> MonitorSnapshot:
        snap = MonitorSnapshot()
        snap.win_ram_total_gb = self._win_ram_total_gb

        # ── GPUs ──────────────────────────────────────────────────────────────
        try:
            r = subprocess.run(
                "nvidia-smi "
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu "
                "--format=csv,noheader,nounits",
                shell=True, capture_output=True, text=True, timeout=5
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

        # ── WSL RAM ───────────────────────────────────────────────────────────
        try:
            r = subprocess.run(
                ["wsl", "-d", self._distro, "-u", self._user,
                 "bash", "-c", "free -b"],
                capture_output=True, text=True, timeout=5
            )
            for line in r.stdout.splitlines():
                if line.startswith("Mem:"):
                    cols = line.split()
                    total_b = int(cols[1])
                    used_b  = int(cols[2])
                    snap.wsl_total_gb = total_b / (1024 ** 3)
                    snap.wsl_used_gb  = used_b  / (1024 ** 3)
                    break
        except Exception:
            pass

        # ── Windows CPU % ─────────────────────────────────────────────────────
        try:
            r = subprocess.run(
                "wmic cpu get LoadPercentage",
                shell=True, capture_output=True, text=True, timeout=5
            )
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    snap.win_cpu_pct = float(line)
                    break
        except Exception:
            pass

        # ── Windows RAM ───────────────────────────────────────────────────────
        try:
            r = subprocess.run(
                "wmic OS get FreePhysicalMemory",
                shell=True, capture_output=True, text=True, timeout=5
            )
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    free_kb = int(line)
                    snap.win_ram_used_gb = (
                        self._win_ram_total_gb - free_kb / (1024 ** 2)
                    )
                    break
        except Exception:
            pass

        return snap


def _win_ram_total() -> float:
    """One-time read of total Windows physical RAM in GB."""
    try:
        r = subprocess.run(
            "wmic ComputerSystem get TotalPhysicalMemory",
            shell=True, capture_output=True, text=True, timeout=5
        )
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line) / (1024 ** 3)
    except Exception:
        pass
    return 0.0
