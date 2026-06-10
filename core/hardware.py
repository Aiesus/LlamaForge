"""
One-time hardware profile detection.
Collects GPU, CPU, and RAM info at startup.
No tkinter imports.
"""
from __future__ import annotations
import subprocess
from dataclasses import dataclass, field


@dataclass
class GpuInfo:
    index:       int
    name:        str
    vram_mb:     int
    driver:      str


@dataclass
class HardwareProfile:
    gpus:              list[GpuInfo]  = field(default_factory=list)
    cpu_name:          str            = ""
    cpu_cores_phys:    int            = 0
    cpu_cores_logic:   int            = 0
    wsl_cpu_cores:     int            = 0
    ram_total_gb:      float          = 0.0
    detected:          bool           = False

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def vram_total_mb(self) -> int:
        return sum(g.vram_mb for g in self.gpus)

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_mb / 1024


def detect(distro: str = "", user: str = "") -> HardwareProfile:
    """
    Detect hardware. distro/user used for WSL CPU core count.
    Safe to call at startup — all failures produce empty/zero values.
    """
    hw = HardwareProfile()

    # ── GPUs (nvidia-smi) ──────────────────────────────────────────────────────
    try:
        r = subprocess.run(
            "nvidia-smi --query-gpu=index,name,memory.total,driver_version "
            "--format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True, timeout=5
        )
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                try:
                    hw.gpus.append(GpuInfo(
                        index   = int(parts[0]),
                        name    = parts[1],
                        vram_mb = int(parts[2]),
                        driver  = parts[3],
                    ))
                except ValueError:
                    pass
    except Exception:
        pass

    # ── CPU (wmic) ─────────────────────────────────────────────────────────────
    try:
        r = subprocess.run(
            "wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors /format:csv",
            shell=True, capture_output=True, text=True, timeout=5
        )
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            # wmic csv: Node,LogicalProcessors,Name,Cores  (order varies)
            # Use header row to map columns
            if line.lower().startswith("node"):
                headers = [h.strip().lower() for h in line.split(",")]
                continue
            try:
                # Fallback: just try to find integers for cores
                ints = [p for p in parts if p.isdigit()]
                name_parts = [p for p in parts if p and not p.isdigit() and p.lower() != "node"]
                if ints and len(ints) >= 2:
                    hw.cpu_cores_logic = int(ints[0])
                    hw.cpu_cores_phys  = int(ints[1])
                if name_parts:
                    hw.cpu_name = max(name_parts, key=len)  # longest string is the CPU name
            except Exception:
                pass
        # Clean up wmic oddities
        if not hw.cpu_name:
            r2 = subprocess.run(
                "wmic cpu get Name",
                shell=True, capture_output=True, text=True, timeout=5
            )
            lines = [l.strip() for l in r2.stdout.splitlines() if l.strip() and l.strip().lower() != "name"]
            if lines:
                hw.cpu_name = lines[0]
    except Exception:
        pass

    if not hw.cpu_cores_phys:
        try:
            r = subprocess.run(
                "wmic cpu get NumberOfCores",
                shell=True, capture_output=True, text=True, timeout=5
            )
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    hw.cpu_cores_phys = int(line)
                    break
        except Exception:
            pass

    if not hw.cpu_cores_logic:
        try:
            r = subprocess.run(
                "wmic cpu get NumberOfLogicalProcessors",
                shell=True, capture_output=True, text=True, timeout=5
            )
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    hw.cpu_cores_logic = int(line)
                    break
        except Exception:
            pass

    # ── WSL CPU cores ──────────────────────────────────────────────────────────
    if distro and user:
        try:
            r = subprocess.run(
                ["wsl", "-d", distro, "-u", user, "nproc"],
                capture_output=True, text=True, timeout=5
            )
            hw.wsl_cpu_cores = int(r.stdout.strip())
        except Exception:
            hw.wsl_cpu_cores = hw.cpu_cores_phys

    # ── Windows total RAM (wmic) ───────────────────────────────────────────────
    try:
        r = subprocess.run(
            "wmic ComputerSystem get TotalPhysicalMemory",
            shell=True, capture_output=True, text=True, timeout=5
        )
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                hw.ram_total_gb = int(line) / (1024 ** 3)
                break
    except Exception:
        pass

    hw.detected = True
    return hw


def summary_lines(hw: HardwareProfile) -> list[str]:
    """Return human-readable summary lines for display in the optimizer."""
    lines = []
    if hw.gpus:
        for g in hw.gpus:
            lines.append(f"GPU {g.index}: {g.name} — {g.vram_mb / 1024:.1f} GB VRAM")
        lines.append(f"Total VRAM: {hw.vram_total_gb:.1f} GB across {hw.gpu_count} GPU(s)")
    else:
        lines.append("GPU: not detected (nvidia-smi not found)")
    if hw.cpu_name:
        lines.append(f"CPU: {hw.cpu_name}")
    if hw.cpu_cores_phys:
        lines.append(f"CPU cores: {hw.cpu_cores_phys} physical / {hw.cpu_cores_logic} logical")
    if hw.wsl_cpu_cores:
        lines.append(f"WSL CPU cores: {hw.wsl_cpu_cores}")
    if hw.ram_total_gb:
        lines.append(f"System RAM: {hw.ram_total_gb:.1f} GB")
    return lines
