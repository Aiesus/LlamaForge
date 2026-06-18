"""
WSL virtual-disk maintenance.

WSL2's ext4.vhdx grows but never auto-shrinks, so space freed by deleting or
moving models stays claimed on the Windows host until the disk is compacted.
This locates the distro's vhdx and compacts it via an elevated diskpart.

No tkinter — the GUI drives this and shows the dialogs/progress.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import tempfile


def _lxss_basepath(distro: str) -> str:
    """Return the registered BasePath dir for a distro from the Lxss registry."""
    try:
        import winreg
    except Exception:
        return ""
    try:
        root = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Lxss")
    except Exception:
        return ""
    try:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            try:
                with winreg.OpenKey(root, sub) as k:
                    if winreg.QueryValueEx(k, "DistributionName")[0] == distro:
                        return winreg.QueryValueEx(k, "BasePath")[0]
            except OSError:
                continue
    finally:
        try:
            winreg.CloseKey(root)
        except Exception:
            pass
    return ""


def vhdx_info(distro: str) -> dict | None:
    """Return {path, drive, logical, host_free} for the distro's ext4.vhdx,
    or None if it can't be located."""
    bp = _lxss_basepath(distro)
    if not bp:
        return None
    path = os.path.join(bp, "ext4.vhdx")
    if not os.path.exists(path):
        return None
    drive = os.path.splitdrive(path)[0] or "C:"
    try:
        logical = os.path.getsize(path)
    except Exception:
        logical = 0
    try:
        host_free = shutil.disk_usage(drive + "\\").free
    except Exception:
        host_free = 0
    return {"path": path, "drive": drive, "logical": logical, "host_free": host_free}


def compact_vhdx(path: str, timeout: int = 1800) -> tuple[bool, str]:
    """Shut down WSL and compact the vhdx via an elevated diskpart (triggers a
    UAC prompt). Blocking — call on a worker thread. Returns (ok, message)."""
    try:
        subprocess.run(["wsl", "--shutdown"], capture_output=True, timeout=60)
    except Exception as e:
        return False, f"wsl --shutdown failed: {e}"

    script = (f'select vdisk file="{path}"\n'
              "attach vdisk readonly\n"
              "compact vdisk\n"
              "detach vdisk\n"
              "exit\n")
    fd, sp = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script)
        ps = (f"$p = Start-Process diskpart -ArgumentList '/s','{sp}' "
              f"-Verb RunAs -Wait -PassThru; exit $p.ExitCode")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return True, "ok"
        return False, (r.stderr or r.stdout or f"diskpart exit {r.returncode}").strip()
    except subprocess.TimeoutExpired:
        return False, "Timed out — the drive may be too full for compaction to finish."
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.remove(sp)
        except Exception:
            pass
