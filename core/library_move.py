"""
Model-library move engine.

Moves .gguf files between WSL model libraries (e.g. off a full C: drive onto a
/mnt/d Windows-drive mount) and repoints every stored reference so nothing
breaks: profiles' "model" keys, settings.last_model, and model_libraries.

No tkinter — the GUI drives this via core.wsl streaming + callbacks.

Safety model: copy and delete are SEPARATE commands. The caller copies, then
verifies, and only then runs the delete command. A cancelled or failed copy
therefore can never delete the originals.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

from core import wsl


# ── Path helpers ────────────────────────────────────────────────────────────

def expand_wsl(path: str, user: str) -> str:
    """Expand a leading ~ to /home/<user>. Leave /mnt/... and absolute paths."""
    path = path.strip()
    if path == "~":
        return f"/home/{user}"
    if path.startswith("~/"):
        return f"/home/{user}/{path[2:]}"
    return path


def _q(p: str) -> str:
    return shlex.quote(p)


def dirname_wsl(model: str) -> str:
    """Directory part of a model path (forward-slash semantics)."""
    return model.rsplit("/", 1)[0] if "/" in model else ""


def basename_wsl(model: str) -> str:
    return model.rsplit("/", 1)[-1]


# ── Capability / space checks ────────────────────────────────────────────────

def rsync_available(distro: str, user: str) -> bool:
    try:
        return wsl.run(distro, user, "command -v rsync", timeout=10).returncode == 0
    except Exception:
        return False


def build_df_check_cmd(dest_wsl: str, user: str) -> str:
    """df on the destination, walking up to the nearest existing parent dir
    (the destination itself may not exist yet)."""
    d = _q(expand_wsl(dest_wsl, user))
    return (f'd={d}; while [ ! -d "$d" ]; do d=$(dirname "$d"); done; '
            f'df -PB1 "$d"')


def parse_df_avail(output: str) -> int:
    """Return available bytes from `df -PB1` output (row 2, column 4)."""
    try:
        lines = [ln for ln in output.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return -1
        return int(lines[-1].split()[3])
    except Exception:
        return -1


def free_space_bytes(distro: str, user: str, dest_wsl: str) -> int:
    try:
        r = wsl.run(distro, user, build_df_check_cmd(dest_wsl, user), timeout=15)
        return parse_df_avail(r.stdout)
    except Exception:
        return -1


# ── Command builders (copy / collision / verify / delete) ─────────────────────

def build_move_cmd(srcs: list[str], dest_dir: str, user: str,
                   use_rsync: bool) -> str:
    """Copy-only command (NO delete). dest_dir is created if missing."""
    esrcs = " ".join(_q(expand_wsl(s, user)) for s in srcs)
    dest = expand_wsl(dest_dir, user)
    qdest = _q(dest)
    if use_rsync:
        # -rt (not -a): skip perms/owner/group preservation, which is noisy and
        # meaningless on a drvfs (/mnt) target. Explicit file args + dir dest.
        copy = f"rsync -rt --info=progress2 --no-inc-recursive {esrcs} {qdest}/"
    else:
        copy = f"cp {esrcs} {qdest}/"
    return f"mkdir -p {qdest} && {copy}"


def build_collision_check_cmd(basenames: list[str], dest_dir: str, user: str) -> str:
    dest = expand_wsl(dest_dir, user)
    names = " ".join(_q(b) for b in basenames)
    return (f'dest={_q(dest)}; for b in {names}; do '
            f'[ -e "$dest/$b" ] && echo "COLLIDE:$b"; done; true')


def parse_collisions(output: str) -> list[str]:
    return [ln[len("COLLIDE:"):].strip()
            for ln in output.splitlines() if ln.startswith("COLLIDE:")]


def build_verify_cmd(srcs: list[str], dest_dir: str, user: str) -> str:
    """Compare each source's byte size to its copy in dest_dir."""
    esrcs = " ".join(_q(expand_wsl(s, user)) for s in srcs)
    dest = expand_wsl(dest_dir, user)
    return (
        f'dest={_q(dest)}; for s in {esrcs}; do '
        f'b=$(basename "$s"); '
        f'ss=$(stat -c %s "$s" 2>/dev/null || echo -1); '
        f'ds=$(stat -c %s "$dest/$b" 2>/dev/null || echo -2); '
        f'if [ "$ss" = "$ds" ]; then echo "VERIFY_OK:$b"; '
        f'else echo "VERIFY_FAIL:$b ($ss vs $ds)"; fi; done'
    )


def parse_verify(output: str) -> tuple[list[str], list[str]]:
    """Return (ok_basenames, fail_descriptions)."""
    ok, fail = [], []
    for ln in output.splitlines():
        if ln.startswith("VERIFY_OK:"):
            ok.append(ln[len("VERIFY_OK:"):].strip())
        elif ln.startswith("VERIFY_FAIL:"):
            fail.append(ln[len("VERIFY_FAIL:"):].strip())
    return ok, fail


def build_delete_cmd(srcs: list[str], user: str) -> str:
    """Delete the ORIGINAL source files. Caller must run this only after a
    fully-successful verify of every file."""
    esrcs = " ".join(_q(expand_wsl(s, user)) for s in srcs)
    return f"rm -f {esrcs}"


_PCT_RE = re.compile(r"(\d{1,3})%")


def parse_rsync_progress(line: str) -> int | None:
    """Pull an overall percentage out of an rsync --info=progress2 line."""
    m = _PCT_RE.search(line)
    if m:
        try:
            v = int(m.group(1))
            if 0 <= v <= 100:
                return v
        except Exception:
            pass
    return None


# ── Pure path remap ───────────────────────────────────────────────────────────

@dataclass
class RemapResult:
    changed_profiles: int = 0
    last_model_changed: bool = False
    libraries_changed: bool = False
    # full old path -> full new path, for every model that moved (so the GUI
    # can repoint the live selection)
    moved: dict = field(default_factory=dict)


def remap_one(settings, model: str, old_base: str, new_base: str,
              basenames: list[str] | None) -> str | None:
    """If `model` lives in old_base (and, when given, is in basenames), return
    its new full path; otherwise None. Handles legacy bare filenames (their
    implicit base is the first library, settings.models_wsl)."""
    if not model:
        return None
    user = settings.wsl_user
    if "/" in model:
        d = dirname_wsl(model)
        n = basename_wsl(model)
    else:
        d = settings.models_wsl   # legacy bare filename → default library
        n = model
    if expand_wsl(d, user) != expand_wsl(old_base, user):
        return None
    if basenames is not None and n not in basenames:
        return None
    return f"{new_base}/{n}"


def remap_paths(settings, profiles: dict, old_base: str, new_base: str,
                basenames: list[str] | None = None) -> RemapResult:
    """Repoint every stored reference from old_base → new_base. Mutates the
    passed settings object and profiles dict in place; the caller persists them
    with save_settings / save_profiles. Returns a summary.

    basenames is None  → whole-library move (also rewrites model_libraries).
    basenames is a list → per-model move (libraries left unchanged).
    """
    res = RemapResult()

    # Profiles
    for prof in profiles.values():
        new = remap_one(settings, prof.get("model", ""), old_base, new_base, basenames)
        if new is not None:
            res.moved[prof["model"]] = new
            prof["model"] = new
            res.changed_profiles += 1

    # last_model
    new_last = remap_one(settings, settings.last_model, old_base, new_base, basenames)
    if new_last is not None:
        res.moved[settings.last_model] = new_last
        settings.last_model = new_last
        res.last_model_changed = True

    # model_libraries (whole-library move only)
    if basenames is None:
        libs = settings.model_libraries
        user = settings.wsl_user
        old_e = expand_wsl(old_base, user)
        idx = next((i for i, l in enumerate(libs)
                    if expand_wsl(l, user) == old_e), -1)
        if idx >= 0:
            already = any(expand_wsl(l, user) == expand_wsl(new_base, user)
                          for l in libs)
            if already:
                libs.pop(idx)
            else:
                libs[idx] = new_base   # preserve position (incl. first=default)
            res.libraries_changed = True

    return res
