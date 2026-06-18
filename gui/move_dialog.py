"""
Move-models progress dialog.

Copies selected .gguf files to a destination library, verifies them, then
deletes the originals (freeing the source drive) and repoints every stored
path (profiles / last_model / model_libraries / live selection).

Copy and delete are separate steps — a cancelled or failed copy never deletes
the originals.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gui.app import AppState

from core import wsl
from core import library_move as lm
from core.settings import save_settings, save_profiles

LogFn = Callable[[str, str | None], None]


class MoveDialog:
    def __init__(self, root, state: "AppState", T: dict, log_fn: LogFn,
                 srcs: list[str], total_bytes: int,
                 old_base: str, dest_lib: str,
                 basenames: list[str] | None,
                 on_done: Callable[[], None] | None = None):
        self._root  = root
        self._state = state
        self._T     = T
        self._log   = log_fn
        self._srcs  = srcs
        self._total = total_bytes
        self._old   = old_base
        self._dest  = dest_lib
        self._bn    = basenames        # None = whole-library move
        self._on_done = on_done

        self._proc = None
        self._cancelled = False
        self._use_rsync = True
        self._win: tk.Toplevel | None = None

    # ── UI ────────────────────────────────────────────────────────────────────
    def show(self) -> None:
        T = self._T
        win = tk.Toplevel(self._root)
        self._win = win
        win.title("Move Models")
        win.geometry("520x200")
        win.configure(bg=T["bg"])
        win.transient(self._root)
        win.grab_set()
        win.protocol("WM_DELETE_WINDOW", self._on_cancel)

        n = len(self._srcs)
        tk.Label(win, text=f"Moving {n} model{'s' if n != 1 else ''} → {self._dest}",
                 bg=T["bg"], fg=T["accent"], font=("Consolas", 10, "bold"),
                 anchor="w", justify="left", wraplength=480).pack(
                     fill="x", padx=14, pady=(14, 4))

        self._status = tk.Label(win, text="Preparing…", bg=T["bg"], fg=T["fg2"],
                                font=("Segoe UI", 9), anchor="w", justify="left",
                                wraplength=480)
        self._status.pack(fill="x", padx=14, pady=(0, 8))

        self._bar = ttk.Progressbar(win, orient="horizontal", mode="determinate",
                                    maximum=100, length=480)
        self._bar.pack(padx=14, pady=4)

        self._btn = tk.Button(win, text="Cancel", bg=T["red"], fg=T["bg"],
                              relief="flat", cursor="hand2", font=("Segoe UI", 9, "bold"),
                              command=self._on_cancel)
        self._btn.pack(pady=10)

        # Run pre-checks + copy off the main thread.
        import threading
        threading.Thread(target=self._prechecks_and_start, daemon=True).start()

    # ── Pre-checks (worker thread) ──────────────────────────────────────────────
    def _prechecks_and_start(self) -> None:
        s = self._state.settings
        d, u = s.wsl_distro, s.wsl_user

        self._set_status("Checking free space…")
        avail = lm.free_space_bytes(d, u, self._dest)
        if avail >= 0 and self._total > 0 and avail < self._total:
            self._ui(self._fail, f"Not enough space on destination.\n"
                     f"Need {self._total // (1024**2)} MB, "
                     f"have {avail // (1024**2)} MB free.")
            return

        self._use_rsync = lm.rsync_available(d, u)

        # Collision check
        self._set_status("Checking for name collisions…")
        bns = self._bn if self._bn is not None else [lm.basename_wsl(p) for p in self._srcs]
        try:
            r = wsl.run(d, u, lm.build_collision_check_cmd(bns, self._dest, u), timeout=20)
            collisions = lm.parse_collisions(r.stdout)
        except Exception:
            collisions = []
        if collisions:
            # Ask on main thread, block worker until answered
            if not self._ask_overwrite(collisions):
                self._ui(self._abort, "Cancelled — destination already has files with the same name.")
                return

        # Start copy
        cmd = lm.build_move_cmd(self._srcs, self._dest, u, self._use_rsync)
        self._set_status(f"Copying with {'rsync' if self._use_rsync else 'cp'}…")
        if not self._use_rsync:
            self._root.after(0, lambda: self._bar.config(mode="indeterminate"))
            self._root.after(0, self._bar.start)
        self._log(f"[MOVE] {cmd}", "info")

        wsl.stream_async(d, u, cmd, self._on_copy_line,
                         done_fn=self._on_copy_done, timeout=86400,
                         on_proc=self._capture_proc)

    # ── Copy callbacks (worker thread) ──────────────────────────────────────────
    def _capture_proc(self, proc) -> None:
        self._proc = proc

    def _on_copy_line(self, line: str, _sev) -> None:
        pct = lm.parse_rsync_progress(line)
        if pct is not None:
            self._root.after(0, lambda v=pct: self._bar.config(value=v))

    def _on_copy_done(self, rc: int) -> None:
        if self._cancelled:
            self._root.after(0, lambda: self._abort("Cancelled — originals left untouched."))
            return
        if rc != 0:
            self._root.after(0, lambda: self._fail(
                f"Copy failed (exit {rc}). Originals left untouched."))
            return

        # Verify (worker thread, synchronous)
        self._set_status("Verifying copies…")
        s = self._state.settings
        try:
            r = wsl.run(s.wsl_distro, s.wsl_user,
                        lm.build_verify_cmd(self._srcs, self._dest, s.wsl_user),
                        timeout=120)
            ok, fail = lm.parse_verify(r.stdout)
        except Exception as e:
            self._root.after(0, lambda: self._fail(f"Verify error: {e}\nOriginals kept."))
            return
        if fail or len(ok) != len(self._srcs):
            self._root.after(0, lambda: self._fail(
                "Verification failed for: " + ", ".join(fail or ["(missing files)"]) +
                "\nOriginals kept — nothing deleted."))
            return

        # Delete originals (only after full verify)
        self._set_status("Verified. Removing originals…")
        try:
            wsl.run(s.wsl_distro, s.wsl_user,
                    lm.build_delete_cmd(self._srcs, s.wsl_user), timeout=120)
        except Exception as e:
            self._root.after(0, lambda: self._fail(
                f"Copied & verified, but failed to delete originals: {e}"))
            return

        self._root.after(0, self._finalize_success)

    # ── Finalize (main thread) ──────────────────────────────────────────────────
    def _finalize_success(self) -> None:
        s = self._state
        res = lm.remap_paths(s.settings, s.profiles, self._old, self._dest, self._bn)
        save_settings(s.settings)
        save_profiles(s.profiles)

        # Repoint the live selection if the loaded/selected model moved
        cur = s.model_var.get()
        new_cur = lm.remap_one(s.settings, cur, self._old, self._dest, self._bn)
        if new_cur:
            s.model_var.set(new_cur)

        self._log(f"[MOVE] Done. {len(self._srcs)} file(s) moved; "
                  f"{res.changed_profiles} profile(s) repointed.", "success")
        try:
            self._bar.config(mode="determinate", value=100)
        except Exception:
            pass
        if self._on_done:
            self._on_done()
        messagebox.showinfo(
            "Move complete",
            f"Moved {len(self._srcs)} model(s) to:\n{self._dest}\n\n"
            f"Repointed {res.changed_profiles} profile(s)"
            + (", last-used model, and library list." if res.libraries_changed
               else " and last-used model."),
            parent=self._win)
        self._close()

    # ── Cancel / fail / status helpers ───────────────────────────────────────────
    def _on_cancel(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._cancelled = True
            self._set_status("Cancelling…")
            try:
                self._proc.terminate()
            except Exception:
                pass
        else:
            # Not copying yet (or already done) — just close.
            self._cancelled = True
            self._close()

    def _abort(self, msg: str) -> None:
        try:
            self._bar.stop()
        except Exception:
            pass
        self._log(f"[MOVE] {msg}", "warn")
        messagebox.showinfo("Move cancelled", msg, parent=self._win)
        self._close()

    def _fail(self, msg: str) -> None:
        try:
            self._bar.stop()
        except Exception:
            pass
        self._log(f"[MOVE] {msg}", "error")
        messagebox.showerror("Move failed", msg, parent=self._win)
        self._close()

    def _ask_overwrite(self, collisions: list[str]) -> bool:
        """Blocking ask from a worker thread, marshalled to the main thread."""
        result = {}
        ev = __import__("threading").Event()
        def _ask():
            preview = "\n".join(collisions[:8]) + ("\n…" if len(collisions) > 8 else "")
            result["ok"] = messagebox.askyesno(
                "Files already exist",
                f"The destination already contains:\n{preview}\n\nOverwrite them?",
                parent=self._win, icon="warning")
            ev.set()
        self._root.after(0, _ask)
        ev.wait()
        return result.get("ok", False)

    def _ui(self, fn, *args) -> None:
        """Run a Tk-touching callback on the main thread."""
        self._root.after(0, lambda: fn(*args))

    def _set_status(self, text: str) -> None:
        self._root.after(0, lambda: self._status.config(text=text))

    def _close(self) -> None:
        if self._win:
            try:
                self._win.grab_release()
            except Exception:
                pass
            self._win.destroy()
            self._win = None
