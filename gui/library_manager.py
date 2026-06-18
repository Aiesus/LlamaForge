"""
Library Manager — manage model storage directories (Steam-style) and the
models inside them: browse, move between drives, delete.
"""
from __future__ import annotations
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gui.app import AppState

from core.settings import save_settings


def _fmt_size(size_b: int) -> str:
    if size_b <= 0:
        return "—"
    gb = size_b / 1024 ** 3
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    return f"{size_b / 1024 ** 2:.0f} MB"


class LibraryManager:

    def __init__(self, root: tk.Tk, state: "AppState", T: dict,
                 on_close: Callable[[], None] | None = None,
                 log_fn: Callable[[str, str | None], None] | None = None):
        self._root     = root
        self._state    = state
        self._T        = T
        self._on_close = on_close
        self._log_fn   = log_fn
        self._win: tk.Toplevel | None = None

        # models in the currently-selected library: full_wsl_path -> size_bytes
        self._cur_lib: str = ""
        self._cur_sizes: dict[str, int] = {}
        self._msort_col = "name"   # sort column for the models panel
        self._msort_rev = False

    # ── Window ────────────────────────────────────────────────────────────────
    def show(self) -> None:
        T = self._T
        win = tk.Toplevel(self._root)
        win.title("Model Libraries")
        win.geometry("660x640")
        win.configure(bg=T["bg"])
        win.transient(self._root)
        win.grab_set()
        self._win = win

        tk.Label(win, text="MODEL LIBRARIES", bg=T["bg"], fg=T["accent"],
                 font=("Consolas", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
        tk.Label(win, text="All libraries are scanned for models.  First entry ★ = default download destination.",
                 bg=T["bg"], fg=T["fg2"], font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(0, 8))

        self._init_styles(T)

        # ── Libraries tree ────────────────────────────────────────────────────
        lib_frame = tk.Frame(win, bg=T["bg"])
        lib_frame.pack(fill="x", padx=14, pady=2)
        self._tree = ttk.Treeview(lib_frame, columns=("path", "models", "size"),
                                  show="headings", selectmode="browse",
                                  style="Libs.Treeview", height=5)
        self._tree.heading("path",   text="Path (WSL)")
        self._tree.heading("models", text="Models")
        self._tree.heading("size",   text="Total Size")
        self._tree.column("path",   width=360, stretch=True,  anchor="w")
        self._tree.column("models", width=65,  stretch=False, anchor="center")
        self._tree.column("size",   width=100, stretch=False, anchor="e")
        self._tree.tag_configure("default", foreground=T["accent"])
        vsb = ttk.Scrollbar(lib_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", lambda e: self._on_lib_select())

        lib_btns = tk.Frame(win, bg=T["bg"])
        lib_btns.pack(fill="x", padx=14, pady=(6, 2))
        self._mkbtn(lib_btns, "+ Add…",      T["green"], T["bg"], self._add_library, bold=True)
        self._mkbtn(lib_btns, "↑",           T["btn"], T["btn_fg"], self._move_up, width=3)
        self._mkbtn(lib_btns, "↓",           T["btn"], T["btn_fg"], self._move_down, width=3)
        self._mkbtn(lib_btns, "✕ Remove",    T["red"], T["bg"], self._remove_library, bold=True)
        self._mkbtn(lib_btns, "➜ Move library…", T["btn"], T["accent"], self._move_library, bold=True)
        self._mkbtn(lib_btns, "Explorer",    T["btn"], T["btn_fg"], lambda: self._open_in_explorer(self._cur_lib), side="right")

        # ── Models-in-library panel ───────────────────────────────────────────
        tk.Label(win, text="MODELS IN SELECTED LIBRARY", bg=T["bg"], fg=T["accent"],
                 font=("Consolas", 9, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
        m_frame = tk.Frame(win, bg=T["bg"])
        m_frame.pack(fill="both", expand=True, padx=14, pady=2)
        self._mtree = ttk.Treeview(m_frame, columns=("name", "size", "quant"),
                                   show="headings", selectmode="extended",
                                   style="Libs.Treeview")
        self._mtree.heading("name",  text="Name",  command=lambda: self._sort_models("name"))
        self._mtree.heading("size",  text="Size",  command=lambda: self._sort_models("size"))
        self._mtree.heading("quant", text="Quant", command=lambda: self._sort_models("quant"))
        self._mtree.column("name",  width=360, stretch=True,  anchor="w")
        self._mtree.column("size",  width=90,  stretch=False, anchor="e")
        self._mtree.column("quant", width=90,  stretch=False, anchor="center")
        mvsb = ttk.Scrollbar(m_frame, orient="vertical", command=self._mtree.yview)
        self._mtree.configure(yscrollcommand=mvsb.set)
        mvsb.pack(side="right", fill="y")
        self._mtree.pack(fill="both", expand=True)

        m_btns = tk.Frame(win, bg=T["bg"])
        m_btns.pack(fill="x", padx=14, pady=(6, 2))
        self._mkbtn(m_btns, "➜ Move selected…", T["btn"], T["accent"], self._move_selected, bold=True)
        self._mkbtn(m_btns, "🗑 Delete",         T["red"], T["bg"], self._delete_selected, bold=True)

        tk.Button(win, text="Close", bg=T["btn"], fg=T["btn_fg"], relief="flat",
                  cursor="hand2", font=("Segoe UI", 9), pady=4,
                  command=self._close).pack(side="right", padx=14, pady=8)

        self._refresh_tree()
        win.protocol("WM_DELETE_WINDOW", self._close)

    def _init_styles(self, T: dict) -> None:
        style = ttk.Style()
        style.configure("Libs.Treeview", background=T["bg3"], fieldbackground=T["bg3"],
                        foreground=T["fg"], rowheight=24, font=("Consolas", 9), borderwidth=0)
        style.configure("Libs.Treeview.Heading", background=T["bg2"], foreground=T["fg2"],
                        font=("Segoe UI", 8, "bold"), relief="flat")
        style.map("Libs.Treeview", background=[("selected", T["select_bg"])],
                  foreground=[("selected", T["select_fg"])])

    def _mkbtn(self, parent, text, bg, fg, cmd, *, bold=False, width=None, side="left"):
        b = tk.Button(parent, text=text, bg=bg, fg=fg, relief="flat", cursor="hand2",
                      font=("Segoe UI", 9, "bold" if bold else "normal"), pady=4,
                      command=cmd)
        if width:
            b.config(width=width)
        b.pack(side=side, padx=2)
        return b

    # ── Libraries tree ────────────────────────────────────────────────────────
    def _scan_lib(self, lib_wsl: str) -> tuple[int, int, dict[str, int]]:
        """Return (n_models, total_bytes, {full_wsl_path: size})."""
        n, total = 0, 0
        sizes: dict[str, int] = {}
        unc = self._state.settings.wsl_path_to_unc(lib_wsl)
        try:
            p = Path(unc)
            if p.exists():
                for f in p.iterdir():
                    if f.suffix.lower() == ".gguf":
                        try:
                            sz = f.stat().st_size
                        except Exception:
                            sz = 0
                        sizes[f"{lib_wsl}/{f.name}"] = sz
                        n += 1
                        total += sz
        except Exception:
            pass
        return n, total, sizes

    def _refresh_tree(self) -> None:
        sel = self._cur_lib
        for row in self._tree.get_children():
            self._tree.delete(row)
        libs = self._state.settings.model_libraries
        for i, lib_wsl in enumerate(libs):
            n, total, _ = self._scan_lib(lib_wsl)
            label = lib_wsl + (" ★" if i == 0 else "")
            self._tree.insert("", tk.END, iid=lib_wsl,
                              values=(label, str(n), _fmt_size(total)),
                              tags=(("default",) if i == 0 else ()))
        if sel and sel in libs:
            self._tree.selection_set(sel)
        elif libs:
            self._tree.selection_set(libs[0])

    def _on_lib_select(self) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        self._cur_lib = sel[0]
        self._refresh_models_panel()

    _TIER_ORDER = {"lossless": 0, "high": 1, "balanced": 2, "low": 3, "vlow": 4}

    def _sort_models(self, col: str) -> None:
        if self._msort_col == col:
            self._msort_rev = not self._msort_rev
        else:
            self._msort_col = col
            self._msort_rev = (col == "size")   # size defaults largest-first
        self._refresh_models_panel()

    def _refresh_models_panel(self) -> None:
        from gui.left_panel import _parse_gguf
        for row in self._mtree.get_children():
            self._mtree.delete(row)
        if not self._cur_lib:
            self._cur_sizes = {}
            return
        _, _, sizes = self._scan_lib(self._cur_lib)
        self._cur_sizes = sizes

        rows = []
        for path, size in sizes.items():
            fname = path.split("/")[-1]
            meta  = _parse_gguf(fname)
            rows.append((path, fname, meta, size))

        col = self._msort_col
        if col == "size":
            keyfn = lambda r: r[3]
        elif col == "quant":
            keyfn = lambda r: (self._TIER_ORDER.get(r[2].get("tier", "balanced"), 2),
                               r[1].lower())
        else:  # name
            keyfn = lambda r: r[1].lower()
        rows.sort(key=keyfn, reverse=self._msort_rev)

        # heading arrows
        for c, base in (("name", "Name"), ("size", "Size"), ("quant", "Quant")):
            arrow = (" ▼" if self._msort_rev else " ▲") if c == col else ""
            self._mtree.heading(c, text=base + arrow)

        for path, fname, meta, size in rows:
            self._mtree.insert("", tk.END, iid=path,
                               values=(meta.get("display", fname),
                                       _fmt_size(size),
                                       meta.get("quant", "")))

    # ── Library add / remove / reorder ────────────────────────────────────────
    def _add_library(self) -> None:
        path = simpledialog.askstring(
            "Add Library",
            "Enter WSL path to model directory:\nExamples:  ~/extra-models   /mnt/d/LLM-models",
            parent=self._win)
        if not path:
            return
        path = path.strip()
        if not path:
            return
        s = self._state.settings
        if path in s.model_libraries:
            messagebox.showwarning("Duplicate", f"'{path}' is already a library.", parent=self._win)
            return
        unc = s.wsl_path_to_unc(path)
        if unc:
            try:
                if not Path(unc).exists() and not messagebox.askyesno(
                    "Path not found",
                    f"'{path}' does not appear to exist yet.\nAdd it anyway?",
                    parent=self._win):
                    return
            except Exception:
                pass
        s.model_libraries.append(path)
        save_settings(s)
        self._refresh_tree()
        self._notify_parent()

    def _remove_library(self) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        path = sel[0]
        s = self._state.settings
        if path not in s.model_libraries:
            return
        if not messagebox.askyesno(
            "Remove Library",
            f"Remove '{path}' from the library list?\nModels on disk are NOT deleted.",
            parent=self._win, icon="warning"):
            return
        s.model_libraries.remove(path)
        save_settings(s)
        self._cur_lib = ""
        self._refresh_tree()
        self._refresh_models_panel()
        self._notify_parent()

    def _move_up(self) -> None:
        self._reorder(-1)

    def _move_down(self) -> None:
        self._reorder(+1)

    def _reorder(self, delta: int) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        path = sel[0]
        libs = self._state.settings.model_libraries
        i = libs.index(path) if path in libs else -1
        j = i + delta
        if i < 0 or j < 0 or j >= len(libs):
            return
        libs[i], libs[j] = libs[j], libs[i]
        save_settings(self._state.settings)
        self._refresh_tree()
        self._tree.selection_set(path)
        self._notify_parent()

    def _open_in_explorer(self, lib_wsl: str) -> None:
        if not lib_wsl:
            return
        unc = self._state.settings.wsl_path_to_unc(lib_wsl)
        if unc:
            subprocess.Popen(["explorer.exe", unc])

    # ── Move ──────────────────────────────────────────────────────────────────
    def _move_library(self) -> None:
        if not self._cur_lib:
            messagebox.showinfo("No library", "Select a library to move.", parent=self._win)
            return
        _, total, sizes = self._scan_lib(self._cur_lib)
        if not sizes:
            messagebox.showinfo("Empty", "That library has no models to move.", parent=self._win)
            return
        dest = self._pick_destination(exclude=self._cur_lib)
        if not dest:
            return
        self._launch_move(list(sizes.keys()), total, self._cur_lib, dest, basenames=None)

    def _move_selected(self) -> None:
        sel = list(self._mtree.selection())
        if not sel:
            messagebox.showinfo("No models", "Select one or more models to move.", parent=self._win)
            return
        dest = self._pick_destination(exclude=self._cur_lib)
        if not dest:
            return
        total = sum(self._cur_sizes.get(p, 0) for p in sel)
        basenames = [p.split("/")[-1] for p in sel]
        self._launch_move(sel, total, self._cur_lib, dest, basenames=basenames)

    def _launch_move(self, srcs, total, old_base, dest, basenames) -> None:
        s = self._state.settings
        # Ensure the destination is a registered library so moved files are
        # scanned afterwards (whole-library remap also handles this, but a new
        # per-model destination must be added here).
        if dest not in s.model_libraries:
            s.model_libraries.append(dest)
            save_settings(s)
            self._refresh_tree()

        from gui.move_dialog import MoveDialog
        MoveDialog(self._root, self._state, self._T,
                   log_fn=self._log, srcs=srcs, total_bytes=total,
                   old_base=old_base, dest_lib=dest, basenames=basenames,
                   on_done=self._after_move).show()

    def _after_move(self) -> None:
        # remap may have removed the (now empty) source library
        if self._cur_lib not in self._state.settings.model_libraries:
            self._cur_lib = ""
        self._refresh_tree()
        self._refresh_models_panel()
        self._notify_parent()

    def _pick_destination(self, exclude: str) -> str | None:
        """Modal chooser: pick an existing library or type a new WSL path."""
        T = self._T
        options = [l for l in self._state.settings.model_libraries if l != exclude]
        win = tk.Toplevel(self._win)
        win.title("Move destination")
        win.geometry("460x220")
        win.configure(bg=T["bg"])
        win.transient(self._win)
        win.grab_set()
        result: dict[str, str | None] = {"v": None}

        tk.Label(win, text="Move to an existing library:", bg=T["bg"], fg=T["fg"],
                 font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(14, 2))
        combo = ttk.Combobox(win, state="readonly", values=options, font=("Segoe UI", 9))
        combo.pack(fill="x", padx=14)
        if options:
            combo.current(0)

        tk.Label(win, text="…or pick a new folder (any drive):", bg=T["bg"],
                 fg=T["fg"], font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(10, 2))
        path_row = tk.Frame(win, bg=T["bg"])
        path_row.pack(fill="x", padx=14)
        entry = tk.Entry(path_row, bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                         font=("Consolas", 9), insertbackground=T["fg"])
        entry.pack(side="left", expand=True, fill="x")

        def _browse():
            start = self._state.settings.wsl_path_to_unc(exclude) or None
            d = filedialog.askdirectory(parent=win, mustexist=False,
                                        title="Pick destination folder",
                                        initialdir=start)
            if not d:
                return
            wsl = self._state.settings.unc_to_wsl(d)
            if wsl:
                entry.delete(0, tk.END)
                entry.insert(0, wsl)
            else:
                messagebox.showwarning(
                    "Unsupported location",
                    f"Couldn't map this folder to a WSL path:\n{d}\n\n"
                    "Pick a drive (e.g. D:\\…) or a \\\\wsl.localhost\\… folder.",
                    parent=win)

        tk.Button(path_row, text="Browse…", bg=T["btn"], fg=T["accent"], relief="flat",
                  cursor="hand2", font=("Segoe UI", 9, "bold"), command=_browse).pack(
                      side="left", padx=(6, 0))

        def _ok():
            new = entry.get().strip()
            result["v"] = new if new else (combo.get().strip() or None)
            win.destroy()

        row = tk.Frame(win, bg=T["bg"])
        row.pack(fill="x", padx=14, pady=14)
        tk.Button(row, text="Move here", bg=T["green"], fg=T["bg"], relief="flat",
                  cursor="hand2", font=("Segoe UI", 9, "bold"), command=_ok).pack(side="left")
        tk.Button(row, text="Cancel", bg=T["btn"], fg=T["btn_fg"], relief="flat",
                  cursor="hand2", font=("Segoe UI", 9), command=win.destroy).pack(side="right")

        win.wait_window()
        dest = result["v"]
        if dest and dest == exclude:
            return None
        return dest

    # ── Delete ────────────────────────────────────────────────────────────────
    def _delete_selected(self) -> None:
        sel = list(self._mtree.selection())
        if not sel:
            return
        total = sum(self._cur_sizes.get(p, 0) for p in sel)
        names = "\n".join(p.split("/")[-1] for p in sel[:8]) + ("\n…" if len(sel) > 8 else "")
        if not messagebox.askyesno(
            "Delete Models",
            f"Permanently delete {len(sel)} model(s)?  {_fmt_size(total)} freed.\n\n{names}\n\n"
            "This cannot be undone.",
            parent=self._win, icon="warning"):
            return
        s = self._state.settings
        import shlex
        abs_paths = [shlex.quote(p.replace("~", f"/home/{s.wsl_user}", 1)) for p in sel]
        try:
            r = subprocess.run(["wsl", "-d", s.wsl_distro, "-u", s.wsl_user,
                                "bash", "-c", "rm -f " + " ".join(abs_paths)],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                messagebox.showerror("Delete failed",
                                     (r.stderr or r.stdout or "Unknown error").strip(),
                                     parent=self._win)
                return
        except Exception as e:
            messagebox.showerror("Delete failed", str(e), parent=self._win)
            return
        # Clear live selection if it was deleted
        if self._state.model_var.get() in sel:
            self._state.model_var.set("")
        self._log(f"[INFO] Deleted {len(sel)} model(s), {_fmt_size(total)} freed.", "warn")
        self._refresh_tree()
        self._refresh_models_panel()
        self._notify_parent()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _log(self, text: str, sev: str | None = None) -> None:
        if self._log_fn:
            self._log_fn(text, sev)

    def _notify_parent(self) -> None:
        if self._on_close:
            try:
                self._on_close()
            except Exception:
                pass

    def _close(self) -> None:
        self._notify_parent()
        if self._win:
            self._win.destroy()
            self._win = None
