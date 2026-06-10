"""
Left panel — model list, server controls, profiles, download button.
"""
from __future__ import annotations
import os
from pathlib import Path
from tkinter import ttk, messagebox, simpledialog
import tkinter as tk
from typing import Callable

from core.settings import BUILTIN_PROFILES, save_profiles
from core.server   import ServerState

LogFn = Callable[[str, str | None], None]


class LeftPanel:

    def __init__(self, root: tk.Tk, state, T: dict, log_fn: LogFn):
        self._root   = root
        self._state  = state
        self._T      = T
        self._log    = log_fn
        self._frame: tk.Frame | None = None
        self._all_models: list[str] = []

    def build(self, parent: tk.Widget) -> None:
        T = self._T
        self._frame = tk.Frame(parent, bg=T["bg2"], width=270)
        self._frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self._frame.pack_propagate(False)

        self._build_models_section()
        _sep(self._frame, T)
        self._build_server_section()
        _sep(self._frame, T)
        self._build_profiles_section()
        _sep(self._frame, T)
        self._build_download_section()

    def update_server_status(self, state: str) -> None:
        try:
            running = state == "running"
            loading = state == "loading"
            self._load_btn.config(
                state="disabled" if loading else "normal"
            )
            self._unload_btn.config(
                state="normal" if (running or loading) else "disabled"
            )
        except Exception:
            pass

    # ── Models ────────────────────────────────────────────────────────────────

    def _build_models_section(self) -> None:
        T = self._T
        _section(self._frame, "MODELS", T)

        # Search box
        search_frame = tk.Frame(self._frame, bg=T["bg2"])
        search_frame.pack(fill="x", padx=8, pady=(0, 4))
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._filter_models())
        search_entry = tk.Entry(
            search_frame, textvariable=self._search_var,
            bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
            font=("Consolas", 9), insertbackground=T["fg"]
        )
        search_entry.pack(fill="x")
        search_entry.insert(0, "")
        # Placeholder behaviour
        search_entry.bind("<FocusIn>",  lambda e: self._on_search_focus_in(search_entry))
        search_entry.bind("<FocusOut>", lambda e: self._on_search_focus_out(search_entry))
        self._search_placeholder = True
        self._show_search_placeholder(search_entry)

        # Model listbox
        list_frame = tk.Frame(self._frame, bg=T["bg2"])
        list_frame.pack(fill="both", expand=True, padx=8, pady=(0, 2))
        sb = tk.Scrollbar(list_frame)
        sb.pack(side="right", fill="y")
        self._model_list = tk.Listbox(
            list_frame, yscrollcommand=sb.set,
            bg=T["bg3"], fg=T["fg"],
            selectbackground=T["select_bg"], selectforeground=T["select_fg"],
            relief="flat", font=("Consolas", 9), activestyle="none",
            borderwidth=0, highlightthickness=0
        )
        self._model_list.pack(fill="both", expand=True)
        sb.config(command=self._model_list.yview)
        self._model_list.bind("<<ListboxSelect>>", self._on_model_select)

        # Size label
        self._size_label = tk.Label(
            self._frame, text="", bg=T["bg2"], fg=T["fg2"],
            font=("Consolas", 8)
        )
        self._size_label.pack(padx=8, anchor="w")

        tk.Button(
            self._frame, text="⟳ Refresh", bg=T["btn"], fg=T["btn_fg"],
            relief="flat", cursor="hand2", font=("Segoe UI", 9),
            command=self.refresh_models
        ).pack(fill="x", padx=8, pady=3)

        self.refresh_models()

    def refresh_models(self) -> None:
        unc = self._state.settings.models_unc
        if not unc:
            self._log("[WARN] Models path not configured — run setup.", "warn")
            return
        try:
            entries = os.listdir(unc)
            self._all_models = sorted(
                f for f in entries if f.lower().endswith(".gguf")
            )
        except Exception as e:
            self._all_models = []
            self._log(f"[WARN] Could not scan models: {e}", "warn")
        self._populate_list(self._all_models)
        self._log(f"[INFO] Found {len(self._all_models)} model(s)", "info")

    def _filter_models(self) -> None:
        if not hasattr(self, "_model_list"):
            return
        q = self._search_var.get().strip().lower()
        if self._search_placeholder or not q:
            self._populate_list(self._all_models)
        else:
            self._populate_list([m for m in self._all_models if q in m.lower()])

    def _populate_list(self, models: list[str]) -> None:
        current = self._state.model_var.get()
        self._model_list.delete(0, tk.END)
        for m in models:
            self._model_list.insert(tk.END, m)
        if current in models:
            idx = models.index(current)
            self._model_list.selection_set(idx)
            self._model_list.see(idx)
        elif models:
            self._model_list.selection_set(0)
            self._state.model_var.set(models[0])
            self._update_size_label(models[0])

    def _on_model_select(self, event=None) -> None:
        sel = self._model_list.curselection()
        if sel:
            model = self._model_list.get(sel[0])
            self._state.model_var.set(model)
            self._update_size_label(model)

    def _update_size_label(self, model: str) -> None:
        try:
            unc  = self._state.settings.models_unc
            size = (Path(unc) / model).stat().st_size
            mb   = size / (1024 * 1024)
            label = f"{mb/1024:.2f} GiB" if mb > 1024 else f"{mb:.0f} MiB"
            self._size_label.config(text=label)
        except Exception:
            self._size_label.config(text="")

    def _show_search_placeholder(self, entry: tk.Entry) -> None:
        entry.config(fg=self._T["fg2"])
        entry.delete(0, tk.END)
        entry.insert(0, "Filter models…")

    def _on_search_focus_in(self, entry: tk.Entry) -> None:
        if self._search_placeholder:
            entry.config(fg=self._T["entry_fg"])
            entry.delete(0, tk.END)
            self._search_placeholder = False

    def _on_search_focus_out(self, entry: tk.Entry) -> None:
        if not self._search_var.get().strip():
            self._search_placeholder = True
            self._show_search_placeholder(entry)

    # ── Server controls ───────────────────────────────────────────────────────

    def _build_server_section(self) -> None:
        T = self._T
        _section(self._frame, "LLAMA SERVER", T)

        self._load_btn = tk.Button(
            self._frame, text="▶  Load Model",
            bg=T["green"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 10, "bold"), pady=6,
            command=self._load_model
        )
        self._load_btn.pack(fill="x", padx=8, pady=3)

        self._unload_btn = tk.Button(
            self._frame, text="■  Unload",
            bg=T["red"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 10, "bold"), pady=6, state="disabled",
            command=self._unload_model
        )
        self._unload_btn.pack(fill="x", padx=8, pady=3)

    def _load_model(self) -> None:
        model = self._state.model_var.get()
        if not model:
            messagebox.showerror("No Model", "Please select a model first.")
            return
        ctrl = self._state.server_ctrl
        if not ctrl:
            self._log("[ERROR] Server controller not ready.", "error")
            return
        cmd = self._state.build_cmd()
        ctrl.start(cmd, model)

    def _unload_model(self) -> None:
        ctrl = self._state.server_ctrl
        if ctrl:
            ctrl.stop()

    # ── Profiles ──────────────────────────────────────────────────────────────

    def _build_profiles_section(self) -> None:
        T = self._T
        _section(self._frame, "PROFILES", T)

        self._profile_combo = ttk.Combobox(
            self._frame, textvariable=self._state.profile_var,
            state="readonly", font=("Segoe UI", 9)
        )
        self._profile_combo.pack(fill="x", padx=8, pady=3)
        self._profile_combo.bind("<<ComboboxSelected>>", self._on_profile_select)
        self._refresh_profile_list()

        btn_row = tk.Frame(self._frame, bg=T["bg2"])
        btn_row.pack(fill="x", padx=8, pady=3)
        tk.Button(
            btn_row, text="Save", bg=T["btn"], fg=T["btn_fg"],
            relief="flat", cursor="hand2", font=("Segoe UI", 9),
            command=self._save_profile
        ).pack(side="left", expand=True, fill="x", padx=(0, 2))
        tk.Button(
            btn_row, text="Delete", bg=T["btn"], fg=T["red"],
            relief="flat", cursor="hand2", font=("Segoe UI", 9),
            command=self._delete_profile
        ).pack(side="left", expand=True, fill="x", padx=(2, 0))

    def _refresh_profile_list(self) -> None:
        names = list(self._state.profiles.keys())
        self._profile_combo["values"] = names
        if self._state.profile_var.get() not in names and names:
            self._state.profile_var.set(names[0])

    def _on_profile_select(self, event=None) -> None:
        name = self._state.profile_var.get()
        if name in self._state.profiles:
            self._state.apply_profile_dict(self._state.profiles[name])
            self._log(f"[PROFILE] Loaded: {name}", "info")

    def _save_profile(self) -> None:
        name = simpledialog.askstring(
            "Save Profile", "Profile name:",
            parent=self._root,
            initialvalue=self._state.profile_var.get()
        )
        if not name:
            return
        self._state.profiles[name] = self._state.get_profile_dict()
        save_profiles(self._state.profiles)
        self._refresh_profile_list()
        self._state.profile_var.set(name)
        self._log(f"[PROFILE] Saved: {name}", "success")

    def _delete_profile(self) -> None:
        name = self._state.profile_var.get()
        if name in BUILTIN_PROFILES:
            messagebox.showwarning(
                "Built-in", f'"{name}" is built-in and cannot be deleted.')
            return
        if name and name in self._state.profiles:
            if messagebox.askyesno("Delete", f'Delete profile "{name}"?'):
                del self._state.profiles[name]
                save_profiles(self._state.profiles)
                self._refresh_profile_list()
                self._log(f"[PROFILE] Deleted: {name}", "warn")

    # ── Download ──────────────────────────────────────────────────────────────

    def _build_download_section(self) -> None:
        T = self._T
        tk.Button(
            self._frame, text="⬇  Download Models",
            bg=T["btn"], fg=T["accent"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9, "bold"), pady=4,
            command=self._open_downloader
        ).pack(fill="x", padx=8, pady=(4, 8))

    def _open_downloader(self) -> None:
        from gui.download_manager import DownloadManager
        DownloadManager(self._root, self._state, self._T,
                        log_fn=self._log,
                        on_complete=self.refresh_models).show()


# ── Widget helpers ────────────────────────────────────────────────────────────

def _section(parent: tk.Widget, text: str, T: dict) -> None:
    tk.Label(parent, text=text, bg=parent.cget("bg"), fg=T["accent"],
             font=("Consolas", 8, "bold")).pack(anchor="w", padx=8, pady=(8, 2))


def _sep(parent: tk.Widget, T: dict) -> None:
    tk.Frame(parent, bg=T["bg3"], height=1).pack(fill="x", padx=8, pady=4)
