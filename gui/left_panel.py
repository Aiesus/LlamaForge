"""
Left panel — model list (Treeview), server controls, profiles, download button.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from tkinter import ttk, messagebox, simpledialog
import tkinter as tk
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gui.app import AppState

from core.settings import BUILTIN_PROFILES, save_profiles
from core.server   import ServerState
from gui.widgets   import ToolTip

LogFn = Callable[[str, str | None], None]

# ── Filename parser ───────────────────────────────────────────────────────────

_QUANT_RE  = re.compile(
    r'(?:[-_.])(IQ[1-5]_[A-Za-z0-9_]+|Q[2-9]_[A-Za-z0-9_]+|BF16|F16|F32)(?=[-_.]|$)',
    re.IGNORECASE,
)
_PARAMS_RE = re.compile(
    r'(?:^|[-_\.])(\d+(?:\.\d+)?)[Bb](?=[-_\.]|$)',
)
_MOE_RE    = re.compile(r'[-_]A\d+(?:\.\d+)?[Bb](?=[-_]|$)|(?:MoE|mixture)', re.IGNORECASE)


def _parse_gguf(filename: str) -> dict:
    """Extract display metadata from a GGUF filename."""
    stem = filename[:-5] if filename.lower().endswith(".gguf") else filename

    # Quant — find last match so model-name tokens don't shadow it
    quant     = ""
    quant_pos = len(stem)
    for m in _QUANT_RE.finditer(stem):
        quant     = m.group(1).upper()
        quant_pos = m.start()

    # Clean name = everything before the quant delimiter, minus -GGUF tag
    clean = stem[:quant_pos].rstrip("-_.")
    clean = re.sub(r"[-_]?GGUF$", "", clean, flags=re.IGNORECASE).rstrip("-_.")

    # MoE detection
    moe = bool(_MOE_RE.search(stem[:quant_pos]))

    # Params — largest B-suffixed number in the pre-quant stem
    params_str = ""
    params_b   = 0.0
    for m in _PARAMS_RE.finditer(stem[:quant_pos]):
        try:
            v = float(m.group(1))
            if v >= 0.4:          # skip version numbers like 0.2
                params_str = f"{m.group(1)}B"
                params_b   = v
        except ValueError:
            pass

    display = clean + (" [M]" if moe else "")

    # Quant tier for row colour
    q = quant.lower()
    if   quant.upper() in ("F16", "BF16", "F32") or q.startswith("q8"):
        tier = "lossless"
    elif q.startswith(("q6", "q5")):
        tier = "high"
    elif q.startswith(("q4", "iq4")):
        tier = "balanced"
    elif q.startswith(("q3", "iq3")):
        tier = "low"
    elif q:
        tier = "vlow"
    else:
        tier = "balanced"

    return {
        "display":    display,
        "params_str": params_str,
        "params_b":   params_b,
        "quant":      quant,
        "tier":       tier,
    }


def _fmt_size(size_b: int) -> str:
    if size_b <= 0:
        return "—"
    gb = size_b / 1024 ** 3
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    return f"{size_b / 1024**2:.0f} MB"


# ── Column header labels (base text without sort arrow) ───────────────────────

_HDR = {"name": "Name", "size": "Size", "params": "Params", "quant": "Quant"}


class LeftPanel:

    def __init__(self, root: tk.Tk, state: AppState, T: dict, log_fn: LogFn):
        self._root   = root
        self._state  = state
        self._T      = T
        self._log    = log_fn
        self._frame: tk.Frame | None = None

        # Model data
        self._all_models:  list[str]       = []
        self._model_meta:  dict[str, dict] = {}
        self._model_sizes: dict[str, int]  = {}

        # Sort state — default: largest files first
        self._sort_col = "size"
        self._sort_rev = True

    def build(self, frame: tk.Frame) -> None:
        T = self._T
        self._frame = frame
        self._frame.columnconfigure(0, weight=1)
        self._frame.rowconfigure(0, weight=1)

        # Left column: model treeview (expands to fill)
        self._tree_col = tk.Frame(self._frame, bg=T["bg2"])
        self._tree_col.grid(row=0, column=0, sticky="nsew")

        # Vertical separator
        tk.Frame(self._frame, bg=T["bg3"], width=1).grid(row=0, column=1, sticky="ns")

        # Right column: server controls, profiles, download (fixed width from content)
        self._ctrl_col = tk.Frame(self._frame, bg=T["bg2"])
        self._ctrl_col.grid(row=0, column=2, sticky="nsew")

        self._build_models_section()
        self._build_server_section()
        _sep(self._ctrl_col, T)
        self._build_profiles_section()
        _sep(self._ctrl_col, T)
        self._build_download_section()
        _sep(self._ctrl_col, T)
        self._build_utilities_section()

    def update_server_status(self, state: str) -> None:
        try:
            running = state == "running"
            loading = state == "loading"
            self._load_btn.config(state="disabled" if loading else "normal")
            self._unload_btn.config(state="normal" if (running or loading) else "disabled")
        except Exception:
            pass

    # ── Models ────────────────────────────────────────────────────────────────

    def _build_models_section(self) -> None:
        T = self._T
        _section(self._tree_col, "MODELS", T)

        # Search box
        search_frame = tk.Frame(self._tree_col, bg=T["bg2"])
        search_frame.pack(fill="x", padx=8, pady=(0, 4))
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._filter_models())
        self._search_entry = tk.Entry(
            search_frame, textvariable=self._search_var,
            bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
            font=("Consolas", 9), insertbackground=T["fg"],
        )
        self._search_entry.pack(fill="x")
        self._search_entry.bind("<FocusIn>",  lambda e: self._on_search_focus_in())
        self._search_entry.bind("<FocusOut>", lambda e: self._on_search_focus_out())
        self._search_placeholder = True
        self._show_search_placeholder()

        # Treeview + scrollbar
        self._apply_tree_style(T)
        tree_frame = tk.Frame(self._tree_col, bg=T["bg2"])
        tree_frame.pack(fill="both", expand=True, padx=8, pady=(0, 2))

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        vsb.pack(side="right", fill="y")

        self._model_tree = ttk.Treeview(
            tree_frame,
            columns=("name", "size", "params", "quant"),
            show="headings",
            selectmode="browse",
            style="Models.Treeview",
            yscrollcommand=vsb.set,
        )
        vsb.config(command=self._model_tree.yview)
        self._model_tree.pack(fill="both", expand=True)

        # Column geometry
        self._model_tree.column("name",   width=172, minwidth=100, stretch=True,  anchor="w")
        self._model_tree.column("size",   width=68,  minwidth=55,  stretch=False, anchor="e")
        self._model_tree.column("params", width=50,  minwidth=40,  stretch=False, anchor="center")
        self._model_tree.column("quant",  width=74,  minwidth=60,  stretch=False, anchor="center")

        # Headings with sort callbacks
        for col in ("name", "size", "params", "quant"):
            arrow = (" ▼" if self._sort_rev else " ▲") if col == self._sort_col else ""
            self._model_tree.heading(
                col, text=_HDR[col] + arrow,
                command=lambda c=col: self._sort_by(c),
            )

        # Tier colour tags (foreground only — selection background still works)
        self._model_tree.tag_configure("lossless", foreground=T["accent"])
        self._model_tree.tag_configure("high",     foreground=T["green"])
        self._model_tree.tag_configure("balanced", foreground=T["fg"])
        self._model_tree.tag_configure("low",      foreground=T["yellow"])
        self._model_tree.tag_configure("vlow",     foreground=T["orange"])

        self._model_tree.bind("<<TreeviewSelect>>", self._on_model_select)

        tk.Button(
            self._tree_col, text="⟳ Refresh", bg=T["btn"], fg=T["btn_fg"],
            relief="flat", cursor="hand2", font=("Segoe UI", 9),
            command=self.refresh_models,
        ).pack(fill="x", padx=8, pady=3)

        self.refresh_models()

    def _apply_tree_style(self, T: dict) -> None:
        style = ttk.Style()
        style.configure("Models.Treeview",
                        background=T["bg3"],
                        fieldbackground=T["bg3"],
                        foreground=T["fg"],
                        rowheight=22,
                        font=("Consolas", 9),
                        borderwidth=0,
                        relief="flat")
        style.configure("Models.Treeview.Heading",
                        background=T["bg2"],
                        foreground=T["fg2"],
                        font=("Segoe UI", 8, "bold"),
                        relief="flat",
                        borderwidth=1)
        style.map("Models.Treeview",
                  background=[("selected", T["select_bg"])],
                  foreground=[("selected", T["select_fg"])])
        style.map("Models.Treeview.Heading",
                  background=[("active", T["bg3"])],
                  foreground=[("active", T["accent"])])

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

        self._model_meta  = {f: _parse_gguf(f) for f in self._all_models}
        self._model_sizes = {}
        for fname in self._all_models:
            try:
                self._model_sizes[fname] = (Path(unc) / fname).stat().st_size
            except Exception:
                self._model_sizes[fname] = 0

        self._populate_tree(self._all_models)
        self._log(f"[INFO] Found {len(self._all_models)} model(s)", "info")

    def _sort_by(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = False
        # Update heading arrows
        for c in ("name", "size", "params", "quant"):
            arrow = (" ▼" if self._sort_rev else " ▲") if c == self._sort_col else ""
            self._model_tree.heading(c, text=_HDR[c] + arrow)
        self._filter_models()

    def _sort_key(self, fname: str):
        m = self._model_meta.get(fname, {})
        if self._sort_col == "name":
            return m.get("display", fname).lower()
        if self._sort_col == "size":
            return self._model_sizes.get(fname, 0)
        if self._sort_col == "params":
            return m.get("params_b", 0.0)
        if self._sort_col == "quant":
            return {"lossless": 0, "high": 1, "balanced": 2, "low": 3, "vlow": 4}.get(
                m.get("tier", "balanced"), 2)
        return fname.lower()

    def _filter_models(self) -> None:
        if not hasattr(self, "_model_tree"):
            return
        q = self._search_var.get().strip().lower()
        if self._search_placeholder or not q:
            self._populate_tree(self._all_models)
        else:
            hits = [
                f for f in self._all_models
                if q in f.lower()
                or q in self._model_meta.get(f, {}).get("display", "").lower()
                or q in self._model_meta.get(f, {}).get("quant", "").lower()
            ]
            self._populate_tree(hits)

    def _populate_tree(self, models: list[str]) -> None:
        current = self._state.model_var.get()
        sorted_models = sorted(models, key=self._sort_key, reverse=self._sort_rev)

        self._model_tree.delete(*self._model_tree.get_children())

        select_iid = None
        for fname in sorted_models:
            meta   = self._model_meta.get(fname, {})
            size_b = self._model_sizes.get(fname, 0)
            self._model_tree.insert(
                "", tk.END,
                iid=fname,
                values=(
                    meta.get("display", fname),
                    _fmt_size(size_b),
                    meta.get("params_str", ""),
                    meta.get("quant", ""),
                ),
                tags=(meta.get("tier", "balanced"),),
            )
            if fname == current:
                select_iid = fname

        if select_iid:
            self._model_tree.selection_set(select_iid)
            self._model_tree.see(select_iid)
        elif sorted_models:
            first = sorted_models[0]
            self._model_tree.selection_set(first)
            self._state.model_var.set(first)

    def _on_model_select(self, event=None) -> None:
        sel = self._model_tree.selection()
        if sel:
            self._state.model_var.set(sel[0])  # iid IS the filename

    # ── Search placeholder ────────────────────────────────────────────────────

    def _show_search_placeholder(self) -> None:
        self._search_entry.config(fg=self._T["fg2"])
        self._search_entry.delete(0, tk.END)
        self._search_entry.insert(0, "Filter models…")

    def _on_search_focus_in(self) -> None:
        if self._search_placeholder:
            self._search_entry.config(fg=self._T["entry_fg"])
            self._search_entry.delete(0, tk.END)
            self._search_placeholder = False

    def _on_search_focus_out(self) -> None:
        if not self._search_var.get().strip():
            self._search_placeholder = True
            self._show_search_placeholder()

    # ── Server controls ───────────────────────────────────────────────────────

    def _build_server_section(self) -> None:
        T = self._T
        _section(self._ctrl_col, "LLAMA SERVER", T)

        self._load_btn = tk.Button(
            self._ctrl_col, text="▶  Load Model",
            bg=T["green"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 10, "bold"), pady=6,
            command=self._load_model,
        )
        self._load_btn.pack(fill="x", padx=8, pady=3)

        self._unload_btn = tk.Button(
            self._ctrl_col, text="■  Unload",
            bg=T["red"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 10, "bold"), pady=6, state="disabled",
            command=self._unload_model,
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
        _section(self._ctrl_col, "PROFILES", T)

        self._profile_combo = ttk.Combobox(
            self._ctrl_col, textvariable=self._state.profile_var,
            state="readonly", font=("Segoe UI", 9), width=18,
        )
        self._profile_combo.pack(fill="x", padx=8, pady=3)
        self._profile_combo.bind("<<ComboboxSelected>>", self._on_profile_select)
        self._refresh_profile_list()

        btn_row = tk.Frame(self._ctrl_col, bg=T["bg2"])
        btn_row.pack(fill="x", padx=8, pady=3)
        tk.Button(
            btn_row, text="Save", bg=T["btn"], fg=T["btn_fg"],
            relief="flat", cursor="hand2", font=("Segoe UI", 9),
            command=self._save_profile,
        ).pack(side="left", expand=True, fill="x", padx=(0, 2))
        tk.Button(
            btn_row, text="Delete", bg=T["btn"], fg=T["red"],
            relief="flat", cursor="hand2", font=("Segoe UI", 9),
            command=self._delete_profile,
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
            initialvalue=self._state.profile_var.get(),
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
            messagebox.showwarning("Built-in",
                                   f'"{name}" is built-in and cannot be deleted.')
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
            self._ctrl_col, text="⬇  Download Models",
            bg=T["btn"], fg=T["accent"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9, "bold"), pady=4,
            command=self._open_downloader,
        ).pack(fill="x", padx=8, pady=(4, 8))

    def _open_downloader(self) -> None:
        from gui.download_manager import DownloadManager
        DownloadManager(self._root, self._state, self._T,
                        log_fn=self._log,
                        on_complete=self.refresh_models).show()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _build_utilities_section(self) -> None:
        T = self._T
        _section(self._ctrl_col, "TOOLS", T)
        tk.Button(
            self._ctrl_col, text="🌐 llama UI",
            bg=T["accent"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9, "bold"), pady=3,
            command=self._open_llama_ui,
        ).pack(fill="x", padx=8, pady=2)
        proxy_btn = tk.Button(
            self._ctrl_col, text="Restart Proxy",
            bg=T["btn"], fg=T["btn_fg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9), pady=3,
            command=self._restart_proxy,
        )
        proxy_btn.pack(fill="x", padx=8, pady=2)
        ToolTip(proxy_btn,
            "Restart the tool-proxy.py process in WSL.\n"
            "Use this if Cline/extensions can't reach the model after a reload,\n"
            "or if the proxy crashed. Reconnects :8088 → :8089.")

        diag_btn = tk.Button(
            self._ctrl_col, text="Diagnose",
            bg=T["yellow"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9, "bold"), pady=3,
            command=self._diagnose,
        )
        diag_btn.pack(fill="x", padx=8, pady=2)
        ToolTip(diag_btn,
            "Run connectivity checks and log results:\n"
            "  • llama-server process running in WSL\n"
            "  • Port 8089 listening\n"
            "  • tool-proxy.py running\n"
            "  • Proxy :8088 reachable\n"
            "  • localhost:8089/health\n"
            "  • WSL IP direct access")
        tk.Button(
            self._ctrl_col, text="Theme",
            bg=T["btn"], fg=T["btn_fg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9), pady=3,
            command=self._open_theme_picker,
        ).pack(fill="x", padx=8, pady=(2, 8))

    def _open_llama_ui(self) -> None:
        import webbrowser
        port    = self._state.port_var.get()
        api_key = self._state.api_key_server_var.get().strip()

        # Pass key in URL fragment — newer llama.cpp builds read it from there.
        # Also copy to clipboard as a fallback for older builds that prompt manually.
        if api_key:
            url = f"http://localhost:{port}/#api_key={api_key}"
            self._root.clipboard_clear()
            self._root.clipboard_append(api_key)
            self._log("[INFO] API key copied to clipboard — paste if the UI prompts.", "info")
        else:
            url = f"http://localhost:{port}/"

        webbrowser.open(url)

    def _restart_proxy(self) -> None:
        from core.wsl import restart_proxy
        s = self._state.settings
        restart_proxy(s.wsl_distro, s.wsl_user, self._log)

    def _diagnose(self) -> None:
        from core.server import run_diagnostics
        s = self._state.settings
        run_diagnostics(s.wsl_distro, s.wsl_user, self._state.port_var.get(), self._log)

    def _open_theme_picker(self) -> None:
        from gui.themes import THEME_LABELS
        from core.settings import save_settings
        T = self._T
        win = tk.Toplevel(self._root)
        win.title("Choose Theme")
        win.geometry("320x260")
        win.configure(bg=T["bg"])
        win.resizable(False, False)
        tk.Label(win, text="Select theme:", bg=T["bg"], fg=T["fg"],
                 font=("Segoe UI", 10)).pack(pady=(16, 8))
        var = tk.StringVar(value=self._state.settings.theme)
        for key, label in THEME_LABELS.items():
            tk.Radiobutton(
                win, text=label, variable=var, value=key,
                bg=T["bg"], fg=T["fg"], selectcolor=T["bg3"],
                activebackground=T["bg"], activeforeground=T["accent"],
                font=("Segoe UI", 9), cursor="hand2"
            ).pack(anchor="w", padx=24, pady=2)
        def _apply():
            chosen = var.get()
            self._state.settings.theme = chosen
            save_settings(self._state.settings)
            messagebox.showinfo(
                "Theme saved",
                f"Theme set to '{THEME_LABELS[chosen]}'.\nRestart the app to apply.",
                parent=win,
            )
            win.destroy()
        btn_row = tk.Frame(win, bg=T["bg"])
        btn_row.pack(pady=12)
        tk.Button(btn_row, text="Apply", bg=T["green"], fg=T["bg"],
                  relief="flat", font=("Segoe UI", 10, "bold"), cursor="hand2",
                  command=_apply).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", font=("Segoe UI", 10), cursor="hand2",
                  command=win.destroy).pack(side="left")


# ── Widget helpers ────────────────────────────────────────────────────────────

def _section(parent: tk.Widget, text: str, T: dict) -> None:
    tk.Label(parent, text=text, bg=parent.cget("bg"), fg=T["accent"],
             font=("Consolas", 8, "bold")).pack(anchor="w", padx=8, pady=(8, 2))


def _sep(parent: tk.Widget, T: dict) -> None:
    tk.Frame(parent, bg=T["bg3"], height=1).pack(fill="x", padx=8, pady=4)
