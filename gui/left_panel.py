"""
Left panel — model list (Treeview), server controls, profiles, download button.
"""
from __future__ import annotations
import os
import queue
import re
import subprocess
import threading
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


class LeftPanel:

    def __init__(self, root: tk.Tk, state: AppState, T: dict, log_fn: LogFn):
        self._root   = root
        self._state  = state
        self._T      = T
        self._log    = log_fn
        self._frame: tk.Frame | None = None

        # Model data — keyed by full WSL path (e.g. ~/llama.cpp/models/foo.gguf)
        self._all_models:    list[str]       = []
        self._model_meta:    dict[str, dict] = {}
        self._model_sizes:   dict[str, int]  = {}
        self._model_lib_map: dict[str, str]  = {}  # full_wsl_path → lib_wsl_base

    def build(self, frame: tk.Frame) -> None:
        T = self._T
        self._frame = frame

        # Single column: model dropdown on top, then server / profiles / tools.
        # (The old two-column tree|controls split is gone now that the model
        # picker is a compact dropdown instead of a full-height treeview.)
        self._col = tk.Frame(self._frame, bg=T["bg2"])
        self._col.pack(fill="both", expand=True)
        self._tree_col = self._ctrl_col = self._col

        self._build_models_section()
        _sep(self._ctrl_col, T)
        self._build_server_section()
        _sep(self._ctrl_col, T)
        self._build_profiles_section()
        _sep(self._ctrl_col, T)
        self._build_download_section()
        _sep(self._ctrl_col, T)
        self._build_utilities_section()

    def update_server_status(self, state: str) -> None:
        try:
            T = self._T
            if state == "running":
                self._server_btn.config(text="■  Unload", bg=T["red"],
                                        fg=T["bg"], state="normal")
            elif state == "loading":
                self._server_btn.config(text="⏳  Loading…", bg=T["yellow"],
                                        fg=T["bg"], state="disabled")
            else:  # stopped / error / crashed / unknown
                self._server_btn.config(text="▶  Load Model", bg=T["green"],
                                        fg=T["bg"], state="normal")
        except Exception:
            pass

    # ── Models ────────────────────────────────────────────────────────────────

    def _build_models_section(self) -> None:
        T = self._T
        _section(self._col, "MODEL", T)

        # display string ↔ full WSL path
        self._display_to_path: dict[str, str] = {}
        self._all_displays:    list[str]      = []

        # Editable combobox = compact picker + type-ahead filter.
        self._model_combo = ttk.Combobox(
            self._col, state="normal", font=("Segoe UI", 9),
        )
        self._model_combo.pack(fill="x", padx=8, pady=(0, 2))
        self._model_combo.bind("<<ComboboxSelected>>", self._on_combo_select)
        self._model_combo.bind("<KeyRelease>", self._on_combo_key)
        ToolTip(self._model_combo,
                "Pick a model to select it. Type to filter the list.")

        btn_row = tk.Frame(self._col, bg=T["bg2"])
        btn_row.pack(fill="x", padx=8, pady=3)
        self._refresh_btn = tk.Button(
            btn_row, text="⟳ Refresh", bg=T["btn"], fg=T["btn_fg"],
            relief="flat", cursor="hand2", font=("Segoe UI", 9),
            command=self.refresh_models,
        )
        self._refresh_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        tk.Button(
            btn_row, text="⊞ Libraries", bg=T["btn"], fg=T["btn_fg"],
            relief="flat", cursor="hand2", font=("Segoe UI", 9),
            command=self._open_library_manager,
        ).pack(side="left", expand=True, fill="x", padx=(2, 0))

        # Queue-based scan result delivery — safe for background threads on Windows.
        # The background thread puts results here; the main-thread polling loop reads them.
        self._scan_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._scan_id: int = 0
        self._root.after(200, self._drain_scan_results)  # start persistent poll (main thread)

        self.refresh_models()

    # ── Dropdown population / filtering ─────────────────────────────────────────

    def _model_display(self, full_wsl: str) -> str:
        meta  = self._model_meta.get(full_wsl, {})
        fname = full_wsl.split("/")[-1]
        base  = meta.get("display", fname)
        quant = meta.get("quant", "") or "?"
        size  = _fmt_size(self._model_sizes.get(full_wsl, 0))
        return f"{base}  ·  {quant}  ·  {size}"

    def _populate_combo(self, models: list[str] | None = None) -> None:
        models = self._all_models if models is None else models
        ordered = sorted(
            models,
            key=lambda p: self._model_meta.get(p, {})
                              .get("display", p.split("/")[-1]).lower(),
        )

        self._display_to_path = {}
        displays: list[str] = []
        for p in ordered:
            disp = self._model_display(p)
            if disp in self._display_to_path:          # same name+quant+size in 2 libs
                lib = self._model_lib_map.get(p, "")
                disp = f"{disp}  [{lib.split('/')[-1]}]"
            self._display_to_path[disp] = p
            displays.append(disp)

        self._all_displays = displays
        self._model_combo["values"] = displays

        # Restore selection (exact full-path or legacy bare-filename match), else
        # auto-select the first model.
        current = self._state.model_var.get()
        match = next(
            ((d, p) for d, p in self._display_to_path.items()
             if p == current or p.split("/")[-1] == current),
            None,
        )
        if match is None and displays:
            match = (displays[0], self._display_to_path[displays[0]])

        if match:
            disp, path = match
            self._model_combo.set(disp)
            if self._state.model_var.get() != path:
                self._state.model_var.set(path)        # upgrade bare → full path
            size_b = self._model_sizes.get(path, 0)
            self._state.model_size_gb = size_b / 1e9 if size_b else 0.0
        else:
            self._model_combo.set("")

    def _on_combo_key(self, event) -> None:
        if event.keysym in ("Up", "Down", "Return", "Escape", "Tab", "Left", "Right"):
            return
        typed = self._model_combo.get().strip().lower()
        if not typed:
            self._model_combo["values"] = self._all_displays
            return
        hits = [d for d in self._all_displays
                if typed in d.lower()
                or typed in self._display_to_path.get(d, "").split("/")[-1].lower()]
        self._model_combo["values"] = hits or self._all_displays

    def _on_combo_select(self, event=None) -> None:
        path = self._display_to_path.get(self._model_combo.get())
        if path:
            self._state.model_var.set(path)
            size_b = self._model_sizes.get(path, 0)
            self._state.model_size_gb = size_b / 1e9 if size_b else 0.0

    def refresh_models(self) -> None:
        """
        Scan model libraries asynchronously so WSL filesystem latency never
        blocks the main thread. Shows a placeholder row while scanning.
        """
        s    = self._state.settings
        libs = s.all_library_uncs

        if not libs:
            unc = s.models_unc
            if unc:
                libs = [(s.models_wsl, unc)]
            else:
                self._log("[WARN] No model libraries configured — run setup.", "warn")
                return

        # Show placeholder immediately so the UI isn't empty
        try:
            self._model_combo.set("Scanning libraries…")
            self._model_combo["values"] = []
            self._refresh_btn.config(state="disabled")
        except Exception:
            pass

        # Stamp each scan so stale results from a previous scan are discarded
        self._scan_id += 1
        current_id    = self._scan_id
        libs_snapshot = list(libs)   # capture before thread starts

        def _scan() -> None:
            all_models:    list[str]       = []
            model_meta:    dict[str, dict] = {}
            model_sizes:   dict[str, int]  = {}
            model_lib_map: dict[str, str]  = {}

            for lib_wsl, lib_unc in libs_snapshot:
                try:
                    # scandir gives DirEntry objects whose .stat() is cached from
                    # the directory read — one WSL round-trip instead of N+1
                    with os.scandir(lib_unc) as it:
                        for entry in it:
                            if not entry.name.lower().endswith(".gguf"):
                                continue
                            full_wsl = f"{lib_wsl}/{entry.name}"
                            all_models.append(full_wsl)
                            model_meta[full_wsl]    = _parse_gguf(entry.name)
                            model_lib_map[full_wsl] = lib_wsl
                            try:
                                model_sizes[full_wsl] = entry.stat().st_size
                            except Exception:
                                model_sizes[full_wsl] = 0
                except Exception as e:
                    self._log(f"[WARN] Cannot scan {lib_wsl}: {e}", "warn")

            n_libs = len(libs_snapshot)
            # Queue put is thread-safe; _drain_scan_results() delivers on main thread
            self._scan_queue.put(
                (current_id, all_models, model_meta, model_sizes, model_lib_map, n_libs)
            )

        threading.Thread(target=_scan, daemon=True).start()

    def _drain_scan_results(self) -> None:
        """Persistent main-thread polling loop — delivers background scan results safely."""
        try:
            while True:
                scan_id, *args = self._scan_queue.get_nowait()
                if scan_id == self._scan_id:   # ignore results from superseded scans
                    self._on_scan_complete(*args)
        except queue.Empty:
            pass
        self._root.after(200, self._drain_scan_results)  # reschedule unconditionally

    def _on_scan_complete(self, all_models: list[str], model_meta: dict,
                          model_sizes: dict, model_lib_map: dict,
                          n_libs: int) -> None:
        """Called on the main thread once the background scan finishes."""
        self._all_models    = all_models
        self._model_meta    = model_meta
        self._model_sizes   = model_sizes
        self._model_lib_map = model_lib_map

        self._populate_combo()

        self._log(
            f"[INFO] Found {len(all_models)} model(s) across "
            f"{n_libs} librar{'y' if n_libs == 1 else 'ies'}",
            "info",
        )
        try:
            self._refresh_btn.config(state="normal")
        except Exception:
            pass

    def _open_library_manager(self) -> None:
        from gui.library_manager import LibraryManager
        LibraryManager(self._root, self._state, self._T,
                       on_close=self.refresh_models, log_fn=self._log).show()

    # ── Server controls ───────────────────────────────────────────────────────

    def _build_server_section(self) -> None:
        T = self._T
        _section(self._ctrl_col, "LLAMA SERVER", T)

        # One state-aware button: Load → Loading… → Unload (driven by
        # update_server_status). The header shows status text separately.
        self._server_btn = tk.Button(
            self._ctrl_col, text="▶  Load Model",
            bg=T["green"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 10, "bold"), pady=5,
            command=self._toggle_server,
        )
        self._server_btn.pack(fill="x", padx=8, pady=3)

    def _toggle_server(self) -> None:
        ctrl = self._state.server_ctrl
        if ctrl and ctrl.state == ServerState.RUNNING:
            self._unload_model()
        else:
            self._load_model()

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

        from core.settings import CRASH_LOG
        has_crashes = CRASH_LOG.exists() and CRASH_LOG.stat().st_size > 0

        row = tk.Frame(self._ctrl_col, bg=T["bg2"])
        row.pack(fill="x", padx=8, pady=(2, 8))

        def _icon(text, fg, cmd, tip, bg=None):
            b = tk.Button(row, text=text, bg=bg or T["btn"], fg=fg,
                          relief="flat", cursor="hand2",
                          font=("Segoe UI", 12), pady=2, command=cmd)
            b.pack(side="left", expand=True, fill="x", padx=1)
            ToolTip(b, tip)
            return b

        _icon("🌐", T["accent"], self._open_llama_ui,
              "Open the llama.cpp web UI in your browser.")
        _icon("🔍", T["yellow"], self._diagnose,
              "Diagnose — connectivity checks:\n"
              "  • llama-server process running in WSL\n"
              "  • Port 8089 listening\n"
              "  • tool-proxy.py running\n"
              "  • Proxy :8088 reachable\n"
              "  • localhost:8089/health\n"
              "  • WSL IP direct access")
        _icon("♻", T["btn_fg"], self._restart_proxy,
              "Restart the tool-proxy.py process in WSL.\n"
              "Use this if Cline/extensions can't reach the model after a reload,\n"
              "or if the proxy crashed. Reconnects :8088 → :8089.")
        _icon("🎨", T["btn_fg"], self._open_theme_picker, "Change theme.")
        self._crash_btn = _icon(
            "⚠", T["bg"] if has_crashes else T["btn_fg"],
            self._open_crash_log, "View the crash log.",
            bg=T["red"] if has_crashes else T["btn"])

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

    def _open_crash_log(self) -> None:
        from core.settings import CRASH_LOG
        from tkinter import scrolledtext
        T = self._T
        win = tk.Toplevel(self._root)
        win.title("Crash Log")
        win.geometry("720x520")
        win.configure(bg=T["bg"])

        hrow = tk.Frame(win, bg=T["bg"])
        hrow.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(hrow, text="CRASH LOG", bg=T["bg"], fg=T["accent"],
                 font=("Consolas", 8, "bold")).pack(side="left")

        txt = scrolledtext.ScrolledText(
            win, bg=T["log_bg"], fg=T["log_fg"],
            font=("Consolas", 8), relief="flat", wrap="word",
            state="disabled",
        )

        def _clear():
            try:
                CRASH_LOG.write_text("", encoding="utf-8")
                txt.config(state="normal")
                txt.delete("1.0", tk.END)
                txt.insert(tk.END, "(cleared)")
                txt.config(state="disabled")
                self._crash_btn.config(
                    text="⚠", bg=T["btn"], fg=T["btn_fg"])
            except Exception:
                pass

        tk.Button(hrow, text="Clear", bg=T["red"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 8),
                  command=_clear).pack(side="right", padx=2)
        tk.Button(hrow, text="Close", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 8),
                  command=win.destroy).pack(side="right", padx=2)

        txt.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        content = ""
        try:
            if CRASH_LOG.exists():
                content = CRASH_LOG.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        txt.config(state="normal")
        txt.insert(tk.END, content.strip() or "(no crashes recorded)")
        txt.config(state="disabled")
        txt.see(tk.END)

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
