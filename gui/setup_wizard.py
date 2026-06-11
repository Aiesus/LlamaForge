"""
First-run setup wizard — 6 steps.
Step 2 has an "Auto-detect everything" button that probes WSL for existing
builds and pre-fills all subsequent steps. Steps 3-6 auto-run their checks
on entry so you only need to act if something is missing.
"""
from __future__ import annotations
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gui.app import AppState

import core.wsl as wsl
from core.settings import DEFAULT_FORKS, save_settings

STEPS = [
    "WSL Check",
    "Distro & User",
    "llama.cpp",
    "TurboQuant (optional)",
    "Dependencies",
    "Deploy Proxy",
]

# Common paths to probe when auto-detecting existing builds
_LLAMA_CANDIDATES  = ["~/llama.cpp", "~/llama-cpp", "~/llama.cpp-official"]
_TURBO_CANDIDATES  = ["~/llama-turbo", "~/llama.cpp-turbo", "~/llama-tq", "~/turbo"]


class SetupWizard:

    def __init__(self, root: tk.Tk, state: AppState, T: dict, on_complete: Callable):
        self._root        = root
        self._state       = state
        self._T           = T
        self._on_complete = on_complete
        self._step        = 0
        self._win: tk.Toplevel | None = None

    def show(self) -> None:
        T = self._T
        win = tk.Toplevel(self._root)
        win.title("llama-gui — First Run Setup")
        win.configure(bg=T["bg"])
        win.geometry("700x580")
        win.resizable(False, False)
        win.grab_set()
        self._win = win

        # Left step list
        left = tk.Frame(win, bg=T["bg2"], width=155)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="SETUP", bg=T["bg2"], fg=T["accent"],
                 font=("Consolas", 9, "bold")).pack(anchor="w", padx=12, pady=(12, 4))

        self._step_labels: list[tk.Label] = []
        for i, name in enumerate(STEPS):
            lbl = tk.Label(left, text=f"  {i+1}. {name}",
                           bg=T["bg2"], fg=T["fg2"],
                           font=("Segoe UI", 9), anchor="w")
            lbl.pack(fill="x", padx=4, pady=2)
            self._step_labels.append(lbl)

        # Right content area
        right = tk.Frame(win, bg=T["bg"])
        right.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        self._content = tk.Frame(right, bg=T["bg"])
        self._content.pack(fill="both", expand=True)

        # Bottom navigation
        nav = tk.Frame(right, bg=T["bg"])
        nav.pack(fill="x", side="bottom", pady=(8, 0))
        self._back_btn = tk.Button(nav, text="← Back", bg=T["btn"], fg=T["btn_fg"],
                                   relief="flat", cursor="hand2", font=("Segoe UI", 9),
                                   command=self._prev_step)
        self._back_btn.pack(side="left")
        self._next_btn = tk.Button(nav, text="Next →", bg=T["accent"], fg=T["bg"],
                                   relief="flat", cursor="hand2", font=("Segoe UI", 9, "bold"),
                                   command=self._next_step)
        self._next_btn.pack(side="right")

        self._render_step()

    def _render_step(self) -> None:
        T = self._T
        for w in self._content.winfo_children():
            w.destroy()

        for i, lbl in enumerate(self._step_labels):
            if i < self._step:
                lbl.config(fg=T["green"], font=("Segoe UI", 9))
            elif i == self._step:
                lbl.config(fg=T["accent"], font=("Segoe UI", 9, "bold"))
            else:
                lbl.config(fg=T["fg2"], font=("Segoe UI", 9))

        self._back_btn.config(state="normal" if self._step > 0 else "disabled")
        last = self._step == len(STEPS) - 1
        self._next_btn.config(text="Finish" if last else "Next →")

        builders = [
            self._step_wsl_check,
            self._step_distro_user,
            self._step_llama_cpp,
            self._step_turbo,
            self._step_deps,
            self._step_proxy,
        ]
        builders[self._step]()

    def _prev_step(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._render_step()

    def _next_step(self) -> None:
        if self._step < len(STEPS) - 1:
            self._step += 1
            self._render_step()
        else:
            self._finish()

    def _finish(self) -> None:
        s = self._state.settings
        s.setup_done = True
        save_settings(s)
        if self._win:
            self._win.destroy()
        self._on_complete()

    # ── Log helper ─────────────────────────────────────────────────────────────

    def _make_log(self, parent: tk.Widget, height: int = 7) -> scrolledtext.ScrolledText:
        T = self._T
        log = scrolledtext.ScrolledText(
            parent, height=height, bg=T["log_bg"], fg=T["log_fg"],
            font=("Consolas", 8), relief="flat", state="disabled", padx=6, pady=4
        )
        log.pack(fill="both", expand=True, pady=4)
        log.tag_config("ok",   foreground=T["green"])
        log.tag_config("err",  foreground=T["red"])
        log.tag_config("warn", foreground=T["orange"])
        return log

    def _log_to(self, log: scrolledtext.ScrolledText,
                text: str, tag: str | None = None) -> None:
        def _do():
            log.config(state="normal")
            log.insert(tk.END, text + "\n", tag or "")
            log.see(tk.END)
            log.config(state="disabled")
        if self._win:
            self._win.after(0, _do)

    # ── Step 1: WSL check ──────────────────────────────────────────────────────

    def _step_wsl_check(self) -> None:
        T = self._T
        _title(self._content, "Check WSL", T)
        tk.Label(self._content,
                 text="Verify WSL2 is installed and list available distros.",
                 bg=T["bg"], fg=T["fg"], font=("Segoe UI", 9),
                 wraplength=480, justify="left").pack(anchor="w", pady=4)
        log = self._make_log(self._content)

        def _check():
            import subprocess
            try:
                r = subprocess.run(["wsl", "--list", "--verbose"],
                                   capture_output=True, text=True, timeout=10)
                if r.returncode == 0:
                    self._log_to(log, "WSL is installed.", "ok")
                    distros = wsl.list_distros()
                    self._log_to(log, f"Distros: {', '.join(distros) or 'none'}", "ok")
                else:
                    self._log_to(log, "WSL not found or not working.", "err")
            except FileNotFoundError:
                self._log_to(log, "WSL not found. Install from Microsoft Store.", "err")
            except Exception as e:
                self._log_to(log, f"Error: {e}", "err")

        tk.Button(self._content, text="Check WSL", bg=T["accent"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda: threading.Thread(target=_check, daemon=True).start()
                  ).pack(anchor="w", pady=4)

        # Auto-run on entry
        threading.Thread(target=_check, daemon=True).start()

    # ── Step 2: Distro & user (+ auto-detect everything) ──────────────────────

    def _step_distro_user(self) -> None:
        T = self._T
        s = self._state.settings
        _title(self._content, "Distro & User", T)

        g = tk.Frame(self._content, bg=T["bg"])
        g.pack(fill="x", pady=4)
        g.columnconfigure(1, weight=1)

        tk.Label(g, text="Distro:", bg=T["bg"], fg=T["fg2"],
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        distro_var = tk.StringVar(value=s.wsl_distro or "")
        distros    = wsl.list_distros()
        distro_combo = ttk.Combobox(g, textvariable=distro_var, values=distros,
                                    font=("Segoe UI", 9), width=22)
        distro_combo.grid(row=0, column=1, sticky="w", pady=3)
        if distros and not distro_var.get():
            distro_var.set(distros[0])

        tk.Label(g, text="Username:", bg=T["bg"], fg=T["fg2"],
                 font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        user_var = tk.StringVar(value=s.wsl_user or "")
        tk.Entry(g, textvariable=user_var, width=24,
                 bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                 font=("Consolas", 10), insertbackground=T["fg"]
                 ).grid(row=1, column=1, sticky="w", pady=3)

        tk.Label(g, text="llama.cpp path:", bg=T["bg"], fg=T["fg2"],
                 font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        llama_var = tk.StringVar(value=s.llama_root or "~/llama.cpp")
        tk.Entry(g, textvariable=llama_var, width=34,
                 bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                 font=("Consolas", 10), insertbackground=T["fg"]
                 ).grid(row=2, column=1, sticky="ew", pady=3)

        tk.Label(g, text="TurboQuant path:", bg=T["bg"], fg=T["fg2"],
                 font=("Segoe UI", 9)).grid(row=3, column=0, sticky="w", padx=(0, 8), pady=3)
        turbo_var = tk.StringVar(value=s.turbo_root or "~/llama-turbo")
        tk.Entry(g, textvariable=turbo_var, width=34,
                 bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                 font=("Consolas", 10), insertbackground=T["fg"]
                 ).grid(row=3, column=1, sticky="ew", pady=3)

        log = self._make_log(self._content, height=5)

        def _save_all():
            s.wsl_distro = distro_var.get().strip()
            s.wsl_user   = user_var.get().strip()
            s.llama_root = llama_var.get().strip()
            s.turbo_root = turbo_var.get().strip()
            save_settings(s)

        def _auto_detect_all():
            d = distro_var.get().strip() or (distros[0] if distros else "")
            if not d:
                self._log_to(log, "No WSL distro available.", "err")
                return

            distro_var.set(d)
            self._log_to(log, f"Distro: {d}", "ok")

            # Detect user
            u = wsl.detect_user(d)
            if u:
                user_var.set(u)
                self._log_to(log, f"User: {u}", "ok")
            else:
                self._log_to(log, "Could not auto-detect user — enter manually.", "warn")
                u = user_var.get().strip()

            # Probe llama.cpp candidates
            found_llama = False
            for candidate in _LLAMA_CANDIDATES:
                if wsl.binary_exists(d, u, f"{candidate}/build/bin/llama-server"):
                    llama_var.set(candidate)
                    ver = wsl.binary_version(d, u, f"{candidate}/build/bin/llama-server")
                    self._log_to(log, f"Found llama.cpp at {candidate}  ({ver})", "ok")
                    found_llama = True
                    break
            if not found_llama:
                self._log_to(log, f"llama.cpp not found in common paths — defaulting to {llama_var.get()}", "warn")

            # Probe TurboQuant candidates
            found_turbo = False
            for candidate in _TURBO_CANDIDATES:
                if wsl.binary_exists(d, u, f"{candidate}/build/bin/llama-server"):
                    turbo_var.set(candidate)
                    ver = wsl.binary_version(d, u, f"{candidate}/build/bin/llama-server")
                    self._log_to(log, f"Found TurboQuant at {candidate}  ({ver})", "ok")
                    found_turbo = True
                    break
            if not found_turbo:
                self._log_to(log, "TurboQuant not found — you can clone it in the next step.", "warn")

            # Save everything detected
            s.wsl_distro = d
            s.wsl_user   = u
            s.llama_root = llama_var.get().strip()
            s.turbo_root = turbo_var.get().strip()
            save_settings(s)
            self._log_to(log, "Settings saved.", "ok")

        bf = tk.Frame(self._content, bg=T["bg"])
        bf.pack(anchor="w", pady=(2, 0))
        tk.Button(bf, text="Auto-detect everything", bg=T["accent"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9, "bold"),
                  command=lambda: threading.Thread(target=_auto_detect_all, daemon=True).start()
                  ).pack(side="left", padx=(0, 8))
        tk.Button(bf, text="Save", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=_save_all).pack(side="left")

    # ── Step 3: llama.cpp ─────────────────────────────────────────────────────

    def _step_llama_cpp(self) -> None:
        T = self._T
        s = self._state.settings
        _title(self._content, "llama.cpp (Official)", T)
        tk.Label(self._content,
                 text="Checking for an existing llama-server binary…",
                 bg=T["bg"], fg=T["fg2"], font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 4))

        g = tk.Frame(self._content, bg=T["bg"])
        g.pack(fill="x", pady=2)
        tk.Label(g, text="WSL path:", bg=T["bg"], fg=T["fg2"],
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        path_var = tk.StringVar(value=s.llama_root or "~/llama.cpp")
        tk.Entry(g, textvariable=path_var, width=36,
                 bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                 font=("Consolas", 10), insertbackground=T["fg"]
                 ).grid(row=0, column=1, sticky="ew", pady=3)

        log = self._make_log(self._content)

        def _check(p: str | None = None):
            p = (p or path_var.get()).strip()
            s.llama_root = p
            save_settings(s)
            if wsl.binary_exists(s.wsl_distro, s.wsl_user, f"{p}/build/bin/llama-server"):
                ver = wsl.binary_version(s.wsl_distro, s.wsl_user, f"{p}/build/bin/llama-server")
                self._log_to(log, f"Found llama-server at {p}  ({ver})", "ok")
                self._log_to(log, "Ready — click Next to continue.", "ok")
            else:
                self._log_to(log, f"llama-server not found at {p}.", "err")
                self._log_to(log, "Use 'Clone & Build' below or adjust the path and recheck.", "warn")

        def _clone():
            p = path_var.get().strip()
            s.llama_root = p
            save_settings(s)
            wsl.clone_and_build(
                s.wsl_distro, s.wsl_user,
                "https://github.com/ggml-org/llama.cpp", p,
                lambda t, tag=None: self._log_to(log, t, tag), None
            )

        bf = tk.Frame(self._content, bg=T["bg"])
        bf.pack(anchor="w", pady=4)
        tk.Button(bf, text="Re-check", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda: threading.Thread(target=_check, daemon=True).start()
                  ).pack(side="left", padx=(0, 6))
        tk.Button(bf, text="Clone & Build", bg=T["accent"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda: threading.Thread(target=_clone, daemon=True).start()
                  ).pack(side="left")

        # Auto-run on entry
        threading.Thread(target=_check, daemon=True).start()

    # ── Step 4: TurboQuant ────────────────────────────────────────────────────

    def _step_turbo(self) -> None:
        T = self._T
        s = self._state.settings
        _title(self._content, "TurboQuant Fork (Optional)", T)
        tk.Label(self._content,
                 text=("TheTom's fork adds turbo2/3/4 KV cache types (Walsh-Hadamard transform).\n"
                       "Recommended for best quality/VRAM ratio. Will NOT merge to upstream."),
                 bg=T["bg"], fg=T["fg"], font=("Segoe UI", 9),
                 wraplength=480, justify="left").pack(anchor="w", pady=4)

        g = tk.Frame(self._content, bg=T["bg"])
        g.pack(fill="x", pady=2)
        tk.Label(g, text="WSL path:", bg=T["bg"], fg=T["fg2"],
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        turbo_var = tk.StringVar(value=s.turbo_root or "~/llama-turbo")
        tk.Entry(g, textvariable=turbo_var, width=36,
                 bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                 font=("Consolas", 10), insertbackground=T["fg"]
                 ).grid(row=0, column=1, sticky="ew", pady=3)

        log = self._make_log(self._content)

        def _check():
            p = turbo_var.get().strip()
            s.turbo_root = p
            save_settings(s)
            if wsl.binary_exists(s.wsl_distro, s.wsl_user, f"{p}/build/bin/llama-server"):
                ver = wsl.binary_version(s.wsl_distro, s.wsl_user, f"{p}/build/bin/llama-server")
                self._log_to(log, f"Found TurboQuant at {p}  ({ver})", "ok")
                self._log_to(log, "Ready — click Next to continue.", "ok")
            else:
                self._log_to(log, f"Not found at {p}. You can clone it or skip.", "warn")

        def _clone():
            p = turbo_var.get().strip()
            s.turbo_root = p
            save_settings(s)
            wsl.clone_and_build(
                s.wsl_distro, s.wsl_user,
                "https://github.com/TheTom/llama.cpp", p,
                lambda t, tag=None: self._log_to(log, t, tag), None
            )

        bf = tk.Frame(self._content, bg=T["bg"])
        bf.pack(anchor="w", pady=4)
        tk.Button(bf, text="Skip", bg=T["btn"], fg=T["fg2"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda: self._log_to(log, "Skipped — official binary only.", None)
                  ).pack(side="left", padx=(0, 6))
        tk.Button(bf, text="Re-check", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda: threading.Thread(target=_check, daemon=True).start()
                  ).pack(side="left", padx=(0, 6))
        tk.Button(bf, text="Clone & Build", bg=T["accent"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda: threading.Thread(target=_clone, daemon=True).start()
                  ).pack(side="left")

        # Auto-run on entry
        threading.Thread(target=_check, daemon=True).start()

    # ── Step 5: Dependencies ──────────────────────────────────────────────────

    def _step_deps(self) -> None:
        T = self._T
        s = self._state.settings
        _title(self._content, "Dependencies", T)
        tk.Label(self._content,
                 text="Checking required WSL packages…",
                 bg=T["bg"], fg=T["fg2"], font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 4))

        log = self._make_log(self._content)

        def _check_all():
            missing = []
            for dep in wsl.DEPS:
                ok = wsl.check_dep(s.wsl_distro, s.wsl_user, dep)
                tag = "ok" if ok else "err"
                self._log_to(log, f"  {'OK' if ok else 'MISSING':8s} {dep}", tag)
                if not ok:
                    missing.append(dep)
            if missing:
                self._log_to(log, f"\n{len(missing)} missing. Click 'Install missing' to fix.", "warn")
            else:
                self._log_to(log, "\nAll dependencies present.", "ok")

        def _install_missing():
            for dep in wsl.DEPS:
                if not wsl.check_dep(s.wsl_distro, s.wsl_user, dep):
                    self._log_to(log, f"Installing {dep}…", None)
                    wsl.install_dep(s.wsl_distro, dep,
                                    lambda t, tag=None: self._log_to(log, t, tag), None)

        bf = tk.Frame(self._content, bg=T["bg"])
        bf.pack(anchor="w", pady=4)
        tk.Button(bf, text="Re-check", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda: threading.Thread(target=_check_all, daemon=True).start()
                  ).pack(side="left", padx=(0, 6))
        tk.Button(bf, text="Install missing", bg=T["accent"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda: threading.Thread(target=_install_missing, daemon=True).start()
                  ).pack(side="left")

        # Auto-run on entry
        threading.Thread(target=_check_all, daemon=True).start()

    # ── Step 6: Deploy proxy ──────────────────────────────────────────────────

    def _step_proxy(self) -> None:
        T = self._T
        s = self._state.settings
        _title(self._content, "Tool Proxy", T)
        tk.Label(self._content,
                 text=("tool-proxy.py runs in WSL on :8088 and forwards to llama-server on :8089.\n"
                       "It auto-converts text-mode tool calls to native OpenAI format.\n"
                       "Checking whether it's already deployed…"),
                 bg=T["bg"], fg=T["fg"], font=("Segoe UI", 9),
                 wraplength=480, justify="left").pack(anchor="w", pady=4)

        log = self._make_log(self._content)

        def _check_existing():
            already = wsl.proxy_running(s.wsl_distro, s.wsl_user)
            exists  = wsl.run(s.wsl_distro, s.wsl_user,
                              "test -f ~/tool-proxy.py && echo yes").stdout.strip() == "yes"
            if exists:
                self._log_to(log, "tool-proxy.py already deployed in WSL.", "ok")
                if already:
                    self._log_to(log, "Proxy is currently running.", "ok")
                else:
                    self._log_to(log, "Proxy not running — it will start automatically when a model loads.", "warn")
                self._log_to(log, "You can click Finish — no action needed.", "ok")
                s.proxy_enabled = True
                save_settings(s)
            else:
                self._log_to(log, "Proxy not yet deployed. Click 'Deploy' below.", "warn")

        def _deploy():
            import sys
            from pathlib import Path
            if getattr(sys, "frozen", False):
                proxy_src = Path(sys._MEIPASS) / "tool_proxy.py"
            else:
                proxy_src = Path(__file__).parent.parent / "tool_proxy.py"
            ok = wsl.deploy_proxy(s.wsl_distro, s.wsl_user, str(proxy_src),
                                  lambda t, tag=None: self._log_to(log, t, tag))
            if ok:
                self._log_to(log, "Proxy deployed successfully.", "ok")
                s.proxy_enabled = True
                save_settings(s)
            else:
                self._log_to(log, "Deploy failed — check log above.", "err")

        bf = tk.Frame(self._content, bg=T["bg"])
        bf.pack(anchor="w", pady=4)
        tk.Button(bf, text="Skip proxy", bg=T["btn"], fg=T["fg2"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda: self._log_to(log, "Skipped.", None)
                  ).pack(side="left", padx=(0, 6))
        tk.Button(bf, text="Deploy / Re-deploy", bg=T["accent"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda: threading.Thread(target=_deploy, daemon=True).start()
                  ).pack(side="left")

        # Auto-run on entry
        threading.Thread(target=_check_existing, daemon=True).start()


# ── Widget helpers ─────────────────────────────────────────────────────────────

def _title(parent: tk.Widget, text: str, T: dict) -> None:
    tk.Label(parent, text=text, bg=T["bg"], fg=T["accent"],
             font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 6))
