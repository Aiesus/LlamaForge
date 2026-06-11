"""Agents tab — list, start/stop/open, add/remove, config editor."""
from __future__ import annotations
import json
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gui.app import AppState

import core.agents as agents_core
from core.settings import save_agents

LogFn = Callable[[str, str | None], None]


class AgentsTab:

    def __init__(self, frame: tk.Frame, state: AppState, T: dict, log_fn: LogFn):
        self._frame = frame
        self._state = state
        self._T     = T
        self._log   = log_fn
        # proc handles indexed by agent name
        self._procs: dict[str, subprocess.Popen | None] = {}

    def build(self) -> None:
        T   = self._T
        top = tk.Frame(self._frame, bg=T["bg2"])
        top.pack(fill="both", expand=True, padx=8, pady=8)

        # Header row
        hrow = tk.Frame(top, bg=T["bg2"])
        hrow.pack(fill="x", pady=(0, 6))
        tk.Label(hrow, text="AGENTS", bg=T["bg2"], fg=T["accent"],
                 font=("Consolas", 8, "bold")).pack(side="left")
        tk.Button(hrow, text="+ Add", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=self._add_agent).pack(side="right", padx=2)
        tk.Button(hrow, text="- Remove", bg=T["btn"], fg=T["red"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=self._remove_agent).pack(side="right", padx=2)

        # Agent list
        self._agent_frame = tk.Frame(top, bg=T["bg3"])
        self._agent_frame.pack(fill="both", expand=True)

        self._refresh_list()

    # ── List rendering ─────────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        T = self._T
        for w in self._agent_frame.winfo_children():
            w.destroy()

        for i, agent in enumerate(self._state.agents):
            self._build_agent_row(self._agent_frame, agent, i)

    def _build_agent_row(self, parent: tk.Frame, agent: dict, idx: int) -> None:
        T    = self._T
        name = agent.get("name", "unnamed")
        proc = self._procs.get(name)

        running = agents_core.is_running(proc)
        pill_bg = T["green"] if running else T["red"]
        pill_tx = "RUNNING" if running else "STOPPED"

        row = tk.Frame(parent, bg=T["bg3"])
        row.pack(fill="x", padx=6, pady=3)

        # Status pill
        tk.Label(row, text=f"  {pill_tx}  ", bg=pill_bg, fg=T["bg"],
                 font=("Consolas", 8, "bold")).pack(side="left", padx=(0, 6))

        # Agent name
        tk.Label(row, text=name, bg=T["bg3"], fg=T["fg"],
                 font=("Segoe UI", 10, "bold"), width=18, anchor="w"
                 ).pack(side="left")

        # Type badge
        tk.Label(row, text=agent.get("type", ""), bg=T["bg3"], fg=T["fg2"],
                 font=("Consolas", 8)).pack(side="left", padx=4)

        # Buttons
        bf = tk.Frame(row, bg=T["bg3"])
        bf.pack(side="right", padx=4)

        tk.Button(bf, text="Start", bg=T["green"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda a=agent: self._start_agent(a)
                  ).pack(side="left", padx=2)
        tk.Button(bf, text="Stop", bg=T["red"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda a=agent: self._stop_agent(a)
                  ).pack(side="left", padx=2)
        tk.Button(bf, text="Edit", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=lambda a=agent, i=idx: self._edit_agent(a, i)
                  ).pack(side="left", padx=2)
        if agent.get("ui_url"):
            tk.Button(bf, text="🌐", bg=T["btn"], fg=T["accent"],
                      relief="flat", cursor="hand2", font=("Segoe UI", 9),
                      command=lambda a=agent: self._open_ui(a)
                      ).pack(side="left", padx=2)

    # ── Actions ────────────────────────────────────────────────────────────────

    def _start_agent(self, agent: dict) -> None:
        name = agent.get("name", "")
        proc = agents_core.start(agent, self._log)
        self._procs[name] = proc
        self._refresh_list()

    def _stop_agent(self, agent: dict) -> None:
        name = agent.get("name", "")
        proc = self._procs.get(name)
        agents_core.stop(proc, name, self._log)
        self._procs[name] = None
        self._refresh_list()

    def _open_ui(self, agent: dict) -> None:
        import webbrowser
        url = agent.get("ui_url", "")
        if url:
            webbrowser.open(url)

    def _add_agent(self) -> None:
        name = simpledialog.askstring("Add Agent", "Agent name:", parent=self._frame)
        if not name:
            return
        new_agent = {
            "name":             name,
            "type":             "generic",
            "exe":              "",
            "config":           "",
            "ui_url":           "",
            "enabled":          True,
            "auto_sync_model":  False,
        }
        self._state.agents.append(new_agent)
        save_agents(self._state.agents)
        self._edit_agent(new_agent, len(self._state.agents) - 1)
        self._refresh_list()

    def _remove_agent(self) -> None:
        if not self._state.agents:
            return
        names = [a.get("name", f"#{i}") for i, a in enumerate(self._state.agents)]
        name  = simpledialog.askstring(
            "Remove Agent",
            "Enter agent name to remove:\n" + "\n".join(f"  {n}" for n in names),
            parent=self._frame
        )
        if not name:
            return
        for i, a in enumerate(self._state.agents):
            if a.get("name") == name:
                if messagebox.askyesno("Remove", f'Remove agent "{name}"?'):
                    del self._state.agents[i]
                    save_agents(self._state.agents)
                    self._refresh_list()
                return
        messagebox.showwarning("Not found", f'No agent named "{name}".')

    def _edit_agent(self, agent: dict, idx: int) -> None:
        _AgentEditor(self._frame, agent, idx, self._T,
                     on_save=lambda: (save_agents(self._state.agents),
                                      self._refresh_list()))


# ── Agent config editor ────────────────────────────────────────────────────────

class _AgentEditor(tk.Toplevel):

    def __init__(self, parent, agent: dict, idx: int, T: dict,
                 on_save: Callable):
        super().__init__(parent)
        self.title(f"Edit Agent — {agent.get('name', '')}")
        self.configure(bg=T["bg"])
        self.resizable(True, True)
        self.attributes("-topmost", True)

        self._agent  = agent
        self._idx    = idx
        self._T      = T
        self._on_save = on_save

        self._build(T)

    def _build(self, T: dict) -> None:
        a = self._agent
        pad = dict(padx=12, pady=4)

        fields = [
            ("Name",           "name"),
            ("Type",           "type"),
            ("Executable",     "exe"),
            ("Config path",    "config"),
            ("UI URL",         "ui_url"),
        ]
        self._vars: dict[str, tk.StringVar] = {}
        for row, (label, key) in enumerate(fields):
            tk.Label(self, text=label, bg=T["bg"], fg=T["fg2"],
                     font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w", **pad)
            var = tk.StringVar(value=str(a.get(key, "")))
            self._vars[key] = var
            tk.Entry(self, textvariable=var, width=46,
                     bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                     font=("Consolas", 9), insertbackground=T["fg"]
                     ).grid(row=row, column=1, sticky="ew", **pad)

        n_rows = len(fields)
        self._enabled_var   = tk.BooleanVar(value=a.get("enabled", True))
        self._sync_var      = tk.BooleanVar(value=a.get("auto_sync_model", False))
        tk.Checkbutton(self, text="Enabled", variable=self._enabled_var,
                       bg=T["bg"], fg=T["fg"], selectcolor=T["bg3"],
                       activebackground=T["bg"], font=("Segoe UI", 9)
                       ).grid(row=n_rows, column=1, sticky="w", padx=12, pady=2)
        tk.Checkbutton(self, text="Auto-sync model on server ready",
                       variable=self._sync_var,
                       bg=T["bg"], fg=T["fg"], selectcolor=T["bg3"],
                       activebackground=T["bg"], font=("Segoe UI", 9)
                       ).grid(row=n_rows + 1, column=1, sticky="w", padx=12, pady=2)

        bf = tk.Frame(self, bg=T["bg"])
        bf.grid(row=n_rows + 2, column=0, columnspan=2, pady=8)
        tk.Button(bf, text="Save", bg=T["green"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9, "bold"),
                  command=self._save).pack(side="left", padx=6)
        tk.Button(bf, text="Cancel", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=self.destroy).pack(side="left", padx=6)

        self.columnconfigure(1, weight=1)
        self.grab_set()

    def _save(self) -> None:
        for key, var in self._vars.items():
            self._agent[key] = var.get()
        self._agent["enabled"]         = self._enabled_var.get()
        self._agent["auto_sync_model"] = self._sync_var.get()
        self._on_save()
        self.destroy()
