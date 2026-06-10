"""
Shared widget helpers used across all tabs.
All functions are stateless and receive parent + T (theme dict).
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk


# ── Layout helpers ────────────────────────────────────────────────────────────

def section(parent: tk.Widget, text: str, T: dict) -> tk.Label:
    lbl = tk.Label(parent, text=text, bg=parent.cget("bg"), fg=T["accent"],
                   font=("Consolas", 8, "bold"))
    lbl.pack(anchor="w", padx=8, pady=(8, 2))
    return lbl


def sep(parent: tk.Widget, T: dict) -> None:
    tk.Frame(parent, bg=T["bg3"], height=1).pack(fill="x", padx=8, pady=4)


def grid_frame(parent: tk.Widget) -> tk.Frame:
    g = tk.Frame(parent, bg=parent.cget("bg"))
    g.pack(fill="x", padx=12, pady=4)
    g.columnconfigure(1, weight=1)
    return g


# ── Input widgets ─────────────────────────────────────────────────────────────

def entry(parent: tk.Frame, row: int, label: str,
          var: tk.Variable, width: int = 14, tip: str = "") -> tk.Entry:
    T = _T(parent)
    tk.Label(parent, text=label, bg=parent.cget("bg"), fg=T["fg2"],
             font=("Segoe UI", 9), anchor="w").grid(
                 row=row, column=0, sticky="w", padx=(0, 8), pady=2)
    w = tk.Entry(parent, textvariable=var, width=width,
                 bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                 font=("Consolas", 10), insertbackground=T["fg"])
    w.grid(row=row, column=1, sticky="w", pady=2)
    if tip:
        ToolTip(w, tip)
    return w


def spinbox(parent: tk.Frame, row: int, label: str,
            var: tk.Variable, lo: int, hi: int,
            inc: int = 1, width: int = 8, tip: str = "") -> tk.Spinbox:
    T = _T(parent)
    tk.Label(parent, text=label, bg=parent.cget("bg"), fg=T["fg2"],
             font=("Segoe UI", 9), anchor="w").grid(
                 row=row, column=0, sticky="w", padx=(0, 8), pady=2)
    w = tk.Spinbox(parent, from_=lo, to=hi, increment=inc,
                   textvariable=var, width=width,
                   bg=T["entry_bg"], fg=T["entry_fg"],
                   buttonbackground=T["btn"], relief="flat",
                   font=("Consolas", 10))
    w.grid(row=row, column=1, sticky="w", pady=2)
    if tip:
        ToolTip(w, tip)
    return w


def combo(parent: tk.Frame, row: int, label: str,
          var: tk.Variable, values: list[str],
          width: int = 12, tip: str = "") -> ttk.Combobox:
    T = _T(parent)
    tk.Label(parent, text=label, bg=parent.cget("bg"), fg=T["fg2"],
             font=("Segoe UI", 9), anchor="w").grid(
                 row=row, column=0, sticky="w", padx=(0, 8), pady=2)
    w = ttk.Combobox(parent, textvariable=var, state="readonly",
                     values=values, font=("Segoe UI", 9), width=width)
    w.grid(row=row, column=1, sticky="w", pady=2)
    if tip:
        ToolTip(w, tip)
    return w


def slider_spin(parent: tk.Frame, row: int, label: str,
                var: tk.Variable, lo: float, hi: float,
                res: float, tip: str = "") -> tk.Frame:
    T = _T(parent)
    tk.Label(parent, text=label, bg=parent.cget("bg"), fg=T["fg2"],
             font=("Segoe UI", 9), anchor="w").grid(
                 row=row, column=0, sticky="w", padx=(0, 8), pady=2)
    f = tk.Frame(parent, bg=parent.cget("bg"))
    f.grid(row=row, column=1, sticky="ew", pady=2)
    s = tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal",
                 variable=var, bg=parent.cget("bg"), fg=T["fg"],
                 troughcolor=T["bar_bg"], highlightthickness=0,
                 activebackground=T["accent"], length=120, showvalue=False)
    s.pack(side="left")
    tk.Label(f, textvariable=var, bg=parent.cget("bg"), fg=T["accent"],
             font=("Consolas", 10), width=6).pack(side="left", padx=2)
    if tip:
        ToolTip(s, tip)
    return f


def cbk(parent: tk.Widget, label: str, var: tk.BooleanVar, tip: str = "") -> None:
    T = _T(parent)
    f = tk.Frame(parent, bg=parent.cget("bg"))
    f.pack(anchor="w", pady=1)
    cb = tk.Checkbutton(f, text=label, variable=var,
                        bg=parent.cget("bg"), fg=T["fg"],
                        activebackground=parent.cget("bg"),
                        activeforeground=T["accent"],
                        selectcolor=T["bg3"], relief="flat",
                        font=("Consolas", 9), cursor="hand2")
    cb.pack(side="left")
    if tip:
        ToolTip(cb, tip)


def flag_row(parent: tk.Widget, flag_name: str,
             en_var: tk.BooleanVar, val_var: tk.Variable | None,
             val_type: str | None, tip: str = "",
             val_values: list[str] | None = None,
             val_width: int = 4) -> None:
    """Checkbox + optional inline value widget. val_type: 'combo'|'entry'|None."""
    T = _T(parent)
    f = tk.Frame(parent, bg=parent.cget("bg"))
    f.pack(anchor="w", pady=1)
    cb = tk.Checkbutton(f, text=flag_name, variable=en_var,
                        bg=parent.cget("bg"), fg=T["fg"],
                        activebackground=parent.cget("bg"),
                        activeforeground=T["accent"],
                        selectcolor=T["bg3"], relief="flat",
                        font=("Consolas", 9), cursor="hand2")
    cb.pack(side="left")
    if tip:
        ToolTip(cb, tip)
    if val_type == "combo" and val_var is not None:
        w = ttk.Combobox(f, textvariable=val_var, values=val_values or [],
                         width=val_width, font=("Consolas", 9), state="readonly")
        w.pack(side="left", padx=(6, 0))
        if tip:
            ToolTip(w, tip)
        def _toggle_c(*_, _w=w, _v=en_var):
            _w.configure(state="readonly" if _v.get() else "disabled")
        en_var.trace_add("write", _toggle_c); _toggle_c()
    elif val_type == "entry" and val_var is not None:
        w = tk.Entry(f, textvariable=val_var, width=val_width,
                     bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                     font=("Consolas", 9), insertbackground=T["fg"])
        w.pack(side="left", padx=(6, 0))
        if tip:
            ToolTip(w, tip)
        def _toggle_e(*_, _w=w, _v=en_var):
            _w.configure(state="normal" if _v.get() else "disabled")
        en_var.trace_add("write", _toggle_e); _toggle_e()


# ── Tooltip ───────────────────────────────────────────────────────────────────

class ToolTip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text   = text
        self.tw: tk.Toplevel | None = None
        widget.bind("<Enter>",       self._schedule, add="+")
        widget.bind("<Leave>",       self._cancel,   add="+")
        widget.bind("<ButtonPress>", self._cancel,   add="+")

    def _schedule(self, event=None):
        self._cancel()
        self._id = self.widget.after(600, self._show)

    def _cancel(self, event=None):
        try:
            self.widget.after_cancel(self._id)
        except Exception:
            pass
        self._hide()

    def _show(self):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        self.tw.attributes("-topmost", True)
        tk.Label(self.tw, text=self.text, justify="left",
                 background="#1e2733", foreground="#c8d0da",
                 relief="flat", font=("Segoe UI", 8),
                 wraplength=380, padx=8, pady=5).pack()

    def _hide(self):
        if self.tw:
            try:
                self.tw.destroy()
            except Exception:
                pass
            self.tw = None


# ── Internal ──────────────────────────────────────────────────────────────────

def _T(widget: tk.Widget) -> dict:
    """Walk up to find the nearest widget that has a stored theme dict."""
    w = widget
    while w is not None:
        if hasattr(w, "_T"):
            return w._T
        parent_name = w.winfo_parent()
        if not parent_name:
            break
        try:
            w = w.nametowidget(parent_name)
        except Exception:
            break
    # Fallback — return a minimal safe dict so nothing crashes
    return {k: "#888888" for k in (
        "fg", "fg2", "accent", "bg3", "entry_bg", "entry_fg",
        "btn", "bar_bg", "green", "red", "orange", "yellow"
    )}
