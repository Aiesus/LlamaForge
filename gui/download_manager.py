"""
HuggingFace model downloader — Browse (keyword search) + Direct (repo ID) modes.
"""
from __future__ import annotations
import json
import re
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Callable
import urllib.request
import urllib.parse

import core.wsl as wsl

LogFn = Callable[[str, str | None], None]

# ── Helpers ───────────────────────────────────────────────────────────────────

_QUANT_RE = re.compile(
    r'\b(IQ[0-9]+_[A-Z0-9]+|Q[0-9]+_K_[SML]|Q[0-9]+_K|Q[0-9]+_[0-9]+|Q[0-9]+|F16|BF16)\b',
    re.IGNORECASE,
)
_MOE_RE = re.compile(r'\b(moe|mixture)\b|-A\d+B\b', re.IGNORECASE)


def _parse_quant(filename: str) -> str:
    m = _QUANT_RE.search(filename)
    return m.group(0).upper() if m else ""


def _is_moe(name: str) -> bool:
    return bool(_MOE_RE.search(name))


def _fmt_size(size: int) -> str:
    if size >= 1024 ** 3:
        return f"{size / 1024**3:.2f} GiB"
    if size >= 1024 ** 2:
        return f"{size / 1024**2:.0f} MiB"
    return "?"


def _fmt_dl(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


def _clean_model_name(model_id: str) -> str:
    """Strip publisher prefix and common suffixes to get a clean model name."""
    name = model_id.split("/")[-1]
    for suffix in ("-GGUF", "-gguf", "-Instruct", "-instruct"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


# ── DownloadManager ───────────────────────────────────────────────────────────

class DownloadManager(tk.Toplevel):

    def __init__(self, root: tk.Tk, state, T: dict,
                 log_fn: LogFn, on_complete: Callable | None = None):
        super().__init__(root)
        self.title("Download Models")
        self.configure(bg=T["bg"])
        self.geometry("860x680")
        self.resizable(True, True)
        self.attributes("-topmost", True)

        self._state         = state
        self._T             = T
        self._log           = log_fn
        self._on_complete   = on_complete

        self._browse_repos: list[dict] = []
        self._browse_files: list[dict] = []
        self._direct_files: list[dict] = []

        self._active_tree: ttk.Treeview | None = None
        self._active_repo: str = ""

        self._build(T)

    def show(self) -> None:
        self.grab_set()

    def _build(self, T: dict) -> None:
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        self._browse_frame = tk.Frame(self._nb, bg=T["bg"])
        self._direct_frame = tk.Frame(self._nb, bg=T["bg"])
        self._nb.add(self._browse_frame, text="  Browse  ")
        self._nb.add(self._direct_frame, text="  By Repo ID  ")

        self._build_browse(T)
        self._build_direct(T)
        self._build_download_bar(T)

    # ── Browse tab ────────────────────────────────────────────────────────────

    def _build_browse(self, T: dict) -> None:
        f = self._browse_frame

        # Search row
        sr = tk.Frame(f, bg=T["bg"])
        sr.pack(fill="x", padx=10, pady=(10, 2))
        tk.Label(sr, text="Search:", bg=T["bg"], fg=T["fg2"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._browse_q_var = tk.StringVar()
        e = tk.Entry(sr, textvariable=self._browse_q_var, width=28,
                     bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                     font=("Consolas", 10), insertbackground=T["fg"])
        e.pack(side="left", padx=6)
        e.bind("<Return>", lambda _: self._browse_search())
        tk.Button(sr, text="Search HF", bg=T["accent"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9, "bold"),
                  command=self._browse_search).pack(side="left")
        self._moe_only_var = tk.BooleanVar()
        tk.Checkbutton(
            sr, text="MoE only", variable=self._moe_only_var,
            bg=T["bg"], fg=T["fg"], selectcolor=T["bg3"],
            activebackground=T["bg"], font=("Segoe UI", 9),
            command=self._browse_apply_filter,
        ).pack(side="left", padx=12)

        self._browse_status = tk.Label(
            f, text='Search HuggingFace — e.g. "Qwen3", "Llama", "bartowski"',
            bg=T["bg"], fg=T["fg2"], font=("Segoe UI", 8),
        )
        self._browse_status.pack(anchor="w", padx=10, pady=(0, 2))

        # ── Step 1: Model list ────────────────────────────────────────────────
        tk.Label(f, text="① Select Model", bg=T["bg"], fg=T["accent"],
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=10, pady=(4, 0))
        rf = tk.Frame(f, bg=T["bg2"])
        rf.pack(fill="x", padx=10, pady=2)
        self._repo_tree = ttk.Treeview(
            rf,
            columns=("model", "publisher", "downloads", "moe"),
            show="headings", height=6,
        )
        self._repo_tree.heading("model",     text="Model")
        self._repo_tree.heading("publisher", text="Publisher")
        self._repo_tree.heading("downloads", text="Downloads")
        self._repo_tree.heading("moe",       text="MoE")
        self._repo_tree.column("model",      width=330)
        self._repo_tree.column("publisher",  width=140)
        self._repo_tree.column("downloads",  width=80, anchor="e")
        self._repo_tree.column("moe",        width=40, anchor="center")
        self._repo_tree.pack(side="left", fill="x", expand=True)
        rsb = ttk.Scrollbar(rf, orient="vertical", command=self._repo_tree.yview)
        self._repo_tree.configure(yscrollcommand=rsb.set)
        rsb.pack(side="right", fill="y")
        self._repo_tree.bind("<<TreeviewSelect>>", self._on_repo_select)

        # ── Step 2: Variant list ──────────────────────────────────────────────
        vhdr = tk.Frame(f, bg=T["bg"])
        vhdr.pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(vhdr, text="② Select Variant", bg=T["bg"], fg=T["accent"],
                 font=("Consolas", 8, "bold")).pack(side="left")
        self._browse_file_filter_var = tk.StringVar()
        self._browse_file_filter_var.trace_add(
            "write", lambda *_: self._browse_populate_files())
        tk.Label(vhdr, text="Filter:", bg=T["bg"], fg=T["fg2"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(20, 4))
        tk.Entry(
            vhdr, textvariable=self._browse_file_filter_var, width=16,
            bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
            font=("Consolas", 9), insertbackground=T["fg"],
        ).pack(side="left")
        tk.Label(vhdr, text="e.g. Q4_K_M", bg=T["bg"],
                 fg=T["fg2"], font=("Segoe UI", 8)).pack(side="left", padx=4)

        # Variant tree (expands to fill remaining space)
        ff = tk.Frame(f, bg=T["bg2"])
        ff.pack(fill="both", expand=True, padx=10, pady=(2, 6))
        self._browse_file_tree = ttk.Treeview(
            ff,
            columns=("quant", "size", "name"),
            show="headings", height=8,
        )
        self._browse_file_tree.heading("quant", text="Quantization")
        self._browse_file_tree.heading("size",  text="Size")
        self._browse_file_tree.heading("name",  text="Filename")
        self._browse_file_tree.column("quant", width=110, anchor="center")
        self._browse_file_tree.column("size",  width=85,  anchor="e")
        self._browse_file_tree.column("name",  width=440)
        self._browse_file_tree.pack(side="left", fill="both", expand=True)
        fsb = ttk.Scrollbar(ff, orient="vertical", command=self._browse_file_tree.yview)
        self._browse_file_tree.configure(yscrollcommand=fsb.set)
        fsb.pack(side="right", fill="y")
        self._browse_file_tree.bind("<<TreeviewSelect>>",
                                    lambda _: self._set_active(self._browse_file_tree))

    def _browse_search(self) -> None:
        q = self._browse_q_var.get().strip()
        if not q:
            messagebox.showwarning("Empty", "Enter a search term.")
            return
        self._browse_status.config(text="Searching HuggingFace…")
        for row in self._repo_tree.get_children():
            self._repo_tree.delete(row)
        for row in self._browse_file_tree.get_children():
            self._browse_file_tree.delete(row)
        self._browse_repos = []
        self._browse_files = []
        threading.Thread(target=self._fetch_repos, args=(q,), daemon=True).start()

    def _fetch_repos(self, query: str) -> None:
        params = urllib.parse.urlencode({
            "search":    query,
            "tags":      "gguf",
            "sort":      "downloads",
            "direction": "-1",
            "limit":     "40",
            "full":      "False",
        })
        url = f"https://huggingface.co/api/models?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llama-gui/2"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                self._browse_repos = json.loads(resp.read())
            self.after(0, self._browse_populate_repos)
        except Exception as e:
            self.after(0, lambda: self._browse_status.config(text=f"Error: {e}"))

    def _browse_populate_repos(self) -> None:
        repos = self._browse_repos
        if self._moe_only_var.get():
            repos = [r for r in repos if _is_moe(r.get("modelId", ""))]
        for row in self._repo_tree.get_children():
            self._repo_tree.delete(row)
        for r in repos:
            mid   = r.get("modelId", "")
            parts = mid.split("/", 1)
            pub   = parts[0] if len(parts) == 2 else ""
            name  = _clean_model_name(mid)
            dl    = _fmt_dl(r.get("downloads", 0))
            moe   = "✓" if _is_moe(mid) else ""
            self._repo_tree.insert("", tk.END, iid=mid,
                                   values=(name, pub, dl, moe))
        self._browse_status.config(text=f"{len(repos)} model(s) found — select one to see variants.")

    def _browse_apply_filter(self) -> None:
        self._browse_populate_repos()

    def _on_repo_select(self, event=None) -> None:
        sel = self._repo_tree.selection()
        if not sel:
            return
        self._active_repo = sel[0]
        for row in self._browse_file_tree.get_children():
            self._browse_file_tree.delete(row)
        self._browse_files = []
        self._browse_status.config(
            text=f"Loading variants for {_clean_model_name(self._active_repo)}…")
        threading.Thread(
            target=self._fetch_files, args=(self._active_repo, "browse"),
            daemon=True,
        ).start()

    def _browse_populate_files(self) -> None:
        q = self._browse_file_filter_var.get().strip().lower()
        for row in self._browse_file_tree.get_children():
            self._browse_file_tree.delete(row)
        shown = 0
        for f in self._browse_files:
            if q and q not in f["name"].lower():
                continue
            quant = _parse_quant(f["name"])
            sz    = _fmt_size(f["size"]) if f["size"] else "?"
            # columns: quant | size | filename
            self._browse_file_tree.insert(
                "", tk.END, iid=f["name"],
                values=(quant or "—", sz, f["name"]),
            )
            shown += 1
        model_name = _clean_model_name(self._active_repo)
        self._browse_status.config(
            text=f"{shown} variant(s) for {model_name}"
        )

    # ── Direct tab ────────────────────────────────────────────────────────────

    def _build_direct(self, T: dict) -> None:
        f = self._direct_frame

        sr = tk.Frame(f, bg=T["bg"])
        sr.pack(fill="x", padx=12, pady=10)
        tk.Label(sr, text="Repo ID:", bg=T["bg"], fg=T["fg2"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._repo_var = tk.StringVar()
        e = tk.Entry(sr, textvariable=self._repo_var, width=42,
                     bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                     font=("Consolas", 10), insertbackground=T["fg"])
        e.pack(side="left", padx=6)
        e.bind("<Return>", lambda _: self._direct_search())
        tk.Button(sr, text="Search", bg=T["accent"], fg=T["bg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9, "bold"),
                  command=self._direct_search).pack(side="left")

        tk.Label(f, text='e.g. "bartowski/Qwen3-8B-Instruct-GGUF"',
                 bg=T["bg"], fg=T["fg2"], font=("Segoe UI", 8)).pack(anchor="w", padx=12)

        lf = tk.Frame(f, bg=T["bg2"])
        lf.pack(fill="both", expand=True, padx=12, pady=4)
        self._direct_tree = ttk.Treeview(
            lf,
            columns=("name", "quant", "size"),
            show="headings", height=16,
        )
        self._direct_tree.heading("name",  text="File")
        self._direct_tree.heading("quant", text="Quant")
        self._direct_tree.heading("size",  text="Size")
        self._direct_tree.column("name",  width=470)
        self._direct_tree.column("quant", width=100, anchor="center")
        self._direct_tree.column("size",  width=90,  anchor="e")
        self._direct_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self._direct_tree.yview)
        self._direct_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._direct_tree.bind("<<TreeviewSelect>>",
                               lambda _: self._set_active(self._direct_tree))

        ff = tk.Frame(f, bg=T["bg"])
        ff.pack(fill="x", padx=12, pady=2)
        tk.Label(ff, text="Filter:", bg=T["bg"], fg=T["fg2"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._direct_filter_var = tk.StringVar()
        self._direct_filter_var.trace_add("write", lambda *_: self._direct_apply_filter())
        tk.Entry(ff, textvariable=self._direct_filter_var, width=28,
                 bg=T["entry_bg"], fg=T["entry_fg"], relief="flat",
                 font=("Consolas", 9), insertbackground=T["fg"]).pack(side="left", padx=6)
        self._direct_status = tk.Label(ff, text="", bg=T["bg"],
                                       fg=T["fg2"], font=("Consolas", 9))
        self._direct_status.pack(side="left", padx=8)

    def _direct_search(self) -> None:
        repo = self._repo_var.get().strip()
        if not repo:
            messagebox.showwarning("Empty", "Enter a HuggingFace repo ID.")
            return
        self._active_repo = repo
        self._direct_status.config(text="Fetching file list…")
        threading.Thread(
            target=self._fetch_files, args=(repo, "direct"),
            daemon=True,
        ).start()

    def _direct_apply_filter(self) -> None:
        q = self._direct_filter_var.get().strip().lower()
        for row in self._direct_tree.get_children():
            self._direct_tree.delete(row)
        for f in self._direct_files:
            if q and q not in f["name"].lower():
                continue
            self._direct_tree.insert(
                "", tk.END, iid=f["name"],
                values=(f["name"], _parse_quant(f["name"]),
                        _fmt_size(f["size"]) if f["size"] else "?"),
            )

    # ── Shared file fetch ─────────────────────────────────────────────────────

    def _fetch_files(self, repo: str, mode: str) -> None:
        url = f"https://huggingface.co/api/models/{repo}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llama-gui/2"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            files = [
                {"name": s["rfilename"], "size": s.get("size", 0)}
                for s in data.get("siblings", [])
                if s.get("rfilename", "").lower().endswith(".gguf")
            ]
            if mode == "browse":
                self._browse_files = files
                self.after(0, self._browse_populate_files)
            else:
                self._direct_files = files
                self.after(0, lambda: (
                    self._direct_apply_filter(),
                    self._direct_status.config(
                        text=f"{len(files)} GGUF file(s) found."),
                ))
        except Exception as e:
            msg = f"Error: {e}"
            if mode == "browse":
                self.after(0, lambda: self._browse_status.config(text=msg))
            else:
                self.after(0, lambda: self._direct_status.config(text=msg))

    # ── Download bar (shared) ─────────────────────────────────────────────────

    def _build_download_bar(self, T: dict) -> None:
        df = tk.Frame(self, bg=T["bg"])
        df.pack(fill="x", padx=12, pady=(4, 2))
        tk.Button(
            df, text="⬇  Download Selected",
            bg=T["green"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 10, "bold"), pady=4,
            command=self._download_selected,
        ).pack(side="left")
        self._dl_status = tk.Label(df, text="", bg=T["bg"],
                                   fg=T["fg2"], font=("Consolas", 9))
        self._dl_status.pack(side="left", padx=10)

        self._prog_log = scrolledtext.ScrolledText(
            self, height=5,
            bg=T["log_bg"], fg=T["log_fg"], font=("Consolas", 8),
            relief="flat", state="disabled", padx=6, pady=4,
        )
        self._prog_log.pack(fill="x", padx=12, pady=(0, 8))
        self._prog_log.tag_config("ok",  foreground=T["green"])
        self._prog_log.tag_config("err", foreground=T["red"])

    def _set_active(self, tree: ttk.Treeview) -> None:
        self._active_tree = tree
        if tree is self._direct_tree:
            self._active_repo = self._repo_var.get().strip()

    def _download_selected(self) -> None:
        tree = self._active_tree
        if tree is None or not tree.selection():
            messagebox.showinfo("Select", "Select a file to download first.")
            return
        filename = tree.selection()[0]
        repo     = self._active_repo
        s        = self._state.settings

        if not repo:
            messagebox.showerror("No Repo", "No repository selected.")
            return
        if not s.wsl_distro or not s.wsl_user:
            messagebox.showerror("Setup", "WSL not configured — run setup first.")
            return

        dest_wsl = f"{s.models_wsl}/{filename}"
        url      = f"https://huggingface.co/{repo}/resolve/main/{filename}"

        self._log_prog(f"Downloading {filename}…")
        self._dl_status.config(text=f"Downloading {filename}…")

        def _run():
            cmd = (f'mkdir -p "{s.models_wsl}" && '
                   f'wget -c --show-progress -O "{dest_wsl}" "{url}"')
            rc = wsl.stream(s.wsl_distro, s.wsl_user, cmd,
                            self._log_prog, timeout=3600)
            if rc == 0:
                self.after(0, lambda: self._dl_status.config(text="Download complete."))
                self._log_prog(f"Saved to {dest_wsl}", "ok")
                if self._on_complete:
                    self.after(0, self._on_complete)
            else:
                self._log_prog(f"Download failed (rc={rc}).", "err")
                self.after(0, lambda: self._dl_status.config(text="Download failed."))

        threading.Thread(target=_run, daemon=True).start()

    def _log_prog(self, text: str, tag: str | None = None) -> None:
        def _do():
            self._prog_log.config(state="normal")
            self._prog_log.insert(tk.END, text + "\n", tag or "")
            self._prog_log.see(tk.END)
            self._prog_log.config(state="disabled")
        self.after(0, _do)
