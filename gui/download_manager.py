"""
HuggingFace model downloader — Browse + Direct modes, concurrent download queue.
"""
from __future__ import annotations
import json
import os
import re
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gui.app import AppState
import urllib.request
import urllib.parse
from pathlib import Path

LogFn = Callable[[str, str | None], None]

_DL_CONCURRENCY = 3

# ── Helpers ───────────────────────────────────────────────────────────────────

_QUANT_RE = re.compile(
    r'\b(IQ[0-9]+_[A-Z0-9]+|Q[0-9]+_K_[SML]|Q[0-9]+_K|Q[0-9]+_[0-9]+|Q[0-9]+|F16|BF16)\b',
    re.IGNORECASE,
)
_MOE_RE   = re.compile(r'\b(moe|mixture|mixtral)\b|-A\d+B\b|\d+x\d+B\b', re.IGNORECASE)
_REAP_RE  = re.compile(r'\bREAP\b', re.IGNORECASE)
_MTP_RE   = re.compile(r'\bMTP\b',  re.IGNORECASE)
_CODER_RE = re.compile(r'\b(coder|coding|codestral|codegen)\b', re.IGNORECASE)

# Vision-language (image understanding) — name patterns + HF pipeline tags
_VISION_RE   = re.compile(
    r'\b(vl|vision|llava|internvl|moondream|pixtral|cogvlm|paligemma|idefics|'
    r'qwen.*vl|phi.*vision|minicpm.*v|yi.*vl|deepseek.*vl|aria|ovis|got.ocr|'
    r'smolvlm|emu|cambrian|mantis|bunny|dragonfly)\b',
    re.IGNORECASE,
)
_VISION_TAGS = frozenset({"vision", "image-text-to-text", "visual-question-answering",
                           "image-to-text", "vqa"})

# Audio — TTS + ASR/STT — name patterns + HF pipeline tags
_AUDIO_RE    = re.compile(
    r'\b(whisper|bark|kokoro|xtts|speecht5|voicecraft|fish.?speech|f5.?tts|'
    r'dia|zonos|chatterbox|orpheus|parler|outetts|csm)\b',
    re.IGNORECASE,
)
_AUDIO_TAGS  = frozenset({"text-to-speech", "automatic-speech-recognition",
                           "audio", "tts", "asr", "speech"})

# Image generation — FLUX, SD3, SDXL GGUF variants + HF pipeline tags
_IMGGEN_RE   = re.compile(
    r'\b(flux|stable.diffusion|sdxl|sd3|kolors|hunyuan.?dit|pixart|'
    r'aura.?flow|lumina)\b',
    re.IGNORECASE,
)
_IMGGEN_TAGS = frozenset({"text-to-image", "image-generation"})


def _parse_quant(filename: str) -> str:
    m = _QUANT_RE.search(filename)
    return m.group(0).upper() if m else ""


def _is_moe(name: str)   -> bool: return bool(_MOE_RE.search(name))
def _is_reap(name: str)  -> bool: return bool(_REAP_RE.search(name))
def _is_mtp(name: str)   -> bool: return bool(_MTP_RE.search(name))
def _is_coder(name: str) -> bool: return bool(_CODER_RE.search(name))

def _is_vision(mid: str, tags: list | tuple = ()) -> bool:
    return bool(_VISION_RE.search(mid)) or bool(_VISION_TAGS & set(tags))

def _is_audio(mid: str, tags: list | tuple = ()) -> bool:
    return bool(_AUDIO_RE.search(mid)) or bool(_AUDIO_TAGS & set(tags))

def _is_imggen(mid: str, tags: list | tuple = ()) -> bool:
    return bool(_IMGGEN_RE.search(mid)) or bool(_IMGGEN_TAGS & set(tags))


def _model_tags(r: dict | str) -> str:
    """Return compact badge string for the Tags column. Accepts full repo dict or bare model ID."""
    if isinstance(r, str):
        mid, hf_tags = r, ()
    else:
        mid, hf_tags = r.get("modelId", ""), r.get("tags", ())
    parts = []
    if _is_moe(mid):             parts.append("MoE")
    if _is_reap(mid):            parts.append("REAP")
    if _is_mtp(mid):             parts.append("MTP")
    if _is_coder(mid):           parts.append("code")
    if _is_vision(mid, hf_tags):  parts.append("vis")
    if _is_audio(mid, hf_tags):   parts.append("audio")
    if _is_imggen(mid, hf_tags):  parts.append("img")
    return " ".join(parts)


def _fmt_size(size: int) -> str:
    if size >= 1024 ** 3:
        return f"{size / 1024**3:.2f} GiB"
    if size >= 1024 ** 2:
        return f"{size / 1024**2:.0f} MiB"
    return "?"


def _fmt_dl(n: int) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}K"
    return str(n)


def _fmt_date(iso: str) -> str:
    return iso[:10] if iso else "—"


def _fmt_speed(bps: float) -> str:
    if bps >= 1024 ** 2: return f"{bps/1024**2:.1f} MB/s"
    if bps >= 1024:      return f"{bps/1024:.0f} KB/s"
    return "—"


def _sibling_size(s: dict) -> int:
    return s.get("lfs", {}).get("size") or s.get("size") or 0


def _clean_model_name(model_id: str) -> str:
    name = model_id.split("/")[-1]
    for suffix in ("-GGUF", "-gguf", "-Instruct", "-instruct"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


# ── DownloadManager ───────────────────────────────────────────────────────────

class DownloadManager(tk.Toplevel):

    def __init__(self, root: tk.Tk, state: AppState, T: dict,
                 log_fn: LogFn, on_complete: Callable | None = None):
        super().__init__(root)
        self.title("Download Models")
        self.configure(bg=T["bg"])
        self.geometry("920x740")
        self.resizable(True, True)
        self.attributes("-topmost", True)

        self._state       = state
        self._T           = T
        self._log         = log_fn
        self._on_complete = on_complete

        self._browse_repos: list[dict] = []
        self._browse_files: list[dict] = []
        self._direct_files: list[dict] = []
        self._active_tree: ttk.Treeview | None = None
        self._active_repo: str = ""
        self._repo_sort: tuple[str, bool] = ("downloads", True)  # (col, reverse)

        # Download queue
        self._queue: list[dict]  = []
        self._queue_lock         = threading.Lock()
        self._semaphore          = threading.Semaphore(_DL_CONCURRENCY)

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
        self._build_queue_panel(T)
        self.after(200, self._load_popular)

    # ── Browse tab ────────────────────────────────────────────────────────────

    def _build_browse(self, T: dict) -> None:
        f = self._browse_frame

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

        # ── Filter row ────────────────────────────────────────────────────────
        fr = tk.Frame(f, bg=T["bg"])
        fr.pack(fill="x", padx=10, pady=(0, 2))
        tk.Label(fr, text="Filter:", bg=T["bg"], fg=T["fg2"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._moe_only_var    = tk.BooleanVar()
        self._reap_only_var   = tk.BooleanVar()
        self._mtp_only_var    = tk.BooleanVar()
        self._coder_only_var  = tk.BooleanVar()
        self._vision_only_var = tk.BooleanVar()
        self._audio_only_var  = tk.BooleanVar()
        self._imggen_only_var = tk.BooleanVar()
        for label, var in (
            ("MoE",    self._moe_only_var),
            ("REAP",   self._reap_only_var),
            ("MTP",    self._mtp_only_var),
            ("Coder",  self._coder_only_var),
            ("Vision", self._vision_only_var),
            ("Audio",  self._audio_only_var),
            ("ImgGen", self._imggen_only_var),
        ):
            tk.Checkbutton(
                fr, text=label, variable=var,
                bg=T["bg"], fg=T["fg"], selectcolor=T["bg3"],
                activebackground=T["bg"], font=("Segoe UI", 9),
                command=self._browse_apply_filter,
            ).pack(side="left", padx=(6, 0))

        self._browse_status = tk.Label(
            f, text='Search HuggingFace — e.g. "Qwen3", "Llama", "bartowski"',
            bg=T["bg"], fg=T["fg2"], font=("Segoe UI", 8),
        )
        self._browse_status.pack(anchor="w", padx=10, pady=(0, 2))

        # ── Step 1: repo list ─────────────────────────────────────────────────
        tk.Label(f, text="① Select Model", bg=T["bg"], fg=T["accent"],
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=10, pady=(4, 0))
        rf = tk.Frame(f, bg=T["bg2"])
        rf.pack(fill="x", padx=10, pady=2)
        self._repo_tree = ttk.Treeview(
            rf,
            columns=("model", "publisher", "downloads", "tags", "updated"),
            show="headings", height=6,
        )
        for _col, _lbl in (("model","Model"),("publisher","Publisher"),
                            ("downloads","Downloads"),("tags","Tags"),("updated","Updated")):
            self._repo_tree.heading(
                _col, text=_lbl,
                command=lambda c=_col: self._sort_repos(c),
            )
        self._repo_tree.column("model",      width=245)
        self._repo_tree.column("publisher",  width=110)
        self._repo_tree.column("downloads",  width=72,  anchor="e")
        self._repo_tree.column("tags",       width=105, anchor="center")
        self._repo_tree.column("updated",    width=85,  anchor="center")
        self._repo_tree.pack(side="left", fill="x", expand=True)
        rsb = ttk.Scrollbar(rf, orient="vertical", command=self._repo_tree.yview)
        self._repo_tree.configure(yscrollcommand=rsb.set)
        rsb.pack(side="right", fill="y")
        self._repo_tree.bind("<<TreeviewSelect>>", self._on_repo_select)

        # ── Step 2: variant list ──────────────────────────────────────────────
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

        ff = tk.Frame(f, bg=T["bg2"])
        ff.pack(fill="both", expand=True, padx=10, pady=(2, 6))
        self._browse_file_tree = ttk.Treeview(
            ff,
            columns=("quant", "size", "have", "name"),
            show="headings", height=7,
        )
        self._browse_file_tree.heading("quant", text="Quantization")
        self._browse_file_tree.heading("size",  text="Size")
        self._browse_file_tree.heading("have",  text="Have")
        self._browse_file_tree.heading("name",  text="Filename")
        self._browse_file_tree.column("quant", width=110, anchor="center")
        self._browse_file_tree.column("size",  width=85,  anchor="e")
        self._browse_file_tree.column("have",  width=40,  anchor="center")
        self._browse_file_tree.column("name",  width=390)
        self._browse_file_tree.tag_configure("owned", foreground=T["green"])
        self._browse_file_tree.pack(side="left", fill="both", expand=True)
        fsb = ttk.Scrollbar(ff, orient="vertical", command=self._browse_file_tree.yview)
        self._browse_file_tree.configure(yscrollcommand=fsb.set)
        fsb.pack(side="right", fill="y")
        self._browse_file_tree.bind("<<TreeviewSelect>>",
                                    lambda _: self._set_active(self._browse_file_tree))

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
            columns=("name", "quant", "size", "have"),
            show="headings", height=16,
        )
        self._direct_tree.heading("name",  text="File")
        self._direct_tree.heading("quant", text="Quant")
        self._direct_tree.heading("size",  text="Size")
        self._direct_tree.heading("have",  text="Have")
        self._direct_tree.column("name",  width=420)
        self._direct_tree.column("quant", width=100, anchor="center")
        self._direct_tree.column("size",  width=90,  anchor="e")
        self._direct_tree.column("have",  width=45,  anchor="center")
        self._direct_tree.tag_configure("owned", foreground=T["green"])
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

    # ── Queue panel ───────────────────────────────────────────────────────────

    def _build_queue_panel(self, T: dict) -> None:
        outer = tk.Frame(self, bg=T["bg"])
        outer.pack(fill="x", padx=8, pady=(4, 8))

        btn_row = tk.Frame(outer, bg=T["bg"])
        btn_row.pack(fill="x", pady=(0, 4))

        tk.Button(
            btn_row, text="⬇  Add to Queue",
            bg=T["green"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9, "bold"), pady=3,
            command=self._add_to_queue,
        ).pack(side="left", padx=(0, 6))

        tk.Button(
            btn_row, text="✕  Cancel",
            bg=T["red"], fg=T["bg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9), pady=3,
            command=self._cancel_selected_dl,
        ).pack(side="left", padx=(0, 6))

        tk.Button(
            btn_row, text="Clear Done",
            bg=T["btn"], fg=T["btn_fg"], relief="flat", cursor="hand2",
            font=("Segoe UI", 9), pady=3,
            command=self._clear_completed_dl,
        ).pack(side="left")

        self._queue_status_lbl = tk.Label(
            btn_row, text="", bg=T["bg"], fg=T["fg2"], font=("Consolas", 9)
        )
        self._queue_status_lbl.pack(side="right")

        qf = tk.Frame(outer, bg=T["bg2"])
        qf.pack(fill="x")
        self._queue_tree = ttk.Treeview(
            qf,
            columns=("file", "size", "progress", "status"),
            show="headings", height=5,
        )
        self._queue_tree.heading("file",     text="File")
        self._queue_tree.heading("size",     text="Size")
        self._queue_tree.heading("progress", text="Progress")
        self._queue_tree.heading("status",   text="Status")
        self._queue_tree.column("file",     width=440, stretch=True)
        self._queue_tree.column("size",     width=85,  anchor="e",      stretch=False)
        self._queue_tree.column("progress", width=72,  anchor="center", stretch=False)
        self._queue_tree.column("status",   width=150, anchor="w",      stretch=False)
        self._queue_tree.tag_configure("active",    foreground=T["accent"])
        self._queue_tree.tag_configure("done",      foreground=T["green"])
        self._queue_tree.tag_configure("failed",    foreground=T["red"])
        self._queue_tree.tag_configure("cancelled", foreground=T["fg2"])
        self._queue_tree.pack(side="left", fill="x", expand=True)
        qsb = ttk.Scrollbar(qf, orient="vertical", command=self._queue_tree.yview)
        self._queue_tree.configure(yscrollcommand=qsb.set)
        qsb.pack(side="right", fill="y")

    # ── Search / populate ─────────────────────────────────────────────────────

    def _load_popular(self) -> None:
        self._browse_status.config(text="Loading popular GGUF models…")
        for row in self._repo_tree.get_children():
            self._repo_tree.delete(row)
        self._browse_repos = []
        threading.Thread(target=self._fetch_repos, daemon=True).start()

    def _browse_search(self) -> None:
        self._browse_status.config(text="Searching HuggingFace…")
        for row in self._repo_tree.get_children():
            self._repo_tree.delete(row)
        for row in self._browse_file_tree.get_children():
            self._browse_file_tree.delete(row)
        self._browse_repos = []
        self._browse_files = []
        threading.Thread(target=self._fetch_repos, daemon=True).start()

    def _hf_fetch(self, search: str | None, limit: int = 200) -> list[dict]:
        """Single HF API request. Returns list of repo dicts (may be empty on error)."""
        params: dict = {"tags": "gguf", "sort": "downloads", "direction": "-1",
                        "limit": str(limit)}
        if search:
            params["search"] = search
        url = f"https://huggingface.co/api/models?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LlamaForge/2"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except Exception:
            return []

    def _fetch_repos(self) -> None:
        base = self._browse_q_var.get().strip()
        bl   = base.lower()

        # Each flag maps to the HF search term that surfaces its models.
        FLAG_TERMS = [
            (self._moe_only_var,    "MoE"),
            (self._reap_only_var,   "REAP"),
            (self._mtp_only_var,    "MTP"),
            (self._coder_only_var,  "coder"),
            (self._vision_only_var, "VL"),
            (self._audio_only_var,  "speech"),
            (self._imggen_only_var, "flux"),
        ]
        active_terms = [t for var, t in FLAG_TERMS if var.get()]

        try:
            if not active_terms:
                # No flags — plain search or top downloads
                self._browse_repos = self._hf_fetch(base or None, limit=40)

            elif len(active_terms) == 1:
                # Single flag: one request, combine with base query
                term = active_terms[0]
                query = " ".join(filter(None, [base, term if term.lower() not in bl else ""]))
                self._browse_repos = self._hf_fetch(query or None, limit=200)

            else:
                # Multiple flags: one request per flag, intersect by model ID.
                # Each fetch gets its own 200-result pool so the intersection
                # contains models that genuinely satisfy every flag.
                results_by_flag: list[list[dict]] = []
                repo_map: dict[str, dict] = {}
                for term in active_terms:
                    query = " ".join(filter(None, [base, term if term.lower() not in bl else ""]))
                    repos = self._hf_fetch(query or None, limit=200)
                    results_by_flag.append(repos)
                    for r in repos:
                        repo_map[r.get("modelId", "")] = r

                # AND: keep only IDs present in every flag's result set
                id_sets = [set(r.get("modelId", "") for r in lst) for lst in results_by_flag]
                common  = id_sets[0].intersection(*id_sets[1:])
                self._browse_repos = [repo_map[mid] for mid in common if mid in repo_map]

            self.after(0, self._browse_populate_repos)
        except Exception as e:
            self.after(0, lambda: self._browse_status.config(text=f"Error: {e}"))

    _REPO_SORT_KEY = {
        "model":     lambda r: _clean_model_name(r.get("modelId", "")).lower(),
        "publisher": lambda r: r.get("modelId", "").split("/")[0].lower(),
        "downloads": lambda r: r.get("downloads", 0),
        "tags":      lambda r: _model_tags(r),
        "updated":   lambda r: r.get("lastModified", ""),
    }

    def _sort_repos(self, col: str) -> None:
        cur_col, cur_rev = self._repo_sort
        reverse = not cur_rev if col == cur_col else (col == "downloads")
        self._repo_sort = (col, reverse)
        self._browse_populate_repos()

    def _browse_populate_repos(self) -> None:
        repos = list(self._browse_repos)

        # AND-logic filters — each checked box narrows the list further
        mid_fn  = lambda r: r.get("modelId", "")
        tags_fn = lambda r: r.get("tags", ())
        if self._moe_only_var.get():
            repos = [r for r in repos if _is_moe(mid_fn(r))]
        if self._reap_only_var.get():
            repos = [r for r in repos if _is_reap(mid_fn(r))]
        if self._mtp_only_var.get():
            repos = [r for r in repos if _is_mtp(mid_fn(r))]
        if self._coder_only_var.get():
            repos = [r for r in repos if _is_coder(mid_fn(r))]
        if self._vision_only_var.get():
            repos = [r for r in repos if _is_vision(mid_fn(r), tags_fn(r))]
        if self._audio_only_var.get():
            repos = [r for r in repos if _is_audio(mid_fn(r), tags_fn(r))]
        if self._imggen_only_var.get():
            repos = [r for r in repos if _is_imggen(mid_fn(r), tags_fn(r))]

        sort_col, reverse = self._repo_sort
        key_fn = self._REPO_SORT_KEY.get(sort_col, self._REPO_SORT_KEY["downloads"])
        repos.sort(key=key_fn, reverse=reverse)

        arrow  = {True: " ▼", False: " ▲"}
        labels = {"model": "Model", "publisher": "Publisher",
                  "downloads": "Downloads", "tags": "Tags", "updated": "Updated"}
        for col, lbl in labels.items():
            self._repo_tree.heading(col, text=lbl + (arrow[reverse] if col == sort_col else ""))

        for row in self._repo_tree.get_children():
            self._repo_tree.delete(row)
        for r in repos:
            mid     = r.get("modelId", "")
            parts   = mid.split("/", 1)
            pub     = parts[0] if len(parts) == 2 else ""
            dl      = _fmt_dl(r.get("downloads", 0))
            tags    = _model_tags(r)
            updated = _fmt_date(r.get("lastModified", ""))
            self._repo_tree.insert("", tk.END, iid=mid,
                                   values=(_clean_model_name(mid), pub, dl, tags, updated))
        label = ("Top GGUF models by downloads"
                 if not self._browse_q_var.get().strip()
                 else f"{len(repos)} model(s) found")
        self._browse_status.config(text=f"{label} — select one to see variants.")

    def _browse_apply_filter(self) -> None:
        # Re-fetch so the API query includes any newly-active niche filter terms
        # (e.g. REAP models don't appear in top-40 without querying for them).
        self._browse_search()

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

    def _owned_files(self) -> set[str]:
        try:
            unc = self._state.settings.models_unc
            if not unc:
                return set()
            return {p.name for p in Path(unc).iterdir()
                    if p.suffix.lower() == ".gguf"}
        except Exception:
            return set()

    def _browse_populate_files(self) -> None:
        q     = self._browse_file_filter_var.get().strip().lower()
        owned = self._owned_files()
        for row in self._browse_file_tree.get_children():
            self._browse_file_tree.delete(row)
        shown = 0
        for f in self._browse_files:
            if q and q not in f["name"].lower():
                continue
            have = "✓" if f["name"] in owned else ""
            tags = ("owned",) if f["name"] in owned else ()
            self._browse_file_tree.insert(
                "", tk.END, iid=f["name"],
                values=(_parse_quant(f["name"]) or "full",
                        _fmt_size(f["size"]) if f["size"] else "?",
                        have, f["name"]),
                tags=tags,
            )
            shown += 1
        if shown == 0 and not self._browse_files:
            self._browse_status.config(
                text=f"No GGUF files found in {_clean_model_name(self._active_repo)} — try By Repo ID tab")
        else:
            self._browse_status.config(
                text=f"{shown} variant(s) for {_clean_model_name(self._active_repo)}")

    def _direct_search(self) -> None:
        repo = self._repo_var.get().strip()
        if not repo:
            messagebox.showwarning("Empty", "Enter a HuggingFace repo ID.")
            return
        self._active_repo = repo
        self._direct_status.config(text="Fetching file list…")
        threading.Thread(
            target=self._fetch_files, args=(repo, "direct"), daemon=True,
        ).start()

    def _direct_apply_filter(self) -> None:
        q     = self._direct_filter_var.get().strip().lower()
        owned = self._owned_files()
        for row in self._direct_tree.get_children():
            self._direct_tree.delete(row)
        for f in self._direct_files:
            if q and q not in f["name"].lower():
                continue
            have = "✓" if f["name"] in owned else ""
            tags = ("owned",) if f["name"] in owned else ()
            self._direct_tree.insert(
                "", tk.END, iid=f["name"],
                values=(f["name"], _parse_quant(f["name"]) or "full",
                        _fmt_size(f["size"]) if f["size"] else "?",
                        have),
                tags=tags,
            )

    def _fetch_files(self, repo: str, mode: str) -> None:
        url = f"https://huggingface.co/api/models/{repo}?blobs=true"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llama-gui/2"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            files = [
                {"name": s["rfilename"], "size": _sibling_size(s)}
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
                    self._direct_status.config(text=f"{len(files)} GGUF file(s) found."),
                ))
        except Exception as e:
            msg = f"Error: {e}"
            if mode == "browse":
                self.after(0, lambda: self._browse_status.config(text=msg))
            else:
                self.after(0, lambda: self._direct_status.config(text=msg))

    # ── Queue ─────────────────────────────────────────────────────────────────

    def _set_active(self, tree: ttk.Treeview) -> None:
        self._active_tree = tree
        if tree is self._direct_tree:
            self._active_repo = self._repo_var.get().strip()

    def _add_to_queue(self) -> None:
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
        if not s.models_unc:
            messagebox.showerror("Setup", "Models path not configured.")
            return

        with self._queue_lock:
            if any(item["filename"] == filename for item in self._queue
                   if item["status"] not in ("done", "failed", "cancelled")):
                self._queue_status_lbl.config(text=f"Already queued: {filename}")
                return

        expected = 0
        for f in self._browse_files + self._direct_files:
            if f["name"] == filename:
                expected = f["size"]
                break

        item: dict = {
            "filename":      filename,
            "repo":          repo,
            "url":           f"https://huggingface.co/{repo}/resolve/main/{filename}",
            "dest_unc":      str(Path(s.models_unc) / filename),
            "expected_size": expected,
            "status":        "queued",
            "bytes_done":    0,
            "speed_bps":     0.0,
            "cancel_event":  threading.Event(),
        }

        with self._queue_lock:
            self._queue.append(item)

        self._queue_tree.insert(
            "", tk.END, iid=filename,
            values=(filename, _fmt_size(expected) if expected else "?", "—", "queued"),
        )
        self._refresh_status_label()
        threading.Thread(target=self._dl_worker, args=(item,), daemon=True).start()

    def _dl_worker(self, item: dict) -> None:
        self._semaphore.acquire()
        if item["cancel_event"].is_set():
            item["status"] = "cancelled"
            self._semaphore.release()
            self.after(0, lambda i=item: self._refresh_queue_row(i))
            self.after(0, self._refresh_status_label)
            return

        item["status"] = "downloading"
        self.after(0, lambda i=item: self._refresh_queue_row(i))
        self.after(0, self._refresh_status_label)

        dest_part = item["dest_unc"] + ".part"
        try:
            req = urllib.request.Request(
                item["url"], headers={"User-Agent": "llama-gui/2"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                total      = int(resp.headers.get("Content-Length", 0)) or item["expected_size"]
                downloaded = 0
                t_prev     = time.time()
                b_prev     = 0

                with open(dest_part, "wb") as f:
                    while True:
                        if item["cancel_event"].is_set():
                            item["status"] = "cancelled"
                            break
                        chunk = resp.read(1 << 17)  # 128 KB
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        item["bytes_done"] = downloaded

                        now = time.time()
                        if now - t_prev >= 1.2:
                            item["speed_bps"] = (downloaded - b_prev) / (now - t_prev)
                            b_prev = downloaded
                            t_prev = now
                            self.after(0, lambda i=item: self._refresh_queue_row(i))

            if item["status"] != "cancelled":
                os.replace(dest_part, item["dest_unc"])
                item["status"]    = "done"
                item["speed_bps"] = 0.0
                if self._on_complete:
                    # Use the root Tk window — this Toplevel may have been closed
                    # by the user before the download thread finished.
                    try:
                        self._state.root.after(0, self._on_complete)
                    except Exception:
                        pass
            else:
                try:
                    os.unlink(dest_part)
                except OSError:
                    pass

        except Exception as e:
            try:
                os.unlink(dest_part)
            except OSError:
                pass
            if item["status"] != "cancelled":
                item["status"] = "failed"
                item["error"]  = str(e)

        finally:
            self._semaphore.release()
            try:
                self.after(0, lambda i=item: self._refresh_queue_row(i))
                self.after(0, self._refresh_status_label)
            except Exception:
                pass

    def _refresh_queue_row(self, item: dict) -> None:
        try:
            st    = item["status"]
            done  = item["bytes_done"]
            total = item["expected_size"]

            if st == "done":
                pct_str    = "100%"
                status_str = "✓ done"
                tag        = "done"
            elif st == "downloading":
                pct_str    = f"{int(done/total*100)}%" if total else "—"
                status_str = f"↓  {_fmt_speed(item['speed_bps'])}"
                tag        = "active"
            elif st == "cancelled":
                pct_str    = "—"
                status_str = "cancelled"
                tag        = "cancelled"
            elif st == "failed":
                pct_str    = "—"
                status_str = f"✗ {item.get('error', 'failed')[:28]}"
                tag        = "failed"
            else:
                pct_str    = "—"
                status_str = "queued"
                tag        = ""

            self._queue_tree.item(
                item["filename"],
                values=(item["filename"],
                        _fmt_size(item["expected_size"]) if item["expected_size"] else "?",
                        pct_str, status_str),
                tags=(tag,) if tag else (),
            )
        except Exception:
            pass

    def _refresh_status_label(self) -> None:
        try:
            active = sum(1 for i in self._queue if i["status"] == "downloading")
            queued = sum(1 for i in self._queue if i["status"] == "queued")
            parts  = []
            if active: parts.append(f"↓ {active} active")
            if queued: parts.append(f"{queued} waiting")
            self._queue_status_lbl.config(
                text="  •  ".join(parts) if parts else ""
            )
        except Exception:
            pass

    def _cancel_selected_dl(self) -> None:
        sel = self._queue_tree.selection()
        if not sel:
            return
        iid = sel[0]
        with self._queue_lock:
            for item in self._queue:
                if item["filename"] == iid and item["status"] in ("queued", "downloading"):
                    item["cancel_event"].set()
                    item["status"] = "cancelled"
                    self.after(0, lambda i=item: self._refresh_queue_row(i))
                    self.after(0, self._refresh_status_label)
                    break

    def _clear_completed_dl(self) -> None:
        done_statuses = ("done", "failed", "cancelled")
        with self._queue_lock:
            to_remove    = [i for i in self._queue if i["status"] in done_statuses]
            self._queue  = [i for i in self._queue if i["status"] not in done_statuses]
        for item in to_remove:
            try:
                self._queue_tree.delete(item["filename"])
            except Exception:
                pass
        self._refresh_status_label()
