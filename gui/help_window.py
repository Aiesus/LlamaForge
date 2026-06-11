"""Quick-reference help window."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

# ── Content ───────────────────────────────────────────────────────────────────

_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Getting Started", [
        ("First-time setup",
         "1. Launch the app — the Setup Wizard opens automatically.\n"
         "2. Select your WSL distro (usually Ubuntu).\n"
         "3. Enter your WSL username (shown in the WSL terminal prompt).\n"
         "4. Set your llama.cpp path (default: ~/llama.cpp).\n"
         "5. Click Finish.\n\n"
         "Re-run any time: Help menu → Re-run Setup Wizard."),
        ("System requirements",
         "• Windows 11\n"
         "• NVIDIA GPU with 8 GB+ VRAM\n"
         "• 16 GB+ RAM\n"
         "• WSL2 with Ubuntu\n"
         "• NVIDIA drivers 525+"),
    ]),
    ("Loading a Model", [
        ("Quick steps",
         "1. Go to the Load tab (center).\n"
         "2. Select a model from the Model dropdown.\n"
         "3. Choose a Profile (or leave on Default).\n"
         "4. Set GPU Layers — higher = more GPU used. 99 = full GPU.\n"
         "5. Click ▶ Load Model.\n\n"
         "Status pill: STOPPED → LOADING → READY (green).\n"
         "Loading takes 5–30 seconds."),
        ("If the model won't load",
         "• Lower GPU Layers (model may not fit in VRAM).\n"
         "• Lower Context size.\n"
         "• Try a smaller quantization (Q4_K_M instead of Q8_0).\n"
         "• Check the log for red error text."),
    ]),
    ("Downloading Models", [
        ("From the browser",
         "1. Click ⬇ Download Models in the left panel.\n"
         "2. Search for a model name (e.g. Qwen3, Llama, bartowski).\n"
         "3. Click a model — variants appear below.\n"
         "4. Click a variant, then ⬇ Add to Queue.\n\n"
         "The Have column (✓) marks models you already own."),
        ("Choosing a quantization",
         "Q4_K_M  — most popular, good quality, fits most GPUs\n"
         "Q5_K_M  — better quality, ~15% more VRAM\n"
         "Q8_0    — near-lossless, needs ~1.1× params in GB\n"
         "IQ4_NL  — smallest file, still good quality\n"
         "full    — unquantized F16, needs 2× params in GB\n\n"
         "Rule of thumb: largest quant that leaves ~1 GB free."),
    ]),
    ("Chat", [
        ("Using the chat panel",
         "The chat panel appears after a model loads.\n\n"
         "• Type your message and press Enter (or click Send).\n"
         "• Shift+Enter adds a new line without sending.\n"
         "• New Chat clears the conversation history.\n"
         "• System prompt sets the AI's persona — leave blank for default.\n"
         "• ▼ Chat in the log header shows/hides the panel."),
        ("If chat isn't working",
         "• Status pill must show READY (green).\n"
         "• Click Restart Proxy in the Tools section.\n"
         "• Check the log for red proxy errors."),
    ]),
    ("Hermes Agent", [
        ("First-time setup",
         "1. Go to the Agents tab.\n"
         "2. Click Edit on the Hermes row.\n"
         "3. Exe and Config path auto-fill if Hermes is installed.\n"
         "   If not found, an orange note shows the install URL.\n"
         "4. Check Auto-sync model on server ready.\n"
         "5. Click Save.\n\n"
         "Install Hermes: hermes-agent.nousresearch.com"),
        ("Starting Hermes",
         "Click Start on the Hermes row.\n"
         "The CLI backend starts first, then the desktop window opens (~2s).\n\n"
         "Every time you load a model, llama-gui automatically updates\n"
         "Hermes's config.yaml with the current model name and\n"
         "base_url: http://localhost:8088/v1"),
        ("Troubleshooting",
         "Can't connect:\n"
         "  • Click Restart Proxy.\n"
         "  • Load a model first to trigger config sync.\n\n"
         "Executable not found:\n"
         "  • Agents → Edit → click … next to Executable.\n"
         "  • Path: %LOCALAPPDATA%\\hermes\\hermes-agent\\\n"
         "          apps\\desktop\\release\\win-unpacked\\Hermes.exe"),
    ]),
    ("Troubleshooting", [
        ("Crash Log",
         "Left panel → Crash Log.\n"
         "Button turns red when a crash has been recorded.\n"
         "Click Clear after reading to reset the badge."),
        ("Diagnose",
         "Left panel → Diagnose.\n"
         "Runs a connectivity check and prints results to the log.\n"
         "Use this if the server, proxy, or WSL connection seems broken."),
        ("Server stuck in LOADING",
         "• WSL may be out of memory — check the RAM meters.\n"
         "• Lower Context or GPU Layers and try again.\n"
         "• In a WSL terminal: free -h to see available memory."),
        ("App re-runs setup wizard every time",
         "Complete all fields in the wizard (distro, username, path)\n"
         "and click Finish. The wizard re-runs if any field is blank."),
    ]),
    ("Tools & Shortcuts", [
        ("Left panel Tools",
         "⬇ Download Models  — HuggingFace download manager\n"
         "🌐 llama UI         — opens built-in web UI in browser\n"
         "Restart Proxy       — restarts the :8088 tool-call proxy\n"
         "Diagnose            — WSL/server connectivity check\n"
         "Theme               — change the colour theme\n"
         "Crash Log           — view crash reports"),
        ("Log panel",
         "Filter box         — type to show only matching log lines\n"
         "Copy               — copies full log to clipboard\n"
         "Clear              — clears the log display\n"
         "⏸ Pause            — stops auto-scroll\n"
         "◀ Hide             — collapses the log panel"),
        ("Keyboard shortcuts",
         "Enter              — send chat message\n"
         "Shift+Enter        — new line in chat input"),
    ]),
]


# ── Window ────────────────────────────────────────────────────────────────────

class HelpWindow(tk.Toplevel):

    def __init__(self, root: tk.Tk, T: dict):
        super().__init__(root)
        self.title("Quick Reference")
        self.geometry("700x540")
        self.configure(bg=T["bg"])
        self.resizable(True, True)
        self._build(T)

    def _build(self, T: dict) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        for section_title, topics in _SECTIONS:
            frame = tk.Frame(nb, bg=T["bg"])
            nb.add(frame, text=f"  {section_title}  ")

            canvas = tk.Canvas(frame, bg=T["bg"], highlightthickness=0)
            vsb    = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=vsb.set)
            vsb.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)

            inner = tk.Frame(canvas, bg=T["bg"])
            win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

            def _on_resize(event, c=canvas, w=win_id):
                c.itemconfig(w, width=event.width)
            canvas.bind("<Configure>", _on_resize)

            def _on_frame_configure(event, c=canvas):
                c.configure(scrollregion=c.bbox("all"))
            inner.bind("<Configure>", _on_frame_configure)

            def _on_mousewheel(event, c=canvas):
                c.yview_scroll(int(-1 * (event.delta / 120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

            for topic_title, body in topics:
                tk.Label(inner, text=topic_title,
                         bg=T["bg"], fg=T["accent"],
                         font=("Segoe UI", 10, "bold"),
                         anchor="w").pack(fill="x", padx=16, pady=(14, 2))
                tk.Label(inner, text=body,
                         bg=T["bg2"], fg=T["fg"],
                         font=("Consolas", 9),
                         justify="left", anchor="nw",
                         wraplength=620).pack(fill="x", padx=16, pady=(0, 2),
                                              ipady=8, ipadx=10)

        tk.Button(self, text="Close", bg=T["btn"], fg=T["btn_fg"],
                  relief="flat", cursor="hand2", font=("Segoe UI", 9),
                  command=self.destroy).pack(pady=(0, 8))
