"""
Color theme definitions.
Each theme is a flat dict of named color keys.
Adding a new theme: append an entry to THEMES — no other changes needed.
"""
from __future__ import annotations

# Key names used throughout the UI
# bg, bg2, bg3       — background hierarchy (darkest to slightly lighter)
# fg, fg2            — foreground text (primary, secondary/muted)
# accent             — highlight / interactive color
# green/red/orange/yellow — semantic status colors
# bar_bg / bar_fg    — progress bar track / fill
# btn / btn_fg       — button background / text
# entry_bg/fg        — text entry background / text
# select_bg/fg       — listbox / combobox selection
# log_bg / log_fg    — live log widget colors

THEMES: dict[str, dict[str, str]] = {

    "catppuccin-mocha": {
        "bg":        "#1e1e2e",
        "bg2":       "#2a2a3e",
        "bg3":       "#313145",
        "fg":        "#cdd6f4",
        "fg2":       "#a6adc8",
        "accent":    "#89b4fa",
        "green":     "#a6e3a1",
        "red":       "#f38ba8",
        "orange":    "#fab387",
        "yellow":    "#f9e2af",
        "bar_bg":    "#313145",
        "bar_fg":    "#89b4fa",
        "btn":       "#45475a",
        "btn_fg":    "#cdd6f4",
        "entry_bg":  "#313145",
        "entry_fg":  "#cdd6f4",
        "select_bg": "#89b4fa",
        "select_fg": "#1e1e2e",
        "log_bg":    "#181825",
        "log_fg":    "#cdd6f4",
    },

    "catppuccin-latte": {
        "bg":        "#eff1f5",
        "bg2":       "#e6e9ef",
        "bg3":       "#dce0e8",
        "fg":        "#4c4f69",
        "fg2":       "#6c6f85",
        "accent":    "#1e66f5",
        "green":     "#40a02b",
        "red":       "#d20f39",
        "orange":    "#fe640b",
        "yellow":    "#df8e1d",
        "bar_bg":    "#dce0e8",
        "bar_fg":    "#1e66f5",
        "btn":       "#ccd0da",
        "btn_fg":    "#4c4f69",
        "entry_bg":  "#dce0e8",
        "entry_fg":  "#4c4f69",
        "select_bg": "#1e66f5",
        "select_fg": "#eff1f5",
        "log_bg":    "#e6e9ef",
        "log_fg":    "#4c4f69",
    },

    "nord": {
        "bg":        "#2e3440",
        "bg2":       "#3b4252",
        "bg3":       "#434c5e",
        "fg":        "#eceff4",
        "fg2":       "#d8dee9",
        "accent":    "#88c0d0",
        "green":     "#a3be8c",
        "red":       "#bf616a",
        "orange":    "#d08770",
        "yellow":    "#ebcb8b",
        "bar_bg":    "#434c5e",
        "bar_fg":    "#88c0d0",
        "btn":       "#4c566a",
        "btn_fg":    "#eceff4",
        "entry_bg":  "#3b4252",
        "entry_fg":  "#eceff4",
        "select_bg": "#88c0d0",
        "select_fg": "#2e3440",
        "log_bg":    "#242933",
        "log_fg":    "#d8dee9",
    },

    "gruvbox-dark": {
        "bg":        "#282828",
        "bg2":       "#3c3836",
        "bg3":       "#504945",
        "fg":        "#ebdbb2",
        "fg2":       "#d5c4a1",
        "accent":    "#83a598",
        "green":     "#b8bb26",
        "red":       "#fb4934",
        "orange":    "#fe8019",
        "yellow":    "#fabd2f",
        "bar_bg":    "#504945",
        "bar_fg":    "#83a598",
        "btn":       "#665c54",
        "btn_fg":    "#ebdbb2",
        "entry_bg":  "#3c3836",
        "entry_fg":  "#ebdbb2",
        "select_bg": "#83a598",
        "select_fg": "#282828",
        "log_bg":    "#1d2021",
        "log_fg":    "#d5c4a1",
    },

    "dracula": {
        "bg":        "#282a36",
        "bg2":       "#343746",
        "bg3":       "#44475a",
        "fg":        "#f8f8f2",
        "fg2":       "#6272a4",
        "accent":    "#bd93f9",
        "green":     "#50fa7b",
        "red":       "#ff5555",
        "orange":    "#ffb86c",
        "yellow":    "#f1fa8c",
        "bar_bg":    "#44475a",
        "bar_fg":    "#bd93f9",
        "btn":       "#44475a",
        "btn_fg":    "#f8f8f2",
        "entry_bg":  "#44475a",
        "entry_fg":  "#f8f8f2",
        "select_bg": "#bd93f9",
        "select_fg": "#282a36",
        "log_bg":    "#1e1f29",
        "log_fg":    "#f8f8f2",
    },

    "high-contrast": {
        "bg":        "#000000",
        "bg2":       "#1a1a1a",
        "bg3":       "#333333",
        "fg":        "#ffffff",
        "fg2":       "#cccccc",
        "accent":    "#00aaff",
        "green":     "#00ff00",
        "red":       "#ff3333",
        "orange":    "#ff8800",
        "yellow":    "#ffff00",
        "bar_bg":    "#333333",
        "bar_fg":    "#00aaff",
        "btn":       "#444444",
        "btn_fg":    "#ffffff",
        "entry_bg":  "#1a1a1a",
        "entry_fg":  "#ffffff",
        "select_bg": "#00aaff",
        "select_fg": "#000000",
        "log_bg":    "#000000",
        "log_fg":    "#cccccc",
    },
}

THEME_LABELS = {
    "catppuccin-mocha": "Catppuccin Mocha (default)",
    "catppuccin-latte": "Catppuccin Latte (light)",
    "nord":             "Nord",
    "gruvbox-dark":     "Gruvbox Dark",
    "dracula":          "Dracula",
    "high-contrast":    "High Contrast",
}

DEFAULT_THEME = "catppuccin-mocha"


def get(name: str) -> dict[str, str]:
    """Return theme dict by name, falling back to default."""
    return THEMES.get(name, THEMES[DEFAULT_THEME])
