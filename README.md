# LlamaForge

A Windows GUI for running local AI models with llama.cpp — model browser, one-click server control, inline chat, and Hermes Agent integration. No cloud, no API costs.

![Platform](https://img.shields.io/badge/platform-Windows%2011-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **HuggingFace model browser** — search, filter, and download GGUF models directly. Concurrent queue (up to 3 simultaneous downloads), live progress and speed display.
- **One-click server control** — start/stop llama.cpp server, live status pill, GPU/RAM meters updated every 2 seconds.
- **Model profiles** — save and switch between settings presets (Coding, Creative, Long Context, Dual GPU, or your own).
- **Inline chat panel** — SSE-streaming chat built into the app. Auto-appears when the server is ready.
- **Hermes Agent integration** — auto-detects install, auto-configures `config.yaml` on every model load, Electron + CLI backend launch.
- **TurboQuant fork support** — switch between official llama.cpp and the TurboQuant fork (turbo KV cache).
- **Hardware optimizer** — runs llama-bench and logs results for comparing configurations.
- **Debugging tools** — crash log viewer with badge, live log filter, copy log button.
- **Help menu** — Quick Reference window and Re-run Setup Wizard.

---

## Requirements

| | |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA 8 GB+ VRAM |
| RAM | 16 GB+ |
| WSL2 | Required (Ubuntu recommended) |
| Python | 3.10+ (Windows) |
| NVIDIA drivers | 525+ |

---

## Getting Started

### 1. Install dependencies

```
pip install tkinter
```

> Tkinter is included with standard Python on Windows. No other Python dependencies required.

### 2. Build llama.cpp in WSL2

In a WSL2 terminal:
```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc)
```

### 3. Run LlamaForge

```
python main.py
```

The Setup Wizard opens on first launch and walks you through WSL distro, username, and llama.cpp path.

---

## Interface Overview

```
┌─────────────────┬────────────────────────┬──────────────────┐
│   Left Panel    │     Center Tabs        │   Log / Chat     │
│                 │                        │                  │
│ Server status   │ Load · Model           │ Live server log  │
│ GPU/RAM meters  │ Sampling · Advanced    │                  │
│ Model controls  │ Agents · Optimizer     │ Chat panel       │
│ Download button │                        │ (after load)     │
│ Tools section   │                        │                  │
└─────────────────┴────────────────────────┴──────────────────┘
```

Dividers are draggable and positions are saved on exit.

---

## Hermes Agent

LlamaForge has first-class support for [Hermes Agent](https://hermes-agent.nousresearch.com).

- Auto-detects the Hermes installation from `%LOCALAPPDATA%\hermes`
- On every model load, automatically writes the correct `base_url` and model name into `config.yaml`
- If no config exists, creates one pre-pointed at the local server
- Handles Electron app detection — starts the CLI backend, then opens the UI

---

## Proxy

LlamaForge includes a tool-call proxy (`tool_proxy.py`) that runs in WSL2 on port **8088** and forwards to llama-server on port **8089**. This adds tool-call support to models that use the Hermes format and enables remote access via Tailscale.

```
Hermes / Cline / remote clients
        ↓
  :8088  tool-proxy (WSL2)
        ↓
  :8089  llama-server (WSL2)
```

---

## Project Structure

```
LlamaForge/
├── main.py                  Entry point
├── tool_proxy.py            WSL2 tool-call proxy
├── core/
│   ├── settings.py          Paths, AppSettings dataclass, load/save
│   ├── server.py            ServerController, HealthChecker
│   ├── monitor.py           GPU/RAM polling (WSL + Windows)
│   ├── wsl.py               WSL subprocess helpers, proxy control
│   ├── agents.py            Agent lifecycle, Hermes sync + launch
│   ├── hardware.py          Hardware profile detection
│   └── optimizer.py         llama-bench runner
└── gui/
    ├── app.py               LlamaApp main class, AppState
    ├── header.py            Status bar, GPU/RAM meters
    ├── left_panel.py        Server controls, model list, tools
    ├── chat_panel.py        SSE streaming chat
    ├── download_manager.py  HuggingFace browser + download queue
    ├── help_window.py       Quick reference window
    ├── setup_wizard.py      First-run wizard
    ├── themes.py            Theme definitions
    ├── widgets.py           Shared widget helpers
    └── tabs/
        ├── server_tab.py
        ├── model_tab.py
        ├── sampling_tab.py
        ├── advanced_tab.py
        ├── agents_tab.py
        └── optimizer_tab.py
```

---

## Documentation

- [User Manual](USER_MANUAL.md) — full end-user guide
- [V2 Plan](V2-PLAN.md) — architecture decisions and feature notes

---

## License

MIT
