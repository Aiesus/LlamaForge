# llama-gui User Manual

## What Is llama-gui?

llama-gui lets you download, manage, and run AI language models on your own PC — no internet connection required once models are downloaded, no API costs, no data sent to anyone. You talk to the AI through a built-in chat window, or connect any compatible AI tool (like Hermes Agent) to it.

It runs models using **llama.cpp**, a fast open-source AI inference engine, inside Windows Subsystem for Linux (WSL2) on your PC.

---

## System Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 11 | Windows 11 22H2+ |
| GPU | NVIDIA 8 GB VRAM | NVIDIA 12–24 GB VRAM |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB free | 100 GB+ free |
| WSL2 | Required | Required |
| NVIDIA drivers | 525+ | Latest |

> **What is WSL2?** It's a lightweight Linux environment built into Windows. llama-gui uses it to run llama.cpp, which performs better on Linux. You don't need to know Linux to use it — the setup wizard handles everything.

---

## First-Time Setup

When you launch llama-gui for the first time, the Setup Wizard opens automatically.

### Step 1 — WSL Distro
Select your WSL2 Linux distribution from the dropdown (usually **Ubuntu**). If the list is empty, open a terminal and run `wsl --install`, then restart and relaunch llama-gui.

### Step 2 — WSL Username
Enter your WSL username. This is the username you chose when you first set up WSL. If you don't know it, open a WSL terminal and it will be shown in the prompt (e.g. `dbent@hostname`).

### Step 3 — llama.cpp Path
Enter the path to your llama.cpp installation inside WSL (e.g. `~/llama.cpp`). If you haven't built llama.cpp yet, the wizard has a **Build** button that will clone and compile it for you.

### Step 4 — Models Folder
This is where your downloaded model files will be stored. The default is inside your llama.cpp folder. You can change it, but it must be accessible from WSL.

### Step 5 — Finish
Click **Finish**. The wizard saves your settings and the main interface opens.

> You can re-run the wizard any time from the **Help** menu → **Re-run Setup Wizard**.

---

## The Main Interface

llama-gui has three columns:

```
┌─────────────────┬────────────────────────┬──────────────────┐
│   Left Panel    │     Center Tabs        │   Log / Chat     │
│                 │                        │                  │
│ Server status   │ Load / Model /         │ Live server log  │
│ GPU/RAM meters  │ Sampling / Advanced /  │                  │
│ Model controls  │ Agents / Optimizer     │ Chat panel       │
│ Download button │                        │ (after load)     │
│ Tools section   │                        │                  │
└─────────────────┴────────────────────────┴──────────────────┘
```

You can drag the dividers between columns to resize them. Sizes are saved when you close the app.

The **◀ Hide** button in the log header collapses the right column. Click the **▶** stub to restore it.

---

## Downloading Models

Click **⬇ Download Models** in the left panel to open the Download Manager.

### Browse Tab (easiest)
1. Type a model name in the search box (e.g. `Qwen3`, `Llama`, `bartowski`) and click **Search HF**, or leave it blank to see popular models.
2. Click a model in the top list — its quantized variants appear below.
3. Use the **Filter checkboxes** to narrow by type (MoE, Vision, Audio, etc.).
4. Click a variant in the bottom list. The **Size** column tells you how large the download is.
5. Click **⬇ Add to Queue** at the bottom. The download starts automatically.

### By Repo ID Tab
If you have a specific HuggingFace repo ID (e.g. `bartowski/Qwen3-8B-Instruct-GGUF`), paste it here and click **Search**.

### What Do the Quantizations Mean?

| Label | Quality | VRAM needed |
|---|---|---|
| Q8_0 | Near-perfect | ~1.1× model params in GB |
| Q6_K | Excellent | ~0.85× |
| Q5_K_M | Very good | ~0.7× |
| Q4_K_M | Good — most popular | ~0.6× |
| Q3_K_M | Acceptable | ~0.5× |
| IQ4_NL / IQ3_M | Smallest, good quality | ~0.55× / ~0.45× |
| full | Unquantized (F16/BF16) | ~2× |

> **Rule of thumb:** Pick the largest quant that fits in your VRAM with ~1 GB to spare. Q4_K_M is usually the best balance.

The **Have** column (✓) shows models you've already downloaded.

---

## Loading a Model

### Load Tab
1. Select a model from the **Model** dropdown (lists `.gguf` files in your models folder).
2. Choose a **Profile** (Coding, Creative, Long Context, etc.) or create your own.
3. Set **GPU Layers** — higher = more GPU, faster. Set to 99 to load the whole model on GPU if it fits.
4. Set **Context** — how many tokens the model can "remember" in one conversation. Larger = more memory needed.
5. Click **▶ Load Model**.

The status pill in the left panel changes from `STOPPED` → `LOADING` → `READY`. Loading typically takes 5–30 seconds.

### Profiles
Profiles save all your settings so you can switch between use cases quickly.

- **Coding** — low temperature (0.2), precise answers
- **Creative** — high temperature (1.1), more varied output
- **Long Context** — 32K context, flash attention on
- **Dual GPU** — tensor split for two-GPU setups

Click **Save Profile** to save your current settings under a new name. Profiles are saved to `profiles.json` in the app folder.

### Stopping the Server
Click **■ Stop** in the left panel, or click **Unload** before closing the app.

---

## Chat Panel

Once a model is loaded and the server is ready, the chat panel appears at the bottom of the right column.

- Type your message and press **Enter** or click **Send**.
- The AI response streams in word by word.
- Click **▼ Chat** in the log header to show/hide the chat panel.
- Click **New Chat** to clear the conversation history.
- The **System prompt** box at the top sets the AI's persona and instructions. Leave it blank for the model's default behavior.

> The chat connects through the built-in proxy on port 8088. If the server isn't running, the chat will show an error.

---

## Sampling Tab

Controls how the AI generates text. You generally don't need to change these, but:

| Setting | What it does |
|---|---|
| Temperature | Higher = more creative/random. Lower = more focused/deterministic. 0.7–0.9 is good for general use. |
| Top-P | Limits token choices to the most likely ones. 0.9–0.95 is standard. |
| Min-P | Filters out very unlikely tokens. 0.05 is a good default. |
| Repeat Penalty | Discourages the AI from repeating itself. 1.1 helps with loops. |
| Max Tokens | Maximum length of each response (-1 = unlimited). |

---

## Advanced Tab

| Setting | What it does |
|---|---|
| Flash Attention | Faster, uses less VRAM. Enable unless you have problems. |
| KV Cache Type | `f16` is standard. `q8_0` / `q4_0` save VRAM at slight quality cost. |
| Continuous Batching | Allows multiple simultaneous requests. Leave on for Hermes. |
| Model Lock (mlock) | Keeps model in RAM, prevents swapping. Enable on systems with plenty of RAM. |
| Alias | Name the model reports itself as (e.g. `local`). Useful so Hermes config doesn't need updating when you swap models. |
| API Key | Optional key required to connect to the server. Set in Hermes config too if used. |

---

## Agents Tab — Hermes

Hermes Agent is an AI assistant that connects to your local model and can use tools (web search, file editing, terminal, etc.).

### Setup (first time)
1. Go to the **Agents** tab.
2. Click **Edit** on the Hermes entry.
3. The **Executable** and **Config path** fields auto-fill if Hermes is installed. If not, an orange note shows the install URL.
4. Check **Auto-sync model on server ready** — this keeps Hermes pointed at whichever model you load.
5. Click **Save**.

### Starting Hermes
Click **Start** on the Hermes row. The status pill changes to `RUNNING`. The CLI backend starts, and after ~2 seconds the Hermes desktop window opens.

### How Auto-Sync Works
Every time you load a model, llama-gui automatically updates Hermes's `config.yaml` with:
- The current model name
- `base_url: http://localhost:8088/v1` (the proxy)
- `provider: custom`

You don't need to manually reconfigure Hermes when you switch models.

### First-Time Hermes Install
Download from: **hermes-agent.nousresearch.com**

After installing, open the Agents tab → Edit → the paths auto-detect. Load a model first so the config.yaml gets created and pre-configured, then start Hermes.

---

## Optimizer Tab

Runs **llama-bench** to benchmark your hardware with the current model settings.

1. Set the parameters (context, batch size, etc.) or leave at defaults.
2. Click **Run Benchmark**.
3. Results show tokens/second for prompt processing and generation.
4. Past results are saved and shown in the history table so you can compare settings.

---

## Tools Section (Left Panel)

| Button | What it does |
|---|---|
| **⬇ Download Models** | Opens the HuggingFace download manager |
| **🌐 llama UI** | Opens the llama.cpp built-in web UI in your browser |
| **Restart Proxy** | Restarts the tool-call proxy (port 8088). Use if Hermes can't connect. |
| **Diagnose** | Runs a connectivity check and prints results to the log |
| **Theme** | Opens the theme picker |
| **Crash Log** | Shows the crash log. Turns red if a crash has been recorded. |

---

## Remote Access

You can connect to your llama-gui server from other devices on your network using **Tailscale**.

1. Install Tailscale in WSL2: `curl -fsSL https://tailscale.com/install.sh | sh`
2. Authenticate: `sudo tailscale up`
3. From another device on your Tailscale network, connect to `http://<your-tailscale-ip>:8088/v1`

> **Known limitation:** Connecting directly to port 8089 from a remote device returns a 400 error. Always use port 8088 (the proxy).

---

## Troubleshooting

### Model won't load
- Check the log for error messages (red text).
- Reduce **GPU Layers** — the model may not fit in your VRAM.
- Reduce **Context** size.
- Try a smaller quantization (Q4_K_M instead of Q8_0).

### Server stays in LOADING
- WSL may have run out of memory. Check the RAM meters at the top of the left panel.
- Open a WSL terminal and run `free -h` to see available memory.
- Reduce the WSL memory limit in Settings or load a smaller model.

### Chat isn't working
- Make sure the status pill shows `READY` (green), not `LOADING` or `STOPPED`.
- Click **Restart Proxy** in the Tools section.
- Check the log for proxy errors.

### Hermes can't connect
- Click **Restart Proxy**.
- Load a model first — this triggers the config sync that sets `base_url`.
- Open Agents → Edit → verify the Config path points to a real file.
- Check that `base_url` in the config is `http://localhost:8088/v1`.

### "Executable not found" on Hermes Start
- Go to Agents → Edit → click `…` next to Executable and browse to `Hermes.exe`.
- Standard location: `%LOCALAPPDATA%\hermes\hermes-agent\apps\desktop\release\win-unpacked\Hermes.exe`

### Crash Log is red
- Click **Crash Log** in the Tools section to view the error.
- After reading, click **Clear** to reset the badge.
- If crashes are repeatable, note the error message and check the llama.cpp GitHub issues.

### App opens but shows the setup wizard every time
- The setup wizard re-runs if WSL distro or username is blank. Complete all fields and click Finish.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Enter` (in chat) | Send message |
| `Shift+Enter` (in chat) | New line |

---

## File Locations

| File | Location | Purpose |
|---|---|---|
| `settings.json` | App folder | All app settings |
| `profiles.json` | App folder | Saved model profiles |
| `agents.json` | App folder | Agent configurations |
| `crash.log` | App folder | Crash reports |
| `bench_results.json` | App folder | Benchmark history |
| Hermes config | `%LOCALAPPDATA%\hermes\config.yaml` | Hermes model/server config |
| Models | Your configured models path | `.gguf` model files |
