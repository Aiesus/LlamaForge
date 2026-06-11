# llama-gui V2 — Planning Document

## Goal

A general-purpose Windows GUI that takes anyone from zero to a running local LLM in WSL2.
No assumptions about the user's username, distro, or existing setup.
Distributable as a standalone `.exe` (PyInstaller).

---

## What V2 Is

- A full GUI manager for `llama-server` running in WSL2
- A first-run setup wizard that detects and installs dependencies
- A model downloader (HuggingFace search + download)
- A tool-call conversion proxy, bundled and always-on
- Support for multiple llama.cpp forks (official + TurboQuant, expandable)
- A benchmark optimizer (guided + sweep modes using `llama-bench`)
- A generic Agents tab (Hermes + any OpenAI-compatible agent frontend)
- Designed for anyone to use, not just the original developer

---

## V1 → V2 Differences

### Remove from V1
- `_setup_port_forward()` / `_remove_port_forward()` — Python TCP proxy, dead code (commented out in `load_model`)
- `self._proxy_server` / `self._proxy_thread` attributes
- The `netsh` portproxy cleanup block inside `_setup_port_forward`
- `netsh interface portproxy show all` check inside `_run_diagnostics` — netsh not used
- "Restart Proxy" button wired to `_setup_port_forward()` — calls dead TCP proxy code
- Hardcoded `WSL_DISTRO = "Ubuntu"`, `WSL_USER = "dbent366"`, `WSL_MODELS_UNC`
- Hardcoded proxy path `/home/dbent366/tool-proxy.py`
- Hardcoded TurboQuant binary paths as the only dropdown options
- `tool-proxy-new.py` naming confusion — V2 ships one file: `tool_proxy.py`
- Hermes-specific left panel section → replaced by generic Agents tab

### Fix in V2 (broken in V1)
- **"Restart Proxy" button** — currently calls dead TCP proxy code; rewire to kill and restart `tool-proxy.py` in WSL

### Carry forward from V1 (keep all features)
- All server flags, checkboxes, and parameter sections
- Profiles (save/load/delete)
- GPU + RAM monitoring bars with temperature color coding
- Live log with pause/clear/hide
- mlock fix button
- WSL memory config
- Update Official / Update TurboQuant buttons (generalized per fork)
- Command preview
- Diagnostics
- Theme system (extended in V2)

---

## File Structure

```
llama-gui-v2/
├── main.py                   # entry point — builds app, runs mainloop
├── gui/
│   ├── app.py                # LlamaGUI class, root window, theme, tab layout
│   ├── header.py             # status pills, GPU/RAM bars, tokens/sec, top buttons
│   ├── left_panel.py         # models list (with search), server controls, profiles, download btn
│   ├── tabs/
│   │   ├── server_tab.py     # port, parallel, API key, endpoints, binary/fork selector
│   │   ├── model_tab.py      # ngl, ctx, batch, KV cache, mlock, moe, flash-attn flags
│   │   ├── sampling_tab.py   # temp, top-k/p, min-p, penalties, mirostat, seed
│   │   ├── advanced_tab.py   # rope, multi-GPU, perf flags, speculative decoding, misc, proxy bypass
│   │   ├── agents_tab.py     # agent list, add/remove, start/stop/open per agent
│   │   └── optimizer_tab.py  # llama-bench GUI (guided + sweep modes)
│   ├── setup_wizard.py       # first-run dialog
│   ├── chat_panel.py         # inline SSE chat panel (vertical split in log pane)
│   └── download_manager.py   # HuggingFace download popup (browse + direct + queue)
├── core/
│   ├── settings.py           # load/save settings.json + profiles.json + agents.json
│   ├── wsl.py                # all WSL shell interactions (run cmd, detect distros, get user)
│   ├── server.py             # llama-server start/stop/health check/log streaming
│   ├── agents.py             # agent start/stop/config sync (generic, Hermes is one type)
│   ├── optimizer.py          # llama-bench runner, result parsing, sweep logic
│   ├── monitor.py            # GPU + RAM + CPU + Windows RAM polling thread
│   └── hardware.py           # one-time hardware profile detection (GPU names, CPU, RAM totals)
├── tool_proxy.py             # bundled proxy (deployed to WSL on first run)
├── build.spec                # PyInstaller build spec
└── V2-PLAN.md                # this file
```

---

## Settings (no hardcoded values)

All of the following live in `settings.json` and are configured on first run:

| Key | Description | Default |
|-----|-------------|---------|
| `wsl_distro` | WSL distro name | auto-detected |
| `wsl_user` | WSL username | auto-detected via `wsl whoami` |
| `llama_root` | llama.cpp root path in WSL | `~/llama.cpp` |
| `turbo_root` | TurboQuant fork root in WSL | `~/llama-turbo` (optional) |
| `models_dir` | Models folder in WSL (relative to llama_root) | `models` |
| `llama_bin` | Active server binary | derived from llama_root |
| `proxy_enabled` | Whether proxy auto-starts with model load | `true` |
| `theme` | theme name | `catppuccin-mocha` |
| `geometry` | Window size/position | `1380x860` |
| `last_model` | Last selected model | `""` |
| `last_profile` | Last used profile | `""` |

Derived at runtime (not stored):
- `WSL_MODELS_UNC` = `\\wsl.localhost\{distro}\home\{user}\{llama_root}\{models_dir}`

Agent configs live in `agents.json` (see Agents tab section).

---

## UI Layout

### Overall structure
```
┌─────────────────────────────────────────────────────────────────┐
│ HEADER — status pills │ GPU/RAM bars │ tokens/sec │ buttons      │
├──────────────┬──────────────────────────────────┬───────────────┤
│              │ [Server][Model][Sampling]         │               │
│  LEFT PANEL  │ [Advanced][Agents][Optimizer]     │   LIVE LOG    │
│              │                                   │               │
│  Models list │  (active tab content)             │               │
│  ─────────── │                                   │               │
│  ▶ Load      │                                   │               │
│  ■ Unload    │                                   │               │
│  ─────────── │                                   │               │
│  Profiles    │                                   │               │
│  ─────────── │                                   │               │
│  ⬇ Download  │                                   │               │
└──────────────┴──────────────────────────────────┴───────────────┘
```

### Header additions vs V1
- **Tokens/sec readout** — parsed from llama-server log output, shown when running
- **CPU usage bar** — Windows CPU % via `wmic` (no extra deps)
- **System RAM bar** — Windows total RAM used/free (shows if Windows is squeezing WSL)
- GPU bars carry over unchanged; tooltips on WSL RAM bar show configured cap
- GPU name shown in tooltip on hover over VRAM bar

### Left panel changes vs V1
- Model list gets a **search/filter** text box (useful once you have 20+ models)
- File size shown inline in the list (already computed, just display it)
- Hermes section removed — replaced by Agents tab
- "⬇ Download Models" button at the bottom

### Center panel — tabbed (replaces single scroll)

**Server tab** — binary/fork selector, port, parallel slots, HTTP threads, API key, alias, timeout, endpoint flags (cont-batching, embeddings, metrics, props, slots), update fork buttons

**Model tab** — ngl slider, ctx, batch/ubatch, threads, KV cache types, flash-attn, mlock, no-mmap, cpu-moe, no-warmup, tensor split, main GPU

**Sampling tab** — temperature, top-k, top-p, min-p, repeat penalty, presence/frequency penalty, predict, seed, mirostat section

**Advanced tab** — rope freq base, rope scaling, WSL memory config, performance flags (prio, prio-batch, cache-reuse), speculative decoding (MTP), misc flags, extra flags, command preview, proxy bypass (debug)

**Agents tab** — see section below

**Optimizer tab** — see section below

---

## Agents Tab

Replaces and generalizes the V1 Hermes section. Any OpenAI-compatible agent frontend can be configured here.

### Agent config (stored in `agents.json`)
```json
[
  {
    "name": "Hermes",
    "type": "hermes",
    "exe": "C:/path/to/hermes.exe",
    "config": "C:/Users/.../config.yaml",
    "url": "http://localhost:8088",
    "auto_sync_model": true
  }
]
```

### Tab UI
- List of configured agents (name, status pill: ON/OFF)
- Per-agent row: **Start / Stop / Open** buttons
- **Add Agent** button — opens a small form: name, exe path (browse), URL, config path (browse), sync model toggle
- **Remove** button per agent
- **Edit config** button — opens inline text editor (carried from V1)

### Model sync
- When `auto_sync_model` is true: on server ready, update the agent's config to point at current model + alias + URL
- Logic carried from V1 `_sync_hermes_model()`, generalized per agent type
- Currently supports `type: hermes` (YAML config). Other types can be added later.

### Future agent types
- Open WebUI (launcher + URL)
- SillyTavern
- Any OpenAI-compatible client — at minimum just stores URL, no config sync needed

---

## Hardware Detection

Collected once at startup, refreshed on demand. Used by the optimizer guided mode and
displayed in the header/dashboard. No extra Windows-side libraries — all via stdlib or
built-in OS tools.

### Collection methods

| Info | Source | Method |
|------|--------|--------|
| GPU count | `nvidia-smi` | already collected by monitor thread |
| GPU VRAM per GPU (total) | `nvidia-smi` | already collected |
| GPU model names | `nvidia-smi --query-gpu=name` | new query, one-time at startup |
| GPU driver version | `nvidia-smi --query-gpu=driver_version` | new query |
| Windows total RAM | `wmic ComputerSystem get TotalPhysicalMemory` | stdlib subprocess, no extra deps |
| Windows RAM in use | `wmic OS get FreePhysicalMemory` | same |
| CPU model | `wmic cpu get Name` | same |
| CPU physical cores | `wmic cpu get NumberOfCores` | same |
| CPU logical cores | `wmic cpu get NumberOfLogicalProcessors` | same |
| WSL CPU cores visible | `wsl nproc` | WSL subprocess |
| WSL total RAM | `wsl free -b` | already collected by monitor thread |
| WSL RAM cap (configured) | `~/.wslconfig` | already read |
| Model file size | UNC path `stat()` | already computed for size label |

### Hardware profile object (built at startup, stored in memory)
```python
hw = {
    "gpu_count":      2,
    "gpus": [
        {"index": 0, "name": "NVIDIA GeForce RTX 3060 Ti", "vram_mb": 8192, "driver": "560.94"},
        {"index": 1, "name": "NVIDIA GeForce RTX 3060",    "vram_mb": 12288, "driver": "560.94"},
    ],
    "vram_total_mb":  20480,
    "cpu_name":       "AMD Ryzen 9 5900X 12-Core Processor",
    "cpu_cores_phys": 12,
    "cpu_cores_logic":24,
    "wsl_cpu_cores":  12,
    "ram_total_gb":   32.0,
    "ram_free_gb":    18.4,
    "wsl_ram_cap_gb": 20.0,
    "wsl_ram_total_gb": 19.8,
}
```

### Header display additions
- **CPU bar** alongside RAM bar — Windows CPU usage % (from `wmic` or `psutil` if available)
- **System RAM bar** — Windows total RAM used/free (so user can see if Windows is squeezing WSL)
- Tooltip on WSL RAM bar: "WSL cap: {wsl_ram_cap_gb}GB — change in WSL Memory settings"

### Stored in `settings.json` at startup
Hardware profile cached so the optimizer can use it offline (no re-detection needed on
every open). Refreshed via "Re-detect Hardware" button in the optimizer tab.

---

## Optimizer Tab

Uses `llama-bench` (ships with llama.cpp) to measure prompt processing (pp) and
token generation (tg) speeds.

### Two modes

---

#### Guided Mode

Analyzes the hardware profile + selected model, then recommends a targeted test matrix.
No guesswork for the user — just click Run and get ranked results.

##### Decision logic

**Step 1 — Fit analysis**
```
model_gb    = model_file_size_bytes / 1e9
vram_total  = sum of all GPU VRAM in GB
fits_fully  = model_gb < vram_total * 0.90     # 90% threshold, leave headroom
fits_multi  = fits_fully and gpu_count > 1
is_moe      = "moe" or "moe" in model filename (heuristic, user can override)
cpu_ram_ok  = wsl_ram_cap_gb > model_gb * 0.5  # enough RAM for partial CPU offload
```

**Step 2 — Scenario selection** (one of these, shown to user with explanation)

| Scenario | Condition | Focus |
|----------|-----------|-------|
| Full GPU | `fits_fully`, single GPU | batch/ubatch/KV cache sweep |
| Full Multi-GPU | `fits_fully`, multi GPU | tensor-split ratios + batch/KV sweep |
| Partial offload | model doesn't fit fully | ngl values + batch/KV sweep |
| MoE CPU offload | `is_moe` and `cpu_ram_ok` | cpu-moe on/off + batch/ubatch sweep |
| Long context | ctx > 32K selected | flash-attn, KV cache types, ubatch sweep |
| Max throughput | user selects explicitly | parallel slots + batch + cont-batching |

**Step 3 — Preset test matrix**

Each scenario generates ~6–12 combos. Examples:

*Full GPU, single:*
```
batch=[256,512], ubatch=[128,256], KV-K=[f16,q8_0], KV-V=[f16,q8_0], flash=[on]
→ 2×2×2×2×1 = 16 combos, ~8 min estimated
```

*MoE CPU offload:*
```
cpu-moe=[on,off], batch=[128,256,512], ubatch=[64,128], threads=[4,8,nproc/2]
→ 2×3×2×3 = 36 combos, ~18 min estimated
```

**Step 4 — Run + rank**
- Progress: "Running combo 4 of 16..."
- Live pp/tg shown per combo as it completes
- Final table sorted by tg (generation speed) by default
- "Apply Best" loads winning settings into current profile

##### Guided UI flow
```
[ Hardware Summary ]
  GPU: 2× RTX 3060 Ti (8GB) + RTX 3060 (12GB) = 20GB VRAM
  CPU: Ryzen 9 5900X — 12 cores / 24 threads
  WSL RAM: 20GB cap / 19.8GB visible
  Model: Qwen3-28B-Q4_K_M.gguf — 16.8GB

[ Detected Scenario ]
  ● Full Multi-GPU  (model fits in combined VRAM with headroom)
  ○ MoE CPU Offload
  ○ Long Context
  ○ Max Throughput
  [ Override scenario ▼ ]

[ Test Matrix — 12 combos, ~6 min ]
  tensor-split: [2,3]  [3,2]
  batch:        [256]  [512]
  KV cache:     [f16 / f16]  [turbo3 / turbo4]
  flash-attn:   on

  [ ▶ Run Guided Benchmark ]
```

---

#### Sweep Mode

Full manual control — pick any parameters and any values to test.

- Checkboxes for which parameters to include
- Per-parameter: comma-separated list of values to test
- Live combo count + estimated time before running
- Progress bar during run (X of N complete)
- Cancel button mid-sweep

### Parameters available for sweep
| Parameter | Type | Example values |
|-----------|------|----------------|
| batch size (-b) | list | 128, 256, 512, 1024 |
| ubatch size (-ub) | list | 64, 128, 256, 512 |
| threads (-t) | list | 2, 4, 8 |
| KV cache type K | list | f16, q8_0, q4_0, turbo3 |
| KV cache type V | list | f16, q8_0, q4_0, turbo4 |
| flash-attn | toggle | on, off |
| ngl | list | 80, 90, 99 |
| ctx | list | 4096, 8192, 16384 |
| tensor-split | list | `2,3`  `3,2`  `1,1` |
| cpu-moe | toggle | on, off |
| threads (-t) | list | 4, 8, 12 |

---

### Results table (shared by both modes)
Columns: batch / ubatch / threads / KV-K / KV-V / flash / ngl / pp (t/s) / tg (t/s)
- Sortable by any column
- Top 3 rows highlighted green
- "Apply to Profile" button on each row
- Saved to `bench_results.json` — timestamp + model name + hardware snapshot
- History list: view past runs, filter by model name
- **Export to CSV** button

### llama-bench command format
```
{llama_bench_bin} -m models/{model} -b {batch} -ub {ubatch} -t {threads} \
  --cache-type-k {ck} --cache-type-v {cv} -ngl {ngl} -c {ctx} \
  {--flash-attn on} -o json
```
Output parsed from JSON (`-o json` flag). Raw output also streamed to live log during run.

---

## Tool Proxy

### Current behavior (working correctly, keep as-is)
- Listens on `:8088`, forwards to llama-server on `:8089`
- Auto-detects per request:
  - Request has `"tools"` key → pure passthrough, zero buffering (Cline, `--jinja`)
  - No `"tools"` → buffers SSE stream, converts text `<tool_call>` blocks to OpenAI format (Hermes)
  - Non-chat requests → always passthrough
- WSL copy confirmed in sync with `tool-proxy-new.py` on Windows Desktop

### V2 design
- `tool_proxy.py` ships bundled with the GUI (single source of truth)
- On first run / setup: copied to WSL at `~/tool-proxy.py`
- **Always started when a model loads** — no manual toggle needed
- All clients connect to `:8088` by default
- Runs in WSL (WSL is already running at model load time)

### Toggle
- Removed from main UI
- Lives in **Advanced tab** as "Bypass proxy (debug)" checkbox
- Switches all agent URLs to `:8089` direct when checked
- Not for normal use

---

## Model Downloader

Popup window opened by "⬇ Download Models" button in the left panel.

### UI
- HuggingFace repo ID field (e.g. `bartowski/Qwen3-8B-GGUF`) + Search button
- OR paste a direct HTTPS URL to a `.gguf` file
- Results listbox: filename + size in GiB (GGUF files only)
- Download Selected button
- Progress shown in main live log
- Auto-calls `refresh_models()` on completion

### Backend
- Search: `GET https://huggingface.co/api/models/{repo_id}` → parse `siblings[]`
- Download URL: `https://huggingface.co/{repo_id}/resolve/main/{filename}`
- WSL command: `wget -c --progress=dot:mega -O ~/llama.cpp/models/{filename} {url} 2>&1`
- `-c` = resume interrupted downloads
- Streamed line-by-line to live log (same threading pattern as update buttons)
- No new Windows-side libraries needed

---

## llama.cpp Fork Support

### Built-in forks
| Label | Path | Description |
|-------|------|-------------|
| Official | `{llama_root}/build/bin/llama-server` | Upstream ggml-org/llama.cpp |
| TurboQuant | `{turbo_root}/build/bin/llama-server` | TheTom fork — turbo2/3/4 KV cache (Walsh-Hadamard quantization). Rejected by upstream (#21089), will not merge to main. |

### Design
- Fork selector in Server tab — shows label + one-line description
- If binary not found at path: shown as `(not installed)` and disabled
- Update button per fork — git pull + rebuild (carried from V1)
- KV cache turbo options appear only when TurboQuant binary selected
- `bench_bin` path derived from same fork root (`llama-bench` alongside `llama-server`)
- Adding new forks: one entry in a list in `settings.py`, no other changes needed

---

## Theme System

Default: **Catppuccin Mocha** (carried from V1 — the dark server-y palette)

### Planned themes (stubs in V2, fully implemented over time)
| Name | Style |
|------|-------|
| Catppuccin Mocha | dark, pastel — default |
| Catppuccin Latte | light version of Mocha |
| Nord | dark, cool blue-grey |
| Gruvbox Dark | dark, warm amber/green |
| Dracula | dark, purple-forward |
| High Contrast | accessibility — stark B&W |

### Implementation
- Each theme is a dict of color keys (same structure as V1 `DARK`/`LIGHT`)
- `THEMES = {"catppuccin-mocha": {...}, "nord": {...}, ...}`
- Theme selector in header (replaces simple dark/light toggle)
- `settings.json` stores theme name string
- New themes can be added by appending to the dict — zero structural changes

---

## First-Run Setup Wizard

Triggered when `settings.json` doesn't exist or `wsl_user` is empty.

### Step 1 — WSL Check
- Run `wsl --status` — if WSL2 not found: show instructions + winget/MS Store link, exit wizard
- Run `wsl --list --quiet` — populate distro selector
- If no distros: offer `wsl --install Ubuntu` button

### Step 2 — Distro + User
- Dropdown of available distros (default: first Ubuntu found)
- Auto-detect username via `wsl -d {distro} whoami`
- Show detected user, allow override

### Step 3 — llama.cpp
- Check if `{llama_root}/build/bin/llama-server` exists in WSL
- If yes: show build info, mark ready
- If no: offer two options:
  - **Clone + Build** — `git clone https://github.com/ggml-org/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)`
  - **Set custom path** — text entry

### Step 4 — TurboQuant (optional)
- Brief description: adds turbo2/3/4 KV cache types, ~10-20% VRAM savings on KV cache
- Note: not in upstream, separate binary required
- Options: **Skip** / **Clone + Build** / **Set custom path**

### Step 5 — Dependencies Check
| Dependency | Check | Install |
|------------|-------|---------|
| git | `git --version` | `apt install git` |
| cmake | `cmake --version` | `apt install cmake` |
| build-essential | `gcc --version` | `apt install build-essential` |
| CUDA toolkit | `nvcc --version` | link to NVIDIA docs (manual) |
| Python3 | `python3 --version` | `apt install python3` |
| pip | `pip3 --version` | `apt install python3-pip` |
| aiohttp | `python3 -c "import aiohttp"` | `pip3 install aiohttp` |
| wget | `wget --version` | `apt install wget` |

Each row: live status (checking… → ✓ / ✗) + individual Install button.
"Install All Missing" bulk button. Checks re-run after each install.

### Step 6 — Deploy Proxy
- Copy bundled `tool_proxy.py` to WSL `~/tool-proxy.py`
- Verify `aiohttp` works
- Show result

### Finish
- Save `settings.json`
- Open main window
- "Re-run Setup Wizard" available anytime via Settings or Help menu

---

## exe Build (PyInstaller)

Windows-side Python only needs stdlib — `aiohttp` lives in WSL, not bundled.

```python
# build.spec (outline)
a = Analysis(
    ['main.py'],
    datas=[('tool_proxy.py', '.')],   # bundled as data file, copied to WSL at setup
    hiddenimports=[],
)
exe = EXE(a, name='llama-gui', console=False, icon='assets/icon.ico')
```

- Single-file exe via `--onefile`
- `tool_proxy.py` extracted at runtime to temp dir, then copied to WSL
- No console window (`console=False`) — errors go to crash.log (carried from V1)
- Icon TBD

---

## V1 Pruning Notes

When porting V1 code into V2 modules:

### Delete entirely
- `_setup_port_forward()` / `_remove_port_forward()` — dead TCP proxy code
- `self._proxy_server` / `self._proxy_thread` attributes
- `netsh` portproxy cleanup block inside `_setup_port_forward`
- Commented-out `self._setup_port_forward()` call in `load_model`
- Top-level constants: `WSL_DISTRO`, `WSL_USER`, `WSL_MODELS_UNC`, `LLAMA_BIN`
- Left panel Hermes section (→ Agents tab)
- Simple dark/light theme toggle (→ theme selector with named palettes)
- `self.use_proxy_var` toggle and `_on_proxy_toggle()` (→ Advanced tab debug checkbox)
- `netsh interface portproxy show all` block inside `_run_diagnostics`

### Rewire (keep the button, fix what it calls)
- **"Restart Proxy" button** — was: `_setup_port_forward()` (dead TCP proxy)
  - Now: kill + restart `tool-proxy.py` in WSL, then health-check `:8088`
  ```python
  def _restart_tool_proxy(self):
      def _run():
          self._log("[PROXY] Restarting tool proxy...", "info")
          wsl_run("pkill -f tool-proxy.py; sleep 1; "
                  "nohup python3 ~/tool-proxy.py > /tmp/tool-proxy.log 2>&1 &")
          time.sleep(2)
          # health check
          try:
              urllib.request.urlopen("http://localhost:8088/v1/models", timeout=3)
              self._log("[PROXY] Tool proxy restarted — :8088 is live", "success")
          except Exception as e:
              self._log(f"[PROXY] Proxy not responding after restart: {e}", "error")
      threading.Thread(target=_run, daemon=True).start()
  ```

### Update (keep the function, fix the dead check)
- **`_run_diagnostics()`** — remove `netsh` check, add two new checks:

  | Check | Command | Was in V1? |
  |-------|---------|------------|
  | llama-server process in WSL | `pgrep -a llama-server` | ✅ keep |
  | Port listening in WSL | `ss -tlnp \| grep {port}` | ✅ keep |
  | ~~netsh portproxy rules~~ | ~~`netsh interface portproxy show all`~~ | ❌ remove |
  | localhost health check | `localhost:{port}/health` | ✅ keep |
  | WSL IP direct check | `{wsl_ip}:{port}/health` | ✅ keep |
  | **tool-proxy.py running** | `pgrep -f tool-proxy.py` | ✅ new |
  | **:8088 reachable** | `localhost:8088/v1/models` | ✅ new |

---

---

## Future Feature: Image Generation Server

**Status:** Designed, not implemented. Implement when user decides to proceed.

### Summary
Add a second server type for `stable-diffusion.cpp` / `sd-server` alongside the existing
llama-server. Lets the user run FLUX/SD GGUF models for text-to-image generation directly
from the GUI. The ImgGen filter in the download browser already exists — this completes
the loop.

### Decisions already made
| Decision | Answer |
|---|---|
| sd.cpp setup location | Embedded in Image tab (not main wizard) |
| Pillow dependency | Install-on-demand prompt inside Image tab |
| Output folder | User-configurable field in Image tab |
| Scope | txt2img only (no img2img, inpainting, LoRA) |

### New files
| File | Purpose |
|---|---|
| `core/sd_server.py` | `SdServerController` — start/stop/monitor `sd-server` in WSL |
| `gui/tabs/image_tab.py` | Full Image tab: setup section, params, canvas, history strip |

### Changes to existing files
| File | Change |
|---|---|
| `core/settings.py` | Add `sd_root`, `sd_bin`, `sd_port`, `sd_output_dir` fields |
| `gui/app.py` | Wire up Image tab + sd server controller |
| `gui/left_panel.py` | Image server status pill + start/stop button below main server block |

### Image tab layout
```
┌─ Setup (collapses once configured) ────────────────────────────────┐
│ sd.cpp root in WSL: [~/stable-diffusion.cpp   ] [Test]            │
│ Output folder:      [C:\Users\...\sd-output   ] [Browse]          │
│ Port: [8090]    [Install Pillow]  status: ● ready / ✗ not found   │
└────────────────────────────────────────────────────────────────────┘
┌─ Left column ──────────┐  ┌─ Right: image canvas ──────────────┐
│ Model (GGUF dropdown)  │  │                                    │
│ Width × Height preset  │  │   [generated image displayed here] │
│ Steps  CFG  Seed       │  │                                    │
│ Sampler                │  ├────────────────────────────────────┤
│ Positive prompt        │  │ [thumb] [thumb] [thumb] ← history  │
│ Negative prompt        │  └────────────────────────────────────┘
│ [Generate]  [■ Stop]   │
│ progress bar           │
└────────────────────────┘
```

### SdServerController pattern
Mirrors `core/server.py` `ServerController`:
- `start(model_path)` — builds `sd-server` command, launches via `wsl -d {distro}`
- `stop()` — `pkill -f sd-server` in WSL
- `_stream_log()` — same pattern: reads stdout, fires log_fn callbacks
- `state` — same `ServerState` enum (STOPPED / LOADING / RUNNING)

### sd-server API (stable-diffusion.cpp)
```
POST http://localhost:{sd_port}/txt2img
{
  "prompt": "...", "negative_prompt": "...",
  "width": 1024, "height": 1024,
  "sample_steps": 20, "cfg_scale": 7.0,
  "seed": -1, "sampling_method": "euler"
}
→ { "images": ["<base64 PNG>"] }
```
Note: FLUX models ignore cfg_scale — hide that field when a FLUX model is selected
(detected by `_IMGGEN_RE` matching "flux" in the filename).

### Pillow install flow
On tab open: `try: import PIL` — if ImportError:
Show inline message with a "Install Pillow" button that runs:
`wsl -d {distro} python3 -m pip install pillow` (no — Pillow is Windows-side here, for
tkinter ImageTk). Run: `subprocess.run([sys.executable, "-m", "pip", "install", "pillow"])`.
Re-import after. Only needed once.

### Image display
```python
from PIL import Image, ImageTk
import base64, io
img_data = base64.b64decode(response["images"][0])
img = Image.open(io.BytesIO(img_data))
photo = ImageTk.PhotoImage(img)
canvas.create_image(0, 0, anchor="nw", image=photo)
canvas.image = photo  # keep reference
```

### Rough effort
~450 lines across 3 new/modified files. Larger than chat panel, same architecture pattern.

---

## Open Questions

- [ ] **Name** — not decided yet
- [ ] **Multiple model folders** — some users spread models across drives; worth supporting in V2?
- [ ] **Auto-update the GUI itself** — check GitHub releases on startup? Opt-in?
- [ ] **Non-WSL support** — native Linux / macOS out of scope for V2
- [ ] **Wizard re-run** — decided: yes, accessible from Help menu
- [ ] **Agents: config sync for non-Hermes types** — define per-type sync logic as needed

---

## Status

| Area | Status |
|------|--------|
| Plan document | ✅ |
| `main.py` entry point | ✅ |
| `core/settings.py` | ✅ |
| `core/wsl.py` | ✅ |
| `core/hardware.py` | ✅ |
| `core/monitor.py` | ✅ |
| `core/server.py` | ✅ |
| `core/optimizer.py` | ✅ |
| `core/agents.py` | ✅ |
| `gui/themes.py` | ✅ |
| `gui/app.py` | ✅ |
| `gui/header.py` | ✅ |
| `gui/left_panel.py` | ✅ |
| `gui/chat_panel.py` | ✅ |
| `gui/setup_wizard.py` | ✅ |
| `gui/download_manager.py` | ✅ |
| `gui/widgets.py` | ✅ |
| `gui/tabs/server_tab.py` | ✅ |
| `gui/tabs/model_tab.py` | ✅ |
| `gui/tabs/sampling_tab.py` | ✅ |
| `gui/tabs/advanced_tab.py` | ✅ |
| `gui/tabs/agents_tab.py` | ✅ |
| `gui/tabs/optimizer_tab.py` | ✅ |
| Fork support (Official + TurboQuant) | ✅ |
| HuggingFace download browser + queue | ✅ |
| Capability filters (MoE/REAP/MTP/Coder/Vision/Audio/ImgGen) | ✅ |
| Inline chat panel (SSE streaming) | ✅ |
| Image generation server (sd.cpp) | planned — see Future Feature section |
| exe build spec | not started |

---

## Git History

| Hash | Date | Summary |
|------|------|---------|
| `5823859` | 2026-06-10 | **Initial commit** — V2 working baseline with all tabs, profiles, download manager, CUDA swap |
| `e8f6bbe` | 2026-06-10 | **Add .gitignore** — exclude `__pycache__`, `crash.log`, bench results |
| `d0de991` | 2026-06-11 | **Feature batch** — concurrent download queue (max 3), Cancel/Clear Done, live progress/speed, `Have` column, popular model auto-load, click-to-sort, MoE/REAP/MTP/Coder filter checkboxes, Tags column, action buttons moved to left panel Tools section, log toggle stub pane, `--chat-template-file` control, model list default sort by size desc, pane sash restore fix |
| `98cb1dd` | 2026-06-11 | **Feature batch** — inline chat panel (SSE streaming, vertical split, auto-show on server ready, New Chat, system prompt), Vision/Audio/ImgGen filter checkboxes, REAP niche API query augment, file sizes fixed (`?blobs=true`), llama UI button passes API key, image gen server plan in V2-PLAN.md, `TYPE_CHECKING` annotations on all GUI classes |
| `5939ac6` | 2026-06-11 | **Debugging tools** — crash log viewer (left panel, red badge), log filter entry (live buffer replay, 3000 cap), Copy Log button; monitor.py: single PowerShell call for Win CPU%+RAM; HealthChecker skip_fn to stop polling when server stopped |
| `32517d0` | 2026-06-11 | **Hermes + sash fixes** — port V1 Electron detection + CLI launch + UI open; `.gguf` strip on model sync; config auto-detect from `%LOCALAPPDATA%`; fix `url`/`ui_url` key mismatch; add `…` browse buttons to agent editor; fix pane sash persistence (`sashpos()` doesn't exist — use `sash_coord()`/`sash_place()`) |
| `2d01876` | 2026-06-11 | **Hermes config bootstrap** — create `config.yaml` on first sync if missing; always write `provider: custom` so Hermes uses `base_url` and not a cloud provider fallback |
| `609e374` | 2026-06-11 | **Hermes auto-detect** — `find_hermes_exe()` / `find_hermes_config()` check standard install paths; agent editor auto-populates blank fields on open; orange install note with URL shown when Hermes not found on disk |
| `4b9e92f` | 2026-06-11 | **User manual + help** — `USER_MANUAL.md` full end-user doc; `gui/help_window.py` tabbed quick-reference Toplevel (7 sections); Help menu bar with Quick Reference, Re-run Setup Wizard, About |
| `eb7fc65` | 2026-06-11 | **Rename to LlamaForge** — window title, About dialog, USER_MANUAL.md |
| `74c792b` | 2026-06-11 | **Repo URL** — set to github.com/Aiesus/LlamaForge |
