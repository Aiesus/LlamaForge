"""
llama-bench runner and combo generator.
No tkinter imports — communicates via log_fn callback.
"""
from __future__ import annotations
import itertools
import json
import re
from datetime import datetime, timezone
from typing import Callable

from core import wsl

LogFn = Callable[[str, str | None], None]
DoneFn = Callable[[list[dict]], None]

# ── Scenario definitions ──────────────────────────────────────────────────────
# Each scenario yields a list of param dicts for run_bench_combos.

_KV_TURBO = ["turbo4", "turbo3", "q8_0", "f16"]
_KV_BASIC = ["q8_0", "f16"]
_CTX_LONG = ["32768", "65536", "131072"]
_BATCH    = [512, 1024]


def guided_matrix(scenario: str, hw) -> list[dict]:
    """Return the test combo list for a named scenario given current hardware."""
    vram_gb = hw.vram_total_gb if hw and hw.detected else 0
    cores   = hw.wsl_cpu_cores if hw and hw.detected else 4

    if scenario == "Full GPU":
        return [{"ngl": 99, "ctx": c, "batch": b, "ubatch": b,
                 "flash_attn": fa, "cache_type_k": k, "cache_type_v": v,
                 "threads": cores, "threads_batch": -1}
                for c   in ["8192", "16384"]
                for b   in [512, 1024]
                for fa  in [True, False]
                for k, v in [("turbo3", "turbo4"), ("q8_0", "q8_0"), ("f16", "f16")]]

    if scenario == "Full Multi-GPU":
        return [{"ngl": 99, "ctx": c, "batch": b, "ubatch": b,
                 "flash_attn": True, "cache_type_k": k, "cache_type_v": v,
                 "main_gpu": g, "threads": cores, "threads_batch": -1}
                for c   in ["8192", "16384"]
                for b   in [512]
                for k, v in [("turbo3", "turbo4"), ("q8_0", "q8_0")]
                for g   in [0, 1]]

    if scenario == "Partial Offload":
        if vram_gb > 0:
            # test at 50%, 75%, and 100% of what full GPU would be
            base_ngl = 99
            options  = [max(1, int(base_ngl * r)) for r in [0.5, 0.75, 1.0]]
        else:
            options = [40, 60, 80]
        return [{"ngl": ngl, "ctx": "4096", "batch": 512, "ubatch": 512,
                 "flash_attn": False, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                 "threads": cores, "threads_batch": -1}
                for ngl in options]

    if scenario == "MoE CPU Offload":
        return [{"ngl": ngl, "ctx": "4096", "batch": b, "ubatch": b,
                 "flash_attn": False, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                 "no_mmap": True, "cpu_moe": True,
                 "threads": cores, "threads_batch": -1}
                for ngl in [0, 20, 40]
                for b   in [512, 1024]]

    if scenario == "Long Context":
        return [{"ngl": 99, "ctx": c, "batch": 512, "ubatch": 512,
                 "flash_attn": True, "cache_type_k": k, "cache_type_v": v,
                 "threads": cores, "threads_batch": -1}
                for c   in ["32768", "65536", "131072"]
                for k, v in [("turbo3", "turbo4"), ("q8_0", "q8_0")]]

    if scenario == "Max Throughput":
        return [{"ngl": 99, "ctx": "4096", "batch": b, "ubatch": ub,
                 "flash_attn": True, "cache_type_k": "turbo3", "cache_type_v": "turbo4",
                 "threads": threads, "threads_batch": -1}
                for b       in [512, 1024, 2048]
                for ub      in [512, 1024]
                for threads in [cores, max(1, cores // 2)]
                if ub <= b]

    return []  # unknown scenario


def guided_matrix_count(scenario: str, hw) -> int:
    return len(guided_matrix(scenario, hw))


# ── Sweep combo generator ─────────────────────────────────────────────────────

_SWEEP_RANGES: dict[str, list] = {
    "ngl":          [40, 60, 80, 99],
    "ctx":          ["4096", "8192", "16384", "32768"],
    "batch":        [256, 512, 1024],
    "ubatch":       [256, 512, 1024],
    "flash_attn":   [False, True],
    "cache_type_k": ["f16", "q8_0", "turbo3"],
    "cache_type_v": ["f16", "q8_0", "turbo4"],
    "threads":      [2, 4, 8, 16],
}


def sweep_combos(active_keys: list[str], base_profile: dict, hw) -> list[dict]:
    """Generate all combos for the active sweep params, fixing others from base_profile."""
    ranges = {k: _SWEEP_RANGES.get(k, [base_profile.get(k)]) for k in active_keys}
    keys   = list(ranges.keys())
    vals   = list(ranges.values())
    combos = []
    for combo_vals in itertools.product(*vals):
        c = {
            "ngl":          base_profile.get("ngl",          99),
            "ctx":          str(base_profile.get("ctx",       "8192")),
            "batch":        base_profile.get("batch",         512),
            "ubatch":       base_profile.get("ubatch",        512),
            "flash_attn":   base_profile.get("flash_attn",    False),
            "cache_type_k": base_profile.get("cache_type_k",  "f16"),
            "cache_type_v": base_profile.get("cache_type_v",  "f16"),
            "threads":      base_profile.get("threads",       4),
            "threads_batch":-1,
        }
        for k, v in zip(keys, combo_vals):
            c[k] = v
        # Skip ubatch > batch
        if int(c.get("ubatch", 512)) > int(c.get("batch", 512)):
            continue
        combos.append(c)
    return combos


# ── bench command builder ─────────────────────────────────────────────────────

def _bench_cmd(bench_bin: str, model_wsl: str, model_name: str, combo: dict) -> str:
    parts = [bench_bin]
    parts += ["-m", f'"{model_wsl}/{model_name}"']
    parts += ["-ngl", str(combo.get("ngl", 99))]
    parts += ["-c",   str(combo.get("ctx", 8192))]
    parts += ["-b",   str(combo.get("batch", 512))]
    parts += ["-ub",  str(combo.get("ubatch", 512))]
    parts += ["-t",   str(combo.get("threads", 4))]
    parts += ["-tb",  str(combo.get("threads_batch", -1))]
    if combo.get("flash_attn"):
        parts.append("-fa")
    if k := combo.get("cache_type_k"):
        parts += ["--cache-type-k", k]
    if v := combo.get("cache_type_v"):
        parts += ["--cache-type-v", v]
    if combo.get("no_mmap"):
        parts.append("--no-mmap")
    if combo.get("cpu_moe"):
        parts.append("--cpu-moe")
    parts += ["-o", "json"]
    return " ".join(parts)


# ── Output parser ─────────────────────────────────────────────────────────────

def _parse_bench_json(raw: str) -> tuple[float, float] | None:
    """
    Parse llama-bench JSON output. Returns (pp_tps, tg_tps) or None.
    llama-bench emits a JSON array; we want the avg t/s for pp and tg tests.
    """
    try:
        # Strip non-JSON prefix/suffix lines
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        pp_tps = tg_tps = 0.0
        for entry in data:
            t = entry.get("n_test_type", "")
            v = float(entry.get("avg_ts", 0))
            if t == "pp":
                pp_tps = v
            elif t == "tg":
                tg_tps = v
        return (pp_tps, tg_tps)
    except Exception:
        return None


# ── Runner ────────────────────────────────────────────────────────────────────

def run_bench_combos(distro: str, user: str, bench_bin: str,
                     model_wsl: str, model_name: str,
                     combos: list[dict],
                     log_fn: LogFn, done_fn: DoneFn) -> None:
    """
    Run llama-bench for each combo sequentially.
    Calls done_fn(results) when all are done.
    Called on a background thread — does NOT call Tk directly.
    """
    results = []
    for i, combo in enumerate(combos):
        cmd = _bench_cmd(bench_bin, model_wsl, model_name, combo)
        log_fn(f"[OPT] Run {i+1}/{len(combos)}: {combo_summary(combo)}", "info")
        log_fn(f"[OPT] $ {cmd}", None)

        buf: list[str] = []

        def _collect(line: str, _=None, _b=buf):
            _b.append(line)
            log_fn(line, None)

        rc = wsl.stream(distro, user, cmd, _collect, timeout=300)
        raw = "\n".join(buf)

        if rc != 0:
            log_fn(f"[OPT] Run {i+1} failed (rc={rc}). Skipping.", "warn")
            continue

        parsed = _parse_bench_json(raw)
        if not parsed:
            log_fn(f"[OPT] Run {i+1}: could not parse JSON output.", "warn")
            continue

        pp_tps, tg_tps = parsed
        row = {
            "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "pp":           round(pp_tps, 2),
            "tg":           round(tg_tps, 2),
            "ngl":          combo.get("ngl"),
            "ctx":          combo.get("ctx"),
            "batch":        combo.get("batch"),
            "cache_type_k": combo.get("cache_type_k"),
            "cache_type_v": combo.get("cache_type_v"),
            "flash_attn":   combo.get("flash_attn", False),
            "threads":      combo.get("threads"),
        }
        results.append(row)
        log_fn(f"[OPT] → pp={pp_tps:.1f} t/s  tg={tg_tps:.1f} t/s", "success")

    done_fn(results)


# ── Display helpers ───────────────────────────────────────────────────────────

def combo_summary(combo: dict) -> str:
    parts = [
        f"ngl={combo.get('ngl', '?')}",
        f"ctx={combo.get('ctx', '?')}",
        f"b={combo.get('batch', '?')}",
        f"k={combo.get('cache_type_k', '?')}",
        f"v={combo.get('cache_type_v', '?')}",
    ]
    if combo.get("flash_attn"):
        parts.append("fa")
    if combo.get("cpu_moe"):
        parts.append("cpu-moe")
    return "  ".join(parts)
