"""
llama-bench runner, VRAM probe, and goal-based guided benchmark orchestration.
No tkinter imports — communicates via log_fn callback.
"""
from __future__ import annotations
import itertools
import json
import re
from datetime import datetime, timezone
from typing import Callable

from core import wsl

LogFn   = Callable[[str, str | None], None]
DoneFn  = Callable[[list[dict]], None]
ProbeFn = Callable[[dict], None]   # called with {(kv_k, kv_v): max_ctx} after probes

# ── KV type pairs ─────────────────────────────────────────────────────────────

KV_PAIRS_TURBO    = [("turbo3", "turbo4"), ("q8_0", "q8_0"), ("f16", "f16")]
KV_PAIRS_OFFICIAL = [("q8_0", "q8_0"), ("f16", "f16")]

# ── Goals ─────────────────────────────────────────────────────────────────────

GOALS  = ["Best Speed", "Max Context", "Balanced", "Best PP"]
DEPTHS = ["Quick", "Full"]

GOAL_DESCRIPTIONS = {
    "Best Speed":  "Maximum TG t/s at a modest fixed context (2048 tokens). Best for chat.",
    "Max Context": "Largest context that fits in VRAM per KV type. Probes hardware limits first.",
    "Balanced":    "Best TG speed at ~50% of max feasible context. Good all-rounder.",
    "Best PP":     "Fastest prompt ingestion. Useful for RAG and long document workflows.",
}

GOAL_METRIC = {
    "Best Speed":  "tg",
    "Max Context": "ctx",
    "Balanced":    "tg",
    "Best PP":     "pp",
}

_SECS_PER_COMBO = 35
_SECS_PER_PROBE = 75   # fit-target runs can take a while


def kv_pairs_for_binary(bench_label: str) -> list:
    """Return KV pairs appropriate for the selected binary."""
    if "turbo" in bench_label.lower():
        return list(KV_PAIRS_TURBO)
    return list(KV_PAIRS_OFFICIAL)


def needs_probe(goal: str) -> bool:
    return goal in ("Max Context", "Balanced")


def _round_ctx(ctx: int) -> int:
    """Round to the nearest standard context size."""
    steps = [512, 1024, 2048, 4096, 6144, 8192, 12288, 16384,
             24576, 32768, 49152, 65536, 98304, 131072]
    return min(steps, key=lambda x: abs(x - ctx))


# ── VRAM probe ────────────────────────────────────────────────────────────────

def _bench_probe_cmd(bench_bin: str, model_path: str, kv_k: str, kv_v: str,
                     threads: int, ngl: int, margin_mib: int) -> str:
    parts = [bench_bin,
             "-m", f'"{model_path}"',
             "-ngl", str(ngl),
             "-ctk", kv_k, "-ctv", kv_v,
             "-fa", "on",
             "--fit-target", str(margin_mib),
             "--fit-ctx", "512",
             "-t", str(threads),
             "-r", "1",
             "-o", "json"]
    return " ".join(parts)


def run_probe(distro: str, user: str, bench_bin: str, model_path: str,
              kv_k: str, kv_v: str, threads: int, ngl: int,
              log_fn: LogFn, margin_mib: int = 512) -> int:
    """
    Run --fit-target probe for one KV pair.
    Returns max feasible context in tokens, or 0 on failure.
    """
    cmd = _bench_probe_cmd(bench_bin, model_path, kv_k, kv_v, threads, ngl, margin_mib)
    buf: list[str] = []

    def _collect(line: str, _=None, _b=buf):
        _b.append(line)
        log_fn(line, None)

    rc = wsl.stream(distro, user, cmd, _collect, timeout=600)
    if rc != 0:
        return 0
    raw = "\n".join(buf)
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if data:
                return int(data[0].get("n_ctx", 0))
    except Exception:
        pass
    return 0


def run_probes(distro: str, user: str, bench_bin: str, model_path: str,
               kv_pairs: list, threads: int, ngl: int,
               log_fn: LogFn, cancel_evt=None,
               margin_mib: int = 512) -> dict:
    """
    Run --fit-target probe for each KV pair.
    Returns {(kv_k, kv_v): max_ctx_tokens}.
    """
    results = {}
    for kv_k, kv_v in kv_pairs:
        if cancel_evt and cancel_evt.is_set():
            break
        log_fn(f"[PROBE] {kv_k}/{kv_v} — finding max context…", "info")
        max_ctx = run_probe(distro, user, bench_bin, model_path,
                            kv_k, kv_v, threads, ngl, log_fn, margin_mib)
        results[(kv_k, kv_v)] = max_ctx
        if max_ctx:
            log_fn(f"[PROBE] {kv_k}/{kv_v}: max {max_ctx:,} tokens", "success")
        else:
            log_fn(f"[PROBE] {kv_k}/{kv_v}: OOM even at minimum context — skipping", "warn")
    return results


# ── Guided combo generation ───────────────────────────────────────────────────

def generate_combos(goal: str, depth: str, kv_pairs: list,
                    probe_results: dict, ngl: int, threads: int) -> list[dict]:
    """
    Generate bench combos for a goal + depth.
    probe_results: {(kv_k, kv_v): max_ctx} — may be empty for non-probing goals.
    """
    is_full = depth == "Full"
    base    = {"ngl": ngl, "threads": threads}

    if goal == "Best Speed":
        ctx     = 2048
        batches = [512, 1024, 2048] if is_full else [512, 1024]
        fa_opts = [True, False]     if is_full else [True]
        combos  = []
        for k, v in kv_pairs:
            for b in batches:
                for fa in fa_opts:
                    combos.append({**base, "ctx": ctx, "batch": b,
                                   "ubatch": min(b, 512), "flash_attn": fa,
                                   "cache_type_k": k, "cache_type_v": v})
        return combos

    if goal == "Max Context":
        fracs  = [0.5, 0.75, 1.0] if is_full else [0.75, 1.0]
        combos = []
        for k, v in kv_pairs:
            max_ctx = probe_results.get((k, v), 0)
            if not max_ctx:
                continue
            seen: set[int] = set()
            for frac in fracs:
                ctx = _round_ctx(int(max_ctx * frac))
                if ctx in seen or ctx < 512:
                    continue
                seen.add(ctx)
                combos.append({**base, "ctx": ctx, "batch": 512, "ubatch": 512,
                               "flash_attn": True,
                               "cache_type_k": k, "cache_type_v": v})
        return combos

    if goal == "Balanced":
        batches = [512, 1024] if is_full else [512]
        fracs   = [0.33, 0.5] if is_full else [0.5]
        combos  = []
        seen_keys: set = set()
        for k, v in kv_pairs:
            max_ctx = probe_results.get((k, v), 0) or 4096
            for frac in fracs:
                ctx = _round_ctx(int(max_ctx * frac))
                ctx = max(ctx, 512)
                for b in batches:
                    key = (k, v, ctx, b)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    combos.append({**base, "ctx": ctx, "batch": b,
                                   "ubatch": min(b, 512), "flash_attn": True,
                                   "cache_type_k": k, "cache_type_v": v})
        return combos

    if goal == "Best PP":
        ctx     = 4096
        batches = [512, 1024, 2048, 4096] if is_full else [512, 1024, 2048]
        combos  = []
        for k, v in kv_pairs:
            for b in batches:
                combos.append({**base, "ctx": ctx, "batch": b,
                               "ubatch": min(b, 512), "flash_attn": True,
                               "cache_type_k": k, "cache_type_v": v})
        return combos

    return []


def guided_combo_count(goal: str, depth: str, kv_pairs: list) -> int:
    """Estimate combo count without running probes."""
    n  = len(kv_pairs)
    fl = depth == "Full"
    if goal == "Best Speed":
        return n * (3 if fl else 2) * (2 if fl else 1)
    if goal == "Max Context":
        return n * (3 if fl else 2)
    if goal == "Balanced":
        return n * (2 if fl else 1) * (2 if fl else 1)
    if goal == "Best PP":
        return n * (4 if fl else 3)
    return 0


# ── Bench command builder ─────────────────────────────────────────────────────

def _bench_cmd(bench_bin: str, model_path: str, combo: dict) -> str:
    parts  = [bench_bin]
    parts += ["-m", f'"{model_path}"']
    parts += ["-ngl", str(combo.get("ngl", 99))]
    parts += ["-p",   str(combo.get("ctx", 512))]
    parts += ["-b",   str(combo.get("batch", 512))]
    parts += ["-ub",  str(combo.get("ubatch", 512))]
    parts += ["-t",   str(combo.get("threads", 4))]
    parts += ["-fa",  "on" if combo.get("flash_attn") else "off"]
    parts += ["-ctk", combo.get("cache_type_k", "f16")]
    parts += ["-ctv", combo.get("cache_type_v", "f16")]
    if combo.get("no_mmap"):
        parts += ["-mmp", "0"]
    if combo.get("cpu_moe"):
        parts += ["-ncmoe", str(combo.get("n_cpu_moe_n", "999"))]
    parts += ["-o", "json"]
    return " ".join(parts)


# ── Output parser ─────────────────────────────────────────────────────────────

def _parse_bench_json(raw: str) -> tuple[float, float] | None:
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return None
        data   = json.loads(match.group())
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


# ── Preflight ─────────────────────────────────────────────────────────────────

def _preflight(distro: str, user: str, bench_bin: str, model_path: str,
               log_fn: LogFn) -> bool:
    chk_bin = wsl.run(distro, user, f"test -x {bench_bin} && echo ok", timeout=10)
    if "ok" not in chk_bin.stdout:
        log_fn(f"[OPT] Bench binary not found or not executable: {bench_bin}", "error")
        log_fn("[OPT] Build llama-bench first: cmake --build build --config Release", "warn")
        return False
    chk_model = wsl.run(distro, user, f'test -f "{model_path}" && echo ok', timeout=10)
    if "ok" not in chk_model.stdout:
        log_fn(f"[OPT] Model file not found: {model_path}", "error")
        return False
    return True


# ── Guided runner ─────────────────────────────────────────────────────────────

def run_guided_bench(distro: str, user: str, bench_bin: str, model_path: str,
                     goal: str, depth: str, kv_pairs: list,
                     ngl: int, threads: int,
                     log_fn: LogFn, done_fn: DoneFn,
                     cancel_evt=None, progress_fn=None,
                     probe_done_fn: ProbeFn | None = None) -> None:
    """
    Orchestrates: preflight → optional VRAM probe → combo generation → bench.
    Runs on a background thread — does NOT call Tk directly.
    """
    if not _preflight(distro, user, bench_bin, model_path, log_fn):
        done_fn([])
        return

    # Probe phase
    probe_results: dict = {}
    if needs_probe(goal):
        log_fn(f"[OPT] Probing VRAM limits for {len(kv_pairs)} KV config(s)…", "info")
        probe_results = run_probes(distro, user, bench_bin, model_path,
                                   kv_pairs, threads, ngl, log_fn, cancel_evt)
        if probe_done_fn:
            probe_done_fn(dict(probe_results))

    if cancel_evt and cancel_evt.is_set():
        log_fn("[OPT] Cancelled.", "warn")
        done_fn([])
        return

    # Generate combos
    combos = generate_combos(goal, depth, kv_pairs, probe_results, ngl, threads)
    if not combos:
        log_fn("[OPT] No viable combos generated.", "error")
        done_fn([])
        return

    log_fn(f"[OPT] Running {len(combos)} benchmark(s)…", "info")
    results: list[dict] = []

    for i, combo in enumerate(combos):
        if cancel_evt and cancel_evt.is_set():
            log_fn(f"[OPT] Cancelled after {i} run(s).", "warn")
            break

        cmd = _bench_cmd(bench_bin, model_path, combo)
        log_fn(f"[OPT] Run {i+1}/{len(combos)}: {combo_summary(combo)}", "info")
        log_fn(f"[OPT] $ {cmd}", None)

        buf: list[str] = []

        def _collect(line: str, _=None, _b=buf):
            _b.append(line)
            log_fn(line, None)

        rc  = wsl.stream(distro, user, cmd, _collect, timeout=600)
        raw = "\n".join(buf)

        if rc != 0:
            log_fn(f"[OPT] Run {i+1} failed (rc={rc}). Skipping.", "warn")
        else:
            parsed = _parse_bench_json(raw)
            if parsed:
                pp_tps, tg_tps = parsed
                results.append({
                    "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    "goal":         goal,
                    "pp":           round(pp_tps, 2),
                    "tg":           round(tg_tps, 2),
                    "ngl":          combo.get("ngl"),
                    "ctx":          combo.get("ctx"),
                    "batch":        combo.get("batch"),
                    "ubatch":       combo.get("ubatch"),
                    "cache_type_k": combo.get("cache_type_k"),
                    "cache_type_v": combo.get("cache_type_v"),
                    "flash_attn":   combo.get("flash_attn", False),
                    "threads":      combo.get("threads"),
                })
                log_fn(f"[OPT] → pp={pp_tps:.1f} t/s  tg={tg_tps:.1f} t/s", "success")
            else:
                log_fn(f"[OPT] Run {i+1}: could not parse JSON output.", "warn")

        if progress_fn:
            progress_fn(i + 1)

    done_fn(results)


# ── Sweep runner ──────────────────────────────────────────────────────────────

def run_bench_combos(distro: str, user: str, bench_bin: str,
                     model_path: str,
                     combos: list[dict],
                     log_fn: LogFn, done_fn: DoneFn,
                     cancel_evt=None, progress_fn=None) -> None:
    """
    Run llama-bench for each combo sequentially (used by Sweep tab).
    Runs on a background thread — does NOT call Tk directly.
    """
    if not _preflight(distro, user, bench_bin, model_path, log_fn):
        done_fn([])
        return
    log_fn("[OPT] Preflight OK — binary and model found.", "success")

    results: list[dict] = []
    for i, combo in enumerate(combos):
        if cancel_evt and cancel_evt.is_set():
            log_fn(f"[OPT] Cancelled after {i} run(s).", "warn")
            break

        cmd = _bench_cmd(bench_bin, model_path, combo)
        log_fn(f"[OPT] Run {i+1}/{len(combos)}: {combo_summary(combo)}", "info")
        log_fn(f"[OPT] $ {cmd}", None)

        buf: list[str] = []

        def _collect(line: str, _=None, _b=buf):
            _b.append(line)
            log_fn(line, None)

        rc  = wsl.stream(distro, user, cmd, _collect, timeout=600)
        raw = "\n".join(buf)

        if rc != 0:
            log_fn(f"[OPT] Run {i+1} failed (rc={rc}). Skipping.", "warn")
        else:
            parsed = _parse_bench_json(raw)
            if parsed:
                pp_tps, tg_tps = parsed
                results.append({
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
                })
                log_fn(f"[OPT] → pp={pp_tps:.1f} t/s  tg={tg_tps:.1f} t/s", "success")
            else:
                log_fn(f"[OPT] Run {i+1}: could not parse JSON output.", "warn")

        if progress_fn:
            progress_fn(i + 1)

    done_fn(results)


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


def sweep_combos_custom(ranges: dict[str, list], base_profile: dict, hw) -> list[dict]:
    if not ranges:
        return []
    keys = list(ranges.keys())
    vals = list(ranges.values())
    combos = []
    for combo_vals in itertools.product(*vals):
        c = {
            "ngl":          base_profile.get("ngl",          99),
            "ctx":          str(base_profile.get("ctx",      "8192")),
            "batch":        base_profile.get("batch",         512),
            "ubatch":       base_profile.get("ubatch",        512),
            "flash_attn":   base_profile.get("flash_attn",   False),
            "cache_type_k": base_profile.get("cache_type_k", "f16"),
            "cache_type_v": base_profile.get("cache_type_v", "f16"),
            "threads":      base_profile.get("threads",        4),
        }
        for k, v in zip(keys, combo_vals):
            c[k] = v
        if int(c.get("ubatch", 512)) > int(c.get("batch", 512)):
            continue
        combos.append(c)
    return combos


def sweep_combos(active_keys: list[str], base_profile: dict, hw) -> list[dict]:
    ranges = {k: _SWEEP_RANGES.get(k, [base_profile.get(k)]) for k in active_keys}
    return sweep_combos_custom(ranges, base_profile, hw)


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


def score_results(results: list[dict], goal: str) -> list[dict]:
    """Return results sorted best-first for the given goal, with 'rank' added."""
    metric = GOAL_METRIC.get(goal, "tg")
    def _key(r):
        if metric == "ctx":
            return int(str(r.get("ctx", 0)))
        return float(r.get(metric, 0) or 0)
    scored = sorted([dict(r) for r in results], key=_key, reverse=True)
    for i, r in enumerate(scored):
        r["rank"] = i + 1
    return scored


def estimated_minutes(n_combos: int, n_probes: int = 0) -> str:
    secs = n_combos * _SECS_PER_COMBO + n_probes * _SECS_PER_PROBE
    if secs < 90:
        return f"~{secs}s"
    return f"~{secs // 60} min"
