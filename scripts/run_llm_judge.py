"""Unified LLM-as-a-judge per-segment classification of CoT reasoning patterns.

Supports three inference providers (gemini, openai, local) with sentence-level
segmentation.  Produces per-trace aggregated results and per-segment raw
labels with character offsets.

Usage
-----
    # Gemini
    python scripts/run_llm_judge.py --provider gemini --config pipeline/code/deepseek-r1-7b

    # OpenAI
    python scripts/run_llm_judge.py --provider openai --config pipeline/code/deepseek-r1-7b

    # Local vLLM
    python scripts/run_llm_judge.py --provider local --config pipeline/code/deepseek-r1-7b

    # All configs
    python scripts/run_llm_judge.py --provider gemini --all

    # Batch ops
    python scripts/run_llm_judge.py --provider gemini --submit --config ...
    python scripts/run_llm_judge.py --provider openai --status
    python scripts/run_llm_judge.py --provider gemini --collect --config ...

    # Local sharding
    python scripts/run_llm_judge.py --provider local --config ... \
        --shard-index 0 --num-shards 4
    python scripts/run_llm_judge.py --provider local --config ... --merge-shards

    # Dry run / test
    python scripts/run_llm_judge.py --provider gemini --config ... --dry-run
    python scripts/run_llm_judge.py --provider openai --config ... --test-n 50
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import list_configs, load_config
from src.llm_judge import (
    CATEGORY_NAMES,
    aggregate_segments,
    build_judge_batch_jsonl,
    build_openai_judge_jsonl,
    build_sentence_windows,
    build_system_prompt,
    collect_judge_results,
    collect_openai_judge_results,
    extract_reasoning_text,
    judge_output_dir as _judge_output_dir,
    load_judge_config,
    load_judge_state,
    parse_judge_batch_results,
    parse_openai_judge_results,
    poll_judge_batch,
    poll_openai_judge_batch,
    prompt_version_hash,
    save_judge_state,
    segment_traces_sentences_batch,
    submit_judge_batch,
    submit_openai_judge_batch,
    _google_client,
    _openai_client,
)


# ---------------------------------------------------------------------------
# Path configuration -- set from --config in main()
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"

DOMAIN = "codeforces"
MODEL_NAME = None
TRACES_PATH = None
JUDGE_OUTPUT_DIR = None
JUDGE_PARQUET_PATH = None
SEGMENTS_PARQUET_PATH = None

# Interrupt flag for graceful Ctrl+C handling (local provider)
_INTERRUPTED = False


def _signal_handler(signum, frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    print("\n  [local-judge] Interrupt received, will checkpoint after "
          "current batch...")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _cache_exists(*paths):
    """Return True if all *paths* exist and are non-empty files."""
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def _ensure_dir(path):
    """Create parent directories for *path* if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _fmt_elapsed(seconds):
    """Format elapsed seconds as ``M:SS`` or ``H:MM:SS``."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _truncate_traces(traces_data, max_segments):
    """Truncate traces_data to at most *max_segments* total segments.

    Returns (truncated_traces_data, actual_segment_count).
    """
    out = []
    count = 0
    for td in traces_data:
        remaining = max_segments - count
        if remaining <= 0:
            break
        segs = td["segments"][:remaining]
        out.append({
            "problem_id": td["problem_id"],
            "run_idx": td["run_idx"],
            "segments": segs,
        })
        count += len(segs)
    return out, count


# ---------------------------------------------------------------------------
# Multi-config context helpers
# ---------------------------------------------------------------------------

def _make_ctx(config_name, judge_config, args):
    """Build a path-context dict for a single config.

    Returns a dict with keys: config_name, domain, model_name, traces_path,
    judge_output_dir, judge_parquet_path, segments_parquet_path.
    """
    cfg = load_config(config_name)
    model_name = cfg["model"]["name"]
    if "math/" in config_name:
        domain = "math"
    elif "sat/" in config_name:
        domain = "sat"
    else:
        domain = "codeforces"

    if args.traces_dir is not None:
        if domain == "math":
            domain_dir = "math"
        elif domain == "sat":
            domain_dir = "sat"
        else:
            domain_dir = "code"
        traces_path = (Path(args.traces_dir) / domain_dir / model_name
                       / "cot_traces.jsonl")
    else:
        traces_path = cfg["paths"]["pipeline"]["cot_traces"]
    cot_analysis_path = cfg["paths"]["analysis"]["cot_analysis"]

    # Determine judge model name for the output dir slug
    provider = args.provider
    if provider == "gemini":
        judge_model = judge_config["judge"]["model"]
        api = True
    elif provider == "openai":
        judge_model = judge_config["judge_openai"]["model"]
        api = True
    else:  # local
        judge_model = (args.judge_model if args.judge_model is not None
                       else judge_config["judge_local"]["model"])
        api = False

    if args.output_dir is not None:
        if domain == "math":
            domain_dir = "math"
        elif domain == "sat":
            domain_dir = "sat"
        else:
            domain_dir = "code"
        model_results_dir = Path(args.output_dir) / domain_dir / model_name
    else:
        model_results_dir = cot_analysis_path.parent.parent

    out_dir = _judge_output_dir(
        model_results_dir, judge_model, api=api)
    return {
        "config_name": config_name,
        "domain": domain,
        "model_name": model_name,
        "traces_path": traces_path,
        "judge_output_dir": out_dir,
        "judge_parquet_path": out_dir / "llm_judge.parquet",
        "segments_parquet_path": (
            out_dir / "llm_judge_segments.parquet"
        ),
    }


def _set_globals_from_ctx(ctx):
    """Set module-level globals from a context dict."""
    global DOMAIN, MODEL_NAME, TRACES_PATH
    global JUDGE_OUTPUT_DIR, JUDGE_PARQUET_PATH, SEGMENTS_PARQUET_PATH
    DOMAIN = ctx["domain"]
    MODEL_NAME = ctx["model_name"]
    TRACES_PATH = ctx["traces_path"]
    JUDGE_OUTPUT_DIR = ctx["judge_output_dir"]
    JUDGE_PARQUET_PATH = ctx["judge_parquet_path"]
    SEGMENTS_PARQUET_PATH = ctx["segments_parquet_path"]


# ---------------------------------------------------------------------------
# Prepare: load traces, extract reasoning, segment
# ---------------------------------------------------------------------------

def _prepare_cache_path(judge_config):
    """Return a deterministic cache path for the current config."""
    seg_cfg = judge_config.get("segmentation", {})
    ctx_size = seg_cfg.get("context_sentences", 1)
    key = f"{MODEL_NAME}_{DOMAIN}_sentence_window_{ctx_size}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return TMP_DIR / f"prepared_traces_{MODEL_NAME}_{DOMAIN}_{h}.json"


def prepare_traces(judge_config):
    """Load traces, extract reasoning text, and segment (sentence-level).

    Caches results in ``tmp/`` so subsequent runs skip the expensive
    extraction and segmentation phase.

    Parameters
    ----------
    judge_config : dict
        Judge configuration.

    Returns
    -------
    (traces_data, total_segments)
        traces_data: list of dicts with problem_id, run_idx, segments
        total_segments: int
    """
    cache_path = _prepare_cache_path(judge_config)
    if cache_path.exists():
        print(f"\n[PREPARE] Loading cached preparation from {cache_path.name}")
        with open(cache_path) as f:
            cached = json.load(f)
        traces_data = cached["traces_data"]
        total_segments = cached["total_segments"]
        avg_segs = total_segments / max(len(traces_data), 1)
        print(f"  {len(traces_data)} traces, {total_segments} segments "
              f"(avg {avg_segs:.1f}/trace)")
        return traces_data, total_segments

    print("\n[PREPARE.1] Loading traces...")
    traces = []
    with open(TRACES_PATH) as f:
        for line in f:
            traces.append(json.loads(line))
    print(f"  Loaded {len(traces)} traces from {TRACES_PATH.name}")

    seg_cfg = judge_config.get("segmentation", {})
    ctx_size = seg_cfg.get("context_sentences", 1)
    print(f"[PREPARE.2] Extracting reasoning text and segmenting "
          f"(sentence_window, context={ctx_size})...")

    traces_data = []
    total_segments = 0

    # Extract all reasoning texts first, then batch-segment with
    # nlp.pipe() for much better throughput.
    print("  Extracting reasoning texts...")
    reasonings = [extract_reasoning_text(tr["trace"], DOMAIN)
                  for tr in traces]
    print(f"  Batch-segmenting {len(reasonings)} traces with "
          "nlp.pipe()...")
    all_sentences = segment_traces_sentences_batch(reasonings)

    for trace_rec, sentences in zip(traces, all_sentences):
        windows = build_sentence_windows(sentences,
                                         context_size=ctx_size)
        segments = []
        for i, w in enumerate(windows):
            segments.append({
                "text": w["context_text"],
                "target_text": w["target_text"],
                "char_start": w["target_char_start"],
                "char_end": w["target_char_end"],
                "segment_idx": i,
            })
        traces_data.append({
            "problem_id": str(trace_rec["problem_id"]),
            "run_idx": int(trace_rec["run_idx"]),
            "segments": segments,
        })
        total_segments += len(segments)

    avg_segs = total_segments / max(len(traces_data), 1)
    print(f"  Segmented {len(traces_data)} traces -> "
          f"{total_segments} segments (avg {avg_segs:.1f}/trace)")

    # Cache for next run
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"traces_data": traces_data,
                    "total_segments": total_segments}, f,
                   ensure_ascii=False)
    print(f"  Cached preparation to {cache_path.name}")

    return traces_data, total_segments


def _build_prompt(judge_config, domain):
    """Build the system prompt for sentence-level classification."""
    return build_system_prompt(judge_config, domain)


# ---------------------------------------------------------------------------
# Shared parquet writing
# ---------------------------------------------------------------------------

def _write_final_parquets(
    segment_df, segments_info, output_dir, judge_config,
    no_segments, provider, prompt_ver="unknown",
):
    """Build DataFrames and write llm_judge.parquet (+ optional segments).

    Parameters
    ----------
    segment_df : pd.DataFrame
        Per-segment results (problem_id, run_idx, segment_idx, + categories).
    segments_info : dict
        Mapping ``(problem_id, run_idx)`` -> total number of segments.
    output_dir : Path
        Output directory.
    judge_config : dict
        Judge configuration.
    no_segments : bool
        Skip segment-level parquet.
    provider : str
        ``"gemini"``, ``"openai"``, or ``"local"``.
    prompt_ver : str
        Prompt version hash.
    """
    if segment_df.empty:
        print("  WARNING: No records to write.")
        return

    # Load segments_info.json for char offsets and segment text
    seg_info_path = output_dir / "segments_info.json"
    segments_lookup = {}
    if seg_info_path.exists():
        with open(seg_info_path) as f:
            seg_info_list = json.load(f)
        for si in seg_info_list:
            key = (str(si["problem_id"]), int(si["run_idx"]))
            if key not in segments_info:
                segments_info[key] = si["num_segments"]
            for seg in si["segments"]:
                seg_key = (str(si["problem_id"]),
                           int(si["run_idx"]),
                           seg["segment_idx"])
                segments_lookup[seg_key] = seg

    # Add char offsets and text
    char_starts = []
    char_ends = []
    seg_texts = []
    for _, row in segment_df.iterrows():
        seg_key = (str(row["problem_id"]),
                   int(row["run_idx"]),
                   int(row["segment_idx"]))
        seg = segments_lookup.get(seg_key, {})
        char_starts.append(seg.get("char_start", -1))
        char_ends.append(seg.get("char_end", -1))
        seg_texts.append(seg.get("target_text", seg.get("text", "")))

    segment_df["char_start"] = char_starts
    segment_df["char_end"] = char_ends
    segment_df["segment_text"] = seg_texts

    # Save segment-level parquet
    segments_parquet = output_dir / "llm_judge_segments.parquet"
    if not no_segments:
        seg_cols = (["problem_id", "run_idx", "segment_idx",
                     "char_start", "char_end", "segment_text"]
                    + CATEGORY_NAMES)
        seg_out = segment_df[[c for c in seg_cols
                              if c in segment_df.columns]]
        seg_out.to_parquet(segments_parquet, index=False)
        print(f"  Saved {segments_parquet.name} "
              f"({len(seg_out)} rows, {len(seg_out.columns)} cols)")

    # Aggregate to per-trace
    print("[FINAL] Aggregating to per-trace summary...")
    trace_df = aggregate_segments(segment_df, segments_info)
    print(f"  Aggregated {len(trace_df)} traces")

    # Determine judge model name and seg config for metadata
    if provider == "gemini":
        judge_model = judge_config["judge"]["model"]
        seg_cfg = judge_config["segmentation"]
    elif provider == "openai":
        judge_model = judge_config["judge_openai"]["model"]
        seg_cfg = judge_config["segmentation"]
    else:
        judge_model = judge_config["judge_local"]["model"]
        seg_cfg = judge_config["segmentation"]

    # Save with metadata
    table = pa.Table.from_pandas(trace_df)
    meta = table.schema.metadata or {}
    meta[b"judge_model"] = judge_model.encode()
    meta[b"judge_provider"] = provider.encode()
    meta[b"judge_prompt_version"] = prompt_ver.encode()
    meta[b"segmentation_config"] = json.dumps(seg_cfg).encode()
    meta[b"segmentation_mode"] = b"sentence_window"

    table = table.replace_schema_metadata(meta)
    judge_parquet = output_dir / "llm_judge.parquet"
    _ensure_dir(judge_parquet)
    pq.write_table(table, judge_parquet)
    print(f"  Saved {judge_parquet.name} "
          f"({len(trace_df)} rows, {len(trace_df.columns)} cols)")

    # Print summary
    print("\n  Detection rates (fraction of traces with pattern):")
    for cat in CATEGORY_NAMES:
        col = f"judge_{cat}_present"
        if col in trace_df.columns:
            rate = trace_df[col].mean()
            print(f"    {cat:<28s} {rate:.3f}")


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

def _gemini_run_submit(judge_config, force=False, test_n=None):
    """Prepare traces, create cache, build JSONL, submit Gemini batch."""
    t0 = time.time()

    state = load_judge_state(JUDGE_OUTPUT_DIR)
    if state is not None and not force:
        print(f"  Active batch found: {state['batch_id'][:30]}... "
              f"(status={state['status']})")
        print("  Use --force to resubmit.")
        return

    traces_data, total_segments = prepare_traces(judge_config)
    if total_segments == 0:
        print("  No segments to classify. Skipping.")
        return

    if test_n is not None:
        traces_data, total_segments = _truncate_traces(traces_data, test_n)
        print(f"  [TEST MODE] Truncated to {total_segments} segments")

    system_prompt = _build_prompt(judge_config, DOMAIN)
    version = prompt_version_hash(system_prompt)
    print(f"\n[SUBMIT.1] System prompt ready "
          f"(version={version}, {len(system_prompt.split())} words)")

    client = _google_client()
    model = judge_config["judge"]["model"]

    print(f"[SUBMIT.2] Building batch JSONL ({total_segments} requests)...")
    batch_parts = build_judge_batch_jsonl(
        traces_data, judge_config, JUDGE_OUTPUT_DIR, system_prompt,
    )
    total_requests = sum(bp["num_requests"] for bp in batch_parts)
    print(f"  Wrote {total_requests} requests across "
          f"{len(batch_parts)} batch file(s)")

    model_id = model
    if model_id.startswith("models/"):
        model_id = model_id[len("models/"):]

    print(f"[SUBMIT.3] Submitting {len(batch_parts)} batch job(s)...")
    sub_batches = []
    merged_custom_id_map = {}
    for i, bp in enumerate(batch_parts):
        suffix = (f" (part {i+1}/{len(batch_parts)})"
                  if len(batch_parts) > 1 else "")
        print(f"  Submitting {bp['num_requests']} requests{suffix}...")
        batch_state = submit_judge_batch(
            client, bp["jsonl_path"], model_id,
            display_name=f"judge/{MODEL_NAME}/{DOMAIN}{suffix}",
        )
        sub_batches.append(batch_state)
        merged_custom_id_map.update(bp["custom_id_map"])

    state = {
        "provider": "gemini",
        "model_name": MODEL_NAME,
        "domain": DOMAIN,
        "segmentation": "sentence_window",
        "num_requests": total_requests,
        "prompt_version": version,
        "custom_id_map": merged_custom_id_map,
        "sub_batches": sub_batches,
        "batch_id": sub_batches[0]["batch_id"],
        "status": "submitted",
        "poll_count": 0,
        "error": None,
    }
    save_judge_state(state, JUDGE_OUTPUT_DIR)

    # Save segments info for later aggregation
    seg_info_path = JUDGE_OUTPUT_DIR / "segments_info.json"
    seg_info = []
    for td in traces_data:
        seg_info.append({
            "problem_id": td["problem_id"],
            "run_idx": td["run_idx"],
            "num_segments": len(td["segments"]),
            "segments": td["segments"],
        })
    with open(seg_info_path, "w") as f:
        json.dump(seg_info, f, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Batch submitted: {state['batch_id'][:30]}...")
    print(f"  Requests: {total_requests}")
    print(f"  Submit time: {_fmt_elapsed(elapsed)}")


def _gemini_run_poll(judge_config):
    """Poll active Gemini batch(es) until all complete."""
    state = load_judge_state(JUDGE_OUTPUT_DIR)
    if state is None:
        print("  No active batch to poll.")
        return state

    if state["status"] not in ("submitted", "polling"):
        print(f"  Batch status is '{state['status']}', nothing to poll.")
        return state

    poll_interval = judge_config["batch"]["poll_interval"]
    client = _google_client()
    max_polls = (25 * 3600) // poll_interval
    sub_batches = state.get("sub_batches", [state])

    print(f"\n[POLL] Polling {len(sub_batches)} batch(es) every "
          f"{poll_interval}s (Ctrl+C to stop, re-run to resume)...")

    while state["status"] in ("submitted", "polling"):
        if state["poll_count"] >= max_polls:
            state["status"] = "failed"
            state["error"] = (
                f"Poll timeout after {state['poll_count']} polls "
                f"({state['poll_count'] * poll_interval / 3600:.1f}h)"
            )
            save_judge_state(state, JUDGE_OUTPUT_DIR)
            break

        time.sleep(poll_interval)
        state["poll_count"] += 1

        all_done = True
        any_failed = False
        for sb in sub_batches:
            if sb["status"] in ("collecting", "completed"):
                continue
            sb = poll_judge_batch(client, sb)
            if sb["status"] in ("submitted", "polling"):
                all_done = False
            elif sb["status"] == "failed":
                any_failed = True

        if any_failed:
            failed = [sb for sb in sub_batches if sb["status"] == "failed"]
            state["status"] = "failed"
            state["error"] = (
                f"{len(failed)} sub-batch(es) failed: "
                + "; ".join(sb.get("error", "?") for sb in failed)
            )
        elif all_done:
            state["status"] = "collecting"

        state["sub_batches"] = sub_batches
        save_judge_state(state, JUDGE_OUTPUT_DIR)

    return state


def _gemini_run_collect(judge_config, no_segments=False, force=False):
    """Collect Gemini batch results and produce output parquets."""
    t0 = time.time()

    state_path = JUDGE_OUTPUT_DIR / "judge_batch_state.json"
    if not state_path.exists():
        print("  No batch state found. Run --submit first.")
        return

    with open(state_path) as f:
        state = json.load(f)

    if state["status"] not in ("collecting", "completed"):
        print(f"  Batch status is '{state['status']}', cannot collect yet.")
        if state["status"] in ("submitted", "polling"):
            print("  Run without --collect to poll, or wait and try again.")
        return

    if (not force and state["status"] == "completed"
            and _cache_exists(JUDGE_PARQUET_PATH)):
        print(f"  SKIP (cached): {JUDGE_PARQUET_PATH.name}")
        print("  Use --force to recollect.")
        return

    custom_id_map = state["custom_id_map"]
    prompt_ver = state.get("prompt_version", "unknown")

    print("\n[COLLECT.1] Downloading batch results...")
    client = _google_client()
    sub_batches = state.get("sub_batches", [state])
    output_parts = []
    for i, sb in enumerate(sub_batches):
        suffix = (f" (part {i+1}/{len(sub_batches)})"
                  if len(sub_batches) > 1 else "")
        part = collect_judge_results(client, sb)
        output_parts.append(part)
        print(f"  Downloaded{suffix}: {len(part)} chars")
    output_text = "\n".join(output_parts)

    print("[COLLECT.2] Parsing structured JSON responses...")
    segment_df = parse_judge_batch_results(output_text, custom_id_map)
    print(f"  Parsed {len(segment_df)} segment results")

    # Load segments_info for aggregation
    seg_info_path = JUDGE_OUTPUT_DIR / "segments_info.json"
    segments_info = {}
    if seg_info_path.exists():
        with open(seg_info_path) as f:
            seg_info_list = json.load(f)
        for si in seg_info_list:
            key = (str(si["problem_id"]), int(si["run_idx"]))
            segments_info[key] = si["num_segments"]

    _write_final_parquets(
        segment_df, segments_info, JUDGE_OUTPUT_DIR, judge_config,
        no_segments, "gemini", prompt_ver,
    )

    state["status"] = "completed"
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_judge_state(state, JUDGE_OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n  Collection complete in {_fmt_elapsed(elapsed)}")


def _gemini_run_status():
    """Scan for active Gemini judge batch state files and print status."""
    results_base = PROJECT_ROOT / "data" / "results"
    found = []
    for state_path in sorted(results_base.rglob("judge_batch_state.json")):
        with open(state_path) as f:
            state = json.load(f)
        if state.get("provider") not in (None, "gemini"):
            continue
        found.append((state_path.parent, state))

    if not found:
        print("No active or completed Gemini judge batches found.")
        return

    print(f"\n{'Model/Domain':<35} {'Status':<14} "
          f"{'Requests':>9} {'Polls':>6}")
    print("-" * 68)
    for output_dir, state in found:
        label = f"{state.get('model_name', '?')}/{state.get('domain', '?')}"
        print(
            f"{label:<35} {state['status']:<14} "
            f"{state.get('num_requests', 0):>9} "
            f"{state.get('poll_count', 0):>6}"
        )
        if state.get("error"):
            print(f"  ERROR: {state['error']}")

    print(f"\nTotal: {len(found)} batch(es)")


def _gemini_poll_with_ctx(ctx, judge_config, stop_event=None):
    """Poll active Gemini batch(es) until complete, using explicit ctx."""
    output_dir = ctx["judge_output_dir"]
    label = f"{ctx['model_name']}/{ctx['domain']}"

    state = load_judge_state(output_dir)
    if state is None:
        print(f"  [{label}] No active batch to poll.")
        return state

    if state["status"] not in ("submitted", "polling"):
        print(f"  [{label}] Batch status is '{state['status']}', "
              "nothing to poll.")
        return state

    poll_interval = judge_config["batch"]["poll_interval"]
    client = _google_client()
    max_polls = (25 * 3600) // poll_interval
    sub_batches = state.get("sub_batches", [state])

    print(f"  [{label}] Polling {len(sub_batches)} batch(es) every "
          f"{poll_interval}s...")

    while state["status"] in ("submitted", "polling"):
        if stop_event is not None and stop_event.is_set():
            print(f"  [{label}] Shutdown requested, stopping poll.")
            return state

        if state["poll_count"] >= max_polls:
            state["status"] = "failed"
            state["error"] = (
                f"Poll timeout after {state['poll_count']} polls "
                f"({state['poll_count'] * poll_interval / 3600:.1f}h)"
            )
            save_judge_state(state, output_dir)
            break

        if stop_event is not None:
            stop_event.wait(poll_interval)
            if stop_event.is_set():
                print(f"  [{label}] Shutdown requested, stopping poll.")
                return state
        else:
            time.sleep(poll_interval)

        state["poll_count"] += 1

        all_done = True
        any_failed = False
        for sb in sub_batches:
            if sb["status"] in ("collecting", "completed"):
                continue
            sb = poll_judge_batch(client, sb)
            if sb["status"] in ("submitted", "polling"):
                all_done = False
            elif sb["status"] == "failed":
                any_failed = True

        if any_failed:
            failed = [sb for sb in sub_batches if sb["status"] == "failed"]
            state["status"] = "failed"
            state["error"] = (
                f"{len(failed)} sub-batch(es) failed: "
                + "; ".join(sb.get("error", "?") for sb in failed)
            )
        elif all_done:
            state["status"] = "collecting"

        state["sub_batches"] = sub_batches
        save_judge_state(state, output_dir)

    return state


def _gemini_collect_with_ctx(ctx, judge_config, no_segments=False,
                             force=False):
    """Collect Gemini batch results using explicit ctx paths."""
    t0 = time.time()
    output_dir = ctx["judge_output_dir"]
    parquet_path = ctx["judge_parquet_path"]
    label = f"{ctx['model_name']}/{ctx['domain']}"

    state_path = output_dir / "judge_batch_state.json"
    if not state_path.exists():
        print(f"  [{label}] No batch state found. Run --submit first.")
        return

    with open(state_path) as f:
        state = json.load(f)

    if state["status"] not in ("collecting", "completed"):
        print(f"  [{label}] Batch status is '{state['status']}', "
              "cannot collect yet.")
        return

    if (not force and state["status"] == "completed"
            and _cache_exists(parquet_path)):
        print(f"  [{label}] SKIP (cached): {parquet_path.name}")
        return

    custom_id_map = state["custom_id_map"]
    prompt_ver = state.get("prompt_version", "unknown")

    print(f"  [{label}] Downloading batch results...")
    client = _google_client()
    sub_batches = state.get("sub_batches", [state])
    output_parts = []
    for i, sb in enumerate(sub_batches):
        suffix = (f" (part {i+1}/{len(sub_batches)})"
                  if len(sub_batches) > 1 else "")
        part = collect_judge_results(client, sb)
        output_parts.append(part)
        print(f"  [{label}] Downloaded{suffix}: {len(part)} chars")
    output_text = "\n".join(output_parts)

    print(f"  [{label}] Parsing structured JSON responses...")
    segment_df = parse_judge_batch_results(output_text, custom_id_map)
    print(f"  [{label}] Parsed {len(segment_df)} segment results")

    # Load segments_info for aggregation
    seg_info_path = output_dir / "segments_info.json"
    segments_info = {}
    if seg_info_path.exists():
        with open(seg_info_path) as f:
            seg_info_list = json.load(f)
        for si in seg_info_list:
            key = (str(si["problem_id"]), int(si["run_idx"]))
            segments_info[key] = si["num_segments"]

    _write_final_parquets(
        segment_df, segments_info, output_dir, judge_config,
        no_segments, "gemini", prompt_ver,
    )

    state["status"] = "completed"
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_judge_state(state, output_dir)

    elapsed = time.time() - t0
    print(f"  [{label}] Collection complete in {_fmt_elapsed(elapsed)}")


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

def _openai_run_submit(judge_config, force=False, test_n=None):
    """Prepare traces, build JSONL, submit OpenAI batch."""
    t0 = time.time()

    state = load_judge_state(JUDGE_OUTPUT_DIR)
    if state is not None and not force:
        print(f"  Active batch found: {state['batch_id'][:30]}... "
              f"(status={state['status']})")
        print("  Use --force to resubmit.")
        return

    traces_data, total_segments = prepare_traces(judge_config)
    if total_segments == 0:
        print("  No segments to classify. Skipping.")
        return

    if test_n is not None:
        traces_data, total_segments = _truncate_traces(traces_data, test_n)
        print(f"  [TEST MODE] Truncated to {total_segments} segments")

    system_prompt = _build_prompt(judge_config, DOMAIN)
    version = prompt_version_hash(system_prompt)
    print(f"\n[SUBMIT.1] System prompt ready "
          f"(version={version}, {len(system_prompt.split())} words)")

    client = _openai_client()

    print(f"[SUBMIT.2] Building batch JSONL ({total_segments} requests)...")
    batch_parts = build_openai_judge_jsonl(
        traces_data, judge_config, JUDGE_OUTPUT_DIR, system_prompt,
    )
    total_requests = sum(bp["num_requests"] for bp in batch_parts)
    print(f"  Wrote {total_requests} requests across "
          f"{len(batch_parts)} batch file(s)")

    print(f"[SUBMIT.3] Submitting {len(batch_parts)} batch job(s)...")
    sub_batches = []
    merged_custom_id_map = {}
    for i, bp in enumerate(batch_parts):
        suffix = (f" (part {i+1}/{len(batch_parts)})"
                  if len(batch_parts) > 1 else "")
        print(f"  Submitting {bp['num_requests']} requests{suffix}...")
        batch_state = submit_openai_judge_batch(
            client, bp["jsonl_path"],
            display_name=f"judge-openai/{MODEL_NAME}/{DOMAIN}{suffix}",
        )
        sub_batches.append(batch_state)
        merged_custom_id_map.update(bp["custom_id_map"])

    state = {
        "provider": "openai",
        "model_name": MODEL_NAME,
        "domain": DOMAIN,
        "segmentation": "sentence_window",
        "num_requests": total_requests,
        "prompt_version": version,
        "custom_id_map": merged_custom_id_map,
        "sub_batches": sub_batches,
        "batch_id": sub_batches[0]["batch_id"],
        "status": "submitted",
        "poll_count": 0,
        "error": None,
    }
    save_judge_state(state, JUDGE_OUTPUT_DIR)

    seg_info_path = JUDGE_OUTPUT_DIR / "segments_info.json"
    seg_info = []
    for td in traces_data:
        seg_info.append({
            "problem_id": td["problem_id"],
            "run_idx": td["run_idx"],
            "num_segments": len(td["segments"]),
            "segments": td["segments"],
        })
    with open(seg_info_path, "w") as f:
        json.dump(seg_info, f, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Batch submitted: {state['batch_id'][:30]}...")
    print(f"  Requests: {total_requests}")
    print(f"  Submit time: {_fmt_elapsed(elapsed)}")


def _openai_run_poll(judge_config):
    """Poll active OpenAI batch(es) until all complete."""
    state = load_judge_state(JUDGE_OUTPUT_DIR)
    if state is None:
        print("  No active batch to poll.")
        return state

    if state["status"] not in ("submitted", "polling"):
        print(f"  Batch status is '{state['status']}', nothing to poll.")
        return state

    poll_interval = judge_config["batch_openai"]["poll_interval"]
    client = _openai_client()
    max_polls = (25 * 3600) // poll_interval
    sub_batches = state.get("sub_batches", [state])

    print(f"\n[POLL] Polling {len(sub_batches)} batch(es) every "
          f"{poll_interval}s (Ctrl+C to stop, re-run to resume)...")

    while state["status"] in ("submitted", "polling"):
        if state["poll_count"] >= max_polls:
            state["status"] = "failed"
            state["error"] = (
                f"Poll timeout after {state['poll_count']} polls "
                f"({state['poll_count'] * poll_interval / 3600:.1f}h)"
            )
            save_judge_state(state, JUDGE_OUTPUT_DIR)
            break

        time.sleep(poll_interval)
        state["poll_count"] += 1

        all_done = True
        any_failed = False
        for sb in sub_batches:
            if sb["status"] in ("collecting", "completed"):
                continue
            sb = poll_openai_judge_batch(client, sb)
            if sb["status"] in ("submitted", "polling"):
                all_done = False
            elif sb["status"] == "failed":
                any_failed = True

        if any_failed:
            failed = [sb for sb in sub_batches if sb["status"] == "failed"]
            state["status"] = "failed"
            state["error"] = (
                f"{len(failed)} sub-batch(es) failed: "
                + "; ".join(sb.get("error", "?") for sb in failed)
            )
        elif all_done:
            state["status"] = "collecting"

        state["sub_batches"] = sub_batches
        save_judge_state(state, JUDGE_OUTPUT_DIR)

    return state


def _openai_run_collect(judge_config, no_segments=False, force=False):
    """Collect OpenAI batch results and produce output parquets."""
    t0 = time.time()

    state_path = JUDGE_OUTPUT_DIR / "judge_batch_state.json"
    if not state_path.exists():
        print("  No batch state found. Run --submit first.")
        return

    with open(state_path) as f:
        state = json.load(f)

    if state["status"] not in ("collecting", "completed"):
        print(f"  Batch status is '{state['status']}', cannot collect yet.")
        if state["status"] in ("submitted", "polling"):
            print("  Run without --collect to poll, or wait and try again.")
        return

    if (not force and state["status"] == "completed"
            and _cache_exists(JUDGE_PARQUET_PATH)):
        print(f"  SKIP (cached): {JUDGE_PARQUET_PATH.name}")
        print("  Use --force to recollect.")
        return

    custom_id_map = state["custom_id_map"]
    prompt_ver = state.get("prompt_version", "unknown")

    print("\n[COLLECT.1] Downloading batch results...")
    client = _openai_client()
    sub_batches = state.get("sub_batches", [state])
    output_parts = []
    for i, sb in enumerate(sub_batches):
        suffix = (f" (part {i+1}/{len(sub_batches)})"
                  if len(sub_batches) > 1 else "")
        part = collect_openai_judge_results(client, sb)
        output_parts.append(part)
        print(f"  Downloaded{suffix}: {len(part)} chars")
    output_text = "\n".join(output_parts)

    print("[COLLECT.2] Parsing structured JSON responses...")
    segment_df = parse_openai_judge_results(output_text, custom_id_map)
    print(f"  Parsed {len(segment_df)} segment results")

    seg_info_path = JUDGE_OUTPUT_DIR / "segments_info.json"
    segments_info = {}
    if seg_info_path.exists():
        with open(seg_info_path) as f:
            seg_info_list = json.load(f)
        for si in seg_info_list:
            key = (str(si["problem_id"]), int(si["run_idx"]))
            segments_info[key] = si["num_segments"]

    _write_final_parquets(
        segment_df, segments_info, JUDGE_OUTPUT_DIR, judge_config,
        no_segments, "openai", prompt_ver,
    )

    state["status"] = "completed"
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_judge_state(state, JUDGE_OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n  Collection complete in {_fmt_elapsed(elapsed)}")


def _openai_run_status():
    """Scan for active OpenAI judge batch state files and print status."""
    results_base = PROJECT_ROOT / "data" / "results"
    found = []
    for state_path in sorted(results_base.rglob("judge_batch_state.json")):
        with open(state_path) as f:
            state = json.load(f)
        if state.get("provider") != "openai":
            continue
        found.append((state_path.parent, state))

    if not found:
        print("No active or completed OpenAI judge batches found.")
        return

    print(f"\n{'Model/Domain':<35} {'Status':<14} "
          f"{'Requests':>9} {'Polls':>6}")
    print("-" * 68)
    for output_dir, state in found:
        label = f"{state.get('model_name', '?')}/{state.get('domain', '?')}"
        print(
            f"{label:<35} {state['status']:<14} "
            f"{state.get('num_requests', 0):>9} "
            f"{state.get('poll_count', 0):>6}"
        )
        if state.get("error"):
            print(f"  ERROR: {state['error']}")

    print(f"\nTotal: {len(found)} batch(es)")


def _openai_poll_with_ctx(ctx, judge_config, stop_event=None):
    """Poll active OpenAI batch(es) until complete, using explicit ctx."""
    output_dir = ctx["judge_output_dir"]
    label = f"{ctx['model_name']}/{ctx['domain']}"

    state = load_judge_state(output_dir)
    if state is None:
        print(f"  [{label}] No active batch to poll.")
        return state

    if state["status"] not in ("submitted", "polling"):
        print(f"  [{label}] Batch status is '{state['status']}', "
              "nothing to poll.")
        return state

    poll_interval = judge_config["batch_openai"]["poll_interval"]
    client = _openai_client()
    max_polls = (25 * 3600) // poll_interval
    sub_batches = state.get("sub_batches", [state])

    print(f"  [{label}] Polling {len(sub_batches)} batch(es) every "
          f"{poll_interval}s...")

    while state["status"] in ("submitted", "polling"):
        if stop_event is not None and stop_event.is_set():
            print(f"  [{label}] Shutdown requested, stopping poll.")
            return state

        if state["poll_count"] >= max_polls:
            state["status"] = "failed"
            state["error"] = (
                f"Poll timeout after {state['poll_count']} polls "
                f"({state['poll_count'] * poll_interval / 3600:.1f}h)"
            )
            save_judge_state(state, output_dir)
            break

        if stop_event is not None:
            stop_event.wait(poll_interval)
            if stop_event.is_set():
                print(f"  [{label}] Shutdown requested, stopping poll.")
                return state
        else:
            time.sleep(poll_interval)

        state["poll_count"] += 1

        all_done = True
        any_failed = False
        for sb in sub_batches:
            if sb["status"] in ("collecting", "completed"):
                continue
            sb = poll_openai_judge_batch(client, sb)
            if sb["status"] in ("submitted", "polling"):
                all_done = False
            elif sb["status"] == "failed":
                any_failed = True

        if any_failed:
            failed = [sb for sb in sub_batches if sb["status"] == "failed"]
            state["status"] = "failed"
            state["error"] = (
                f"{len(failed)} sub-batch(es) failed: "
                + "; ".join(sb.get("error", "?") for sb in failed)
            )
        elif all_done:
            state["status"] = "collecting"

        state["sub_batches"] = sub_batches
        save_judge_state(state, output_dir)

    return state


def _openai_collect_with_ctx(ctx, judge_config, no_segments=False,
                             force=False):
    """Collect OpenAI batch results using explicit ctx paths."""
    t0 = time.time()
    output_dir = ctx["judge_output_dir"]
    parquet_path = ctx["judge_parquet_path"]
    label = f"{ctx['model_name']}/{ctx['domain']}"

    state_path = output_dir / "judge_batch_state.json"
    if not state_path.exists():
        print(f"  [{label}] No batch state found. Run --submit first.")
        return

    with open(state_path) as f:
        state = json.load(f)

    if state["status"] not in ("collecting", "completed"):
        print(f"  [{label}] Batch status is '{state['status']}', "
              "cannot collect yet.")
        return

    if (not force and state["status"] == "completed"
            and _cache_exists(parquet_path)):
        print(f"  [{label}] SKIP (cached): {parquet_path.name}")
        return

    custom_id_map = state["custom_id_map"]
    prompt_ver = state.get("prompt_version", "unknown")

    print(f"  [{label}] Downloading batch results...")
    client = _openai_client()
    sub_batches = state.get("sub_batches", [state])
    output_parts = []
    for i, sb in enumerate(sub_batches):
        suffix = (f" (part {i+1}/{len(sub_batches)})"
                  if len(sub_batches) > 1 else "")
        part = collect_openai_judge_results(client, sb)
        output_parts.append(part)
        print(f"  [{label}] Downloaded{suffix}: {len(part)} chars")
    output_text = "\n".join(output_parts)

    print(f"  [{label}] Parsing structured JSON responses...")
    segment_df = parse_openai_judge_results(output_text, custom_id_map)
    print(f"  [{label}] Parsed {len(segment_df)} segment results")

    seg_info_path = output_dir / "segments_info.json"
    segments_info = {}
    if seg_info_path.exists():
        with open(seg_info_path) as f:
            seg_info_list = json.load(f)
        for si in seg_info_list:
            key = (str(si["problem_id"]), int(si["run_idx"]))
            segments_info[key] = si["num_segments"]

    _write_final_parquets(
        segment_df, segments_info, output_dir, judge_config,
        no_segments, "openai", prompt_ver,
    )

    state["status"] = "completed"
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_judge_state(state, output_dir)

    elapsed = time.time() - t0
    print(f"  [{label}] Collection complete in {_fmt_elapsed(elapsed)}")


# ---------------------------------------------------------------------------
# Local vLLM provider
# ---------------------------------------------------------------------------

def _local_save_state(state, state_path):
    """Atomic write of state JSON file."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=state_path.parent, prefix=".state_", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, state_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _local_load_state(state_path):
    """Load state JSON file, return None if not found."""
    if not state_path.exists():
        return None
    with open(state_path) as f:
        return json.load(f)


def _local_load_checkpoint(checkpoint_path):
    """Load checkpoint.jsonl and return (records, completed_set)."""
    records = []
    completed = set()
    if not checkpoint_path.exists():
        return records, completed

    with open(checkpoint_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                import warnings
                warnings.warn(
                    f"Skipping corrupted checkpoint line {line_no} "
                    f"in {checkpoint_path.name} (truncated write?)"
                )
                continue
            records.append(rec)
            completed.add((
                str(rec["problem_id"]),
                int(rec["run_idx"]),
                int(rec["segment_idx"]),
            ))

    return records, completed


def _local_append_checkpoint(records, checkpoint_path):
    """Append records to checkpoint.jsonl."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _local_flatten_segments(traces_data):
    """Flatten traces_data into a flat list of segment dicts."""
    flat = []
    for td in traces_data:
        for seg in td["segments"]:
            flat.append({
                "problem_id": td["problem_id"],
                "run_idx": td["run_idx"],
                "segment_idx": seg["segment_idx"],
                "text": seg["text"],
                "char_start": seg.get("char_start", -1),
                "char_end": seg.get("char_end", -1),
                "target_text": seg.get("target_text", ""),
            })
    return flat


def _local_run_inference(judge_config, shard_index=None, num_shards=None,
                         no_segments=False, force=False):
    """Run local vLLM judge inference with checkpoint/resume support."""
    global _INTERRUPTED
    t0 = time.time()

    # Install signal handler early (before model loading)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    local_cfg = judge_config["judge_local"]
    judge_model = local_cfg["model"]
    batch_size = local_cfg["batch_size"]

    # Determine shard directory
    if num_shards is not None and num_shards > 1:
        shard_dir = JUDGE_OUTPUT_DIR / f"shard_{shard_index:03d}"
    else:
        shard_dir = JUDGE_OUTPUT_DIR
        shard_index = None
        num_shards = None

    state_path = shard_dir / "judge_local_state.json"
    checkpoint_path = shard_dir / "checkpoint.jsonl"

    # Check for completed state
    state = _local_load_state(state_path)
    if state is not None and state.get("status") == "completed" and not force:
        print(f"  SKIP (completed): shard {shard_index}")
        return

    # Prepare traces
    traces_data, total_segments = prepare_traces(judge_config)
    if total_segments == 0:
        print("  No segments to classify. Skipping.")
        return

    # Flatten all segments
    all_segments = _local_flatten_segments(traces_data)

    # Apply sharding
    if num_shards is not None and num_shards > 1:
        shard_size = len(all_segments) // num_shards
        start = shard_index * shard_size
        if shard_index == num_shards - 1:
            end = len(all_segments)
        else:
            end = start + shard_size
        shard_segments = all_segments[start:end]
        print(f"\n[SHARD] Shard {shard_index}/{num_shards}: "
              f"segments {start}-{end-1} ({len(shard_segments)} total)")
    else:
        shard_segments = all_segments

    # Save segments_info for this shard (needed for merge)
    shard_dir.mkdir(parents=True, exist_ok=True)
    seen_traces = {}
    for seg in shard_segments:
        key = (seg["problem_id"], seg["run_idx"])
        if key not in seen_traces:
            seen_traces[key] = {
                "problem_id": seg["problem_id"],
                "run_idx": seg["run_idx"],
                "num_segments": 0,
                "segments": [],
            }
        seen_traces[key]["num_segments"] += 1
        seg_entry = {
            "segment_idx": seg["segment_idx"],
            "char_start": seg["char_start"],
            "char_end": seg["char_end"],
            "text": seg["text"],
        }
        if seg.get("target_text"):
            seg_entry["target_text"] = seg["target_text"]
        seen_traces[key]["segments"].append(seg_entry)
    seg_info = list(seen_traces.values())
    seg_info_path = shard_dir / "segments_info.json"
    with open(seg_info_path, "w") as f:
        json.dump(seg_info, f, ensure_ascii=False)

    # Load checkpoint
    if force:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        records = []
        completed = set()
    else:
        records, completed = _local_load_checkpoint(checkpoint_path)

    if completed:
        print(f"  [resume] Loaded {len(completed)} completed segments "
              f"from checkpoint")

    # Filter remaining segments
    remaining = [
        seg for seg in shard_segments
        if (seg["problem_id"], seg["run_idx"], seg["segment_idx"])
        not in completed
    ]

    if not remaining:
        print("  All segments already completed.")
    else:
        # Build system prompt
        system_prompt = _build_prompt(judge_config, DOMAIN)
        version = prompt_version_hash(system_prompt)
        print(f"\n[INFER] System prompt ready "
              f"(version={version}, {len(system_prompt.split())} words)")
        print(f"  Remaining: {len(remaining)} segments "
              f"(batch_size={batch_size})")

        # Check if interrupted during setup
        if _INTERRUPTED:
            print("  [local-judge] Interrupted during setup, exiting.")
            sys.exit(130)

        # Load vLLM model
        print(f"\n[INFER] Loading vLLM model: {judge_model}...")
        from src.llm_judge import load_judge_model, classify_segments_batch
        llm = load_judge_model(
            judge_model,
            tensor_parallel_size=local_cfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=local_cfg.get(
                "gpu_memory_utilization", 0.90),
            max_model_len=local_cfg.get("max_model_len", 4096),
            enable_prefix_caching=local_cfg.get(
                "enable_prefix_caching", True),
            dtype=local_cfg.get("dtype", "auto"),
        )
        print(f"  Model loaded")

        # Check if interrupted during model loading
        if _INTERRUPTED:
            print("  [local-judge] Interrupted during model loading, "
                  "exiting.")
            sys.exit(130)

        # Initialize state
        total_shard_segs = len(shard_segments)
        state = {
            "provider": "local",
            "judge_model": judge_model,
            "model_name": MODEL_NAME,
            "domain": DOMAIN,
            "segmentation": "sentence_window",
            "shard_index": shard_index,
            "num_shards": num_shards,
            "total_segments": total_shard_segs,
            "completed_segments": len(completed),
            "prompt_version": version,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "error": None,
        }
        _local_save_state(state, state_path)

        # Process batches with a single global progress bar
        from tqdm import tqdm
        n_batches = (len(remaining) + batch_size - 1) // batch_size
        pbar = tqdm(total=len(remaining), desc="Classifying segments",
                    unit="seg")
        try:
            for batch_idx in range(n_batches):
                if _INTERRUPTED:
                    print(f"\n  [local-judge] Interrupted after batch "
                          f"{batch_idx}/{n_batches}")
                    state["status"] = "interrupted"
                    state["updated_at"] = datetime.now().isoformat()
                    _local_save_state(state, state_path)
                    sys.exit(130)

                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(remaining))
                batch = remaining[batch_start:batch_end]

                batch_results = classify_segments_batch(
                    llm, batch, system_prompt,
                    temperature=local_cfg["temperature"],
                    max_tokens=local_cfg["max_tokens"],
                    use_tqdm=False,
                )

                _local_append_checkpoint(batch_results, checkpoint_path)
                records.extend(batch_results)

                state["completed_segments"] = len(records)
                state["updated_at"] = datetime.now().isoformat()
                _local_save_state(state, state_path)

                pbar.update(len(batch))
        finally:
            pbar.close()
            # Clean up vLLM engine to avoid shutdown warnings
            del llm
            import gc
            gc.collect()
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass

    # Mark completed
    state = _local_load_state(state_path) or {}
    state["status"] = "completed"
    state["completed_segments"] = len(records)
    state["updated_at"] = datetime.now().isoformat()
    _local_save_state(state, state_path)

    # If unsharded, produce final parquets directly
    if num_shards is None:
        segment_df = pd.DataFrame(records)
        seg_info_path = shard_dir / "segments_info.json"
        segments_info = {}
        if seg_info_path.exists():
            with open(seg_info_path) as f:
                seg_info_list = json.load(f)
            for si in seg_info_list:
                key = (str(si["problem_id"]), int(si["run_idx"]))
                segments_info[key] = si["num_segments"]

        prompt_ver = state.get("prompt_version", "unknown")
        _write_final_parquets(
            segment_df, segments_info, shard_dir, judge_config,
            no_segments, "local", prompt_ver,
        )

    elapsed = time.time() - t0
    shard_label = (f" (shard {shard_index})" if shard_index is not None
                   else "")
    print(f"\n  Inference complete{shard_label} in {_fmt_elapsed(elapsed)}")


def _local_run_merge_shards(judge_config, no_segments=False):
    """Merge shard outputs into final parquets."""
    t0 = time.time()

    shard_dirs = sorted(JUDGE_OUTPUT_DIR.glob("shard_*"))
    if not shard_dirs:
        print("  ERROR: No shard directories found.")
        sys.exit(1)

    print(f"\n[MERGE] Found {len(shard_dirs)} shard(s)")

    incomplete = []
    for sd in shard_dirs:
        state_path = sd / "judge_local_state.json"
        state = _local_load_state(state_path)
        if state is None or state.get("status") != "completed":
            status = state.get("status", "missing") if state else "missing"
            incomplete.append((sd.name, status))

    if incomplete:
        print("  ERROR: Incomplete shards:")
        for name, status in incomplete:
            print(f"    {name}: {status}")
        print("  Complete all shards before merging.")
        sys.exit(1)

    all_records = []
    for sd in shard_dirs:
        checkpoint_path = sd / "checkpoint.jsonl"
        records, _ = _local_load_checkpoint(checkpoint_path)
        all_records.extend(records)
        print(f"  {sd.name}: {len(records)} records")

    print(f"  Total: {len(all_records)} segment records")

    # Merge segments_info from all shards
    merged_seg_info = []
    for sd in shard_dirs:
        seg_info_path = sd / "segments_info.json"
        if seg_info_path.exists():
            with open(seg_info_path) as f:
                merged_seg_info.extend(json.load(f))

    JUDGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_info_path = JUDGE_OUTPUT_DIR / "segments_info.json"
    with open(merged_info_path, "w") as f:
        json.dump(merged_seg_info, f, ensure_ascii=False)

    # Get prompt version from first shard
    first_state = _local_load_state(
        shard_dirs[0] / "judge_local_state.json")
    prompt_ver = (first_state.get("prompt_version", "unknown")
                  if first_state else "unknown")

    # Write merged state
    merged_state = {
        "provider": "local",
        "judge_model": judge_config["judge_local"]["model"],
        "model_name": MODEL_NAME,
        "domain": DOMAIN,
        "segmentation": "sentence_window",
        "total_segments": len(all_records),
        "completed_segments": len(all_records),
        "prompt_version": prompt_ver,
        "status": "completed",
        "num_shards_merged": len(shard_dirs),
        "updated_at": datetime.now().isoformat(),
        "error": None,
    }
    _local_save_state(merged_state,
                      JUDGE_OUTPUT_DIR / "judge_local_state.json")

    # Build segments_info dict for aggregation
    segments_info = {}
    for si in merged_seg_info:
        key = (str(si["problem_id"]), int(si["run_idx"]))
        segments_info[key] = si["num_segments"]

    segment_df = pd.DataFrame(all_records)
    _write_final_parquets(
        segment_df, segments_info, JUDGE_OUTPUT_DIR, judge_config,
        no_segments, "local", prompt_ver,
    )

    elapsed = time.time() - t0
    print(f"\n  Merge complete in {_fmt_elapsed(elapsed)}")


# ---------------------------------------------------------------------------
# Dry run (all providers)
# ---------------------------------------------------------------------------

def _run_dry_run(judge_config, args):
    """Print stats and estimated cost without any API calls."""
    t0 = time.time()

    traces_data, total_segments = prepare_traces(judge_config)
    if total_segments == 0:
        print("  No segments to classify.")
        return

    system_prompt = _build_prompt(judge_config, DOMAIN)
    version = prompt_version_hash(system_prompt)
    prompt_words = len(system_prompt.split())

    n_traces = len(traces_data)
    avg_segs = total_segments / max(n_traces, 1)

    provider = args.provider
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"  DRY RUN SUMMARY ({provider}) -- "
          f"{MODEL_NAME} / {DOMAIN}")
    print(f"{'=' * 60}")
    print(f"  Traces:           {n_traces:>12,}")
    print(f"  Total segments:   {total_segments:>12,}")
    print(f"  Avg segs/trace:   {avg_segs:>12.1f}")
    print(f"  Prompt:           {prompt_words:>12} words (v={version})")

    if provider == "local":
        n_shards = args.num_shards or 1
        est_secs = total_segments / (300 * n_shards)
        if n_shards > 1:
            segs_per_shard = total_segments // n_shards
            remainder = total_segments % n_shards
            print(f"  Shards:           {n_shards:>12}")
            print(f"  Segs/shard:       ~{segs_per_shard:>11,}"
                  f"{f' (+{remainder} in last)' if remainder else ''}")
        print(f"  Est. time:        {_fmt_elapsed(est_secs):>12} "
              f"(~300 seg/s{'*' + str(n_shards) if n_shards > 1 else ''})")
        print(f"  Est. cost:                 $0")
    else:
        est_prompt_tokens = 1100
        est_segment_tokens = 150
        est_output_tokens = 40
        total_input = total_segments * (est_prompt_tokens + est_segment_tokens)
        total_output = total_segments * est_output_tokens
        print(f"  Est. input tokens:  {total_input:>10,}")
        print(f"  Est. output tokens: {total_output:>10,}")

    print(f"\n  Prep time: {_fmt_elapsed(elapsed)}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Multi-config orchestrator (batch API providers)
# ---------------------------------------------------------------------------

def _run_multi(config_names, args, judge_config):
    """Submit all configs, then poll + collect concurrently via threads."""
    t0 = time.time()
    provider = args.provider

    # Build contexts for all configs
    contexts = []
    for cn in config_names:
        try:
            ctx = _make_ctx(cn, judge_config, args)
            contexts.append(ctx)
        except Exception as exc:
            print(f"  WARNING: Could not load config '{cn}': {exc}")

    if not contexts:
        print("  No valid configs to process.")
        return

    print(f"\n[MULTI] Processing {len(contexts)} config(s)")

    # Select provider-specific functions
    if provider == "gemini":
        run_submit_fn = _gemini_run_submit
        poll_with_ctx_fn = _gemini_poll_with_ctx
        collect_with_ctx_fn = _gemini_collect_with_ctx
    else:  # openai
        run_submit_fn = _openai_run_submit
        poll_with_ctx_fn = _openai_poll_with_ctx
        collect_with_ctx_fn = _openai_collect_with_ctx

    # Pass 1: Submit all sequentially
    print(f"\n{'=' * 70}")
    print("  Pass 1: Submitting batches")
    print(f"{'=' * 70}")
    pending = []
    for ctx in contexts:
        label = f"{ctx['model_name']}/{ctx['domain']}"
        _set_globals_from_ctx(ctx)

        print(f"\n--- {label} ---")

        if not ctx["traces_path"].exists():
            print(f"  SKIP: Traces not found: {ctx['traces_path']}")
            continue

        if (not args.force
                and _cache_exists(ctx["judge_parquet_path"])):
            print(f"  SKIP (cached): {ctx['judge_parquet_path'].name}")
            continue

        try:
            run_submit_fn(judge_config,
                          force=args.force, test_n=args.test_n)
        except Exception as exc:
            print(f"  ERROR submitting {label}: {exc}")
            continue

        state = load_judge_state(ctx["judge_output_dir"])
        if state is not None and state["status"] in (
            "submitted", "polling", "collecting"
        ):
            pending.append(ctx)

    if not pending:
        print("\n  No batches to poll or collect.")
        elapsed = time.time() - t0
        print(f"\n{'=' * 70}")
        print(f"  Done in {_fmt_elapsed(elapsed)}")
        print(f"{'=' * 70}")
        return

    # Pass 2: Poll + collect concurrently via threads
    print(f"\n{'=' * 70}")
    print(f"  Pass 2: Polling {len(pending)} batch(es) concurrently")
    print(f"{'=' * 70}")

    stop_event = threading.Event()
    results = {}

    def _worker(ctx):
        label = f"{ctx['model_name']}/{ctx['domain']}"
        try:
            state = poll_with_ctx_fn(ctx, judge_config,
                                     stop_event=stop_event)
            if state is None:
                results[ctx["config_name"]] = (None, None)
                return

            if state["status"] == "failed":
                print(f"  [{label}] BATCH FAILED: "
                      f"{state.get('error', 'unknown')}")
                results[ctx["config_name"]] = (state, None)
                return

            if state["status"] == "collecting":
                collect_with_ctx_fn(
                    ctx, judge_config,
                    no_segments=args.no_segments,
                    force=args.force,
                )

            results[ctx["config_name"]] = (state, None)
        except Exception as exc:
            print(f"  [{label}] ERROR: {exc}")
            results[ctx["config_name"]] = (None, str(exc))

    threads = []
    for ctx in pending:
        t = threading.Thread(target=_worker, args=(ctx,),
                             name=ctx["config_name"])
        t.daemon = True
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\n  Ctrl+C received, signaling threads to stop...")
        stop_event.set()
        for t in threads:
            t.join(timeout=30)
        print("  All threads stopped. State saved -- re-run to resume.")
        return

    # Summary
    elapsed = time.time() - t0
    n_ok = sum(1 for s, e in results.values()
               if s is not None and s.get("status") == "completed"
               and e is None)
    n_fail = sum(1 for s, e in results.values()
                 if e is not None
                 or (s is not None and s.get("status") == "failed"))
    n_skip = len(contexts) - len(pending)

    print(f"\n{'=' * 70}")
    print(f"  Multi-config complete in {_fmt_elapsed(elapsed)}")
    print(f"  Completed: {n_ok}  Failed: {n_fail}  Skipped: {n_skip}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

def run_gemini(args, judge_config):
    """Run the Gemini provider for the current config."""
    if args.dry_run:
        _run_dry_run(judge_config, args)
        return

    if args.submit:
        _gemini_run_submit(judge_config,
                           force=args.force, test_n=args.test_n)
    elif args.collect:
        _gemini_run_collect(judge_config,
                            no_segments=args.no_segments,
                            force=args.force)
    else:
        _gemini_run_submit(judge_config,
                           force=args.force, test_n=args.test_n)

        state = _gemini_run_poll(judge_config)
        if state is None:
            return

        if state["status"] == "failed":
            print(f"\n  BATCH FAILED: {state.get('error', 'unknown')}")
            sys.exit(1)

        if state["status"] == "collecting":
            _gemini_run_collect(judge_config,
                                no_segments=args.no_segments,
                                force=args.force)


def run_openai(args, judge_config):
    """Run the OpenAI provider for the current config."""
    if args.dry_run:
        _run_dry_run(judge_config, args)
        return

    if args.submit:
        _openai_run_submit(judge_config,
                           force=args.force, test_n=args.test_n)
    elif args.collect:
        _openai_run_collect(judge_config,
                            no_segments=args.no_segments,
                            force=args.force)
    else:
        _openai_run_submit(judge_config,
                           force=args.force, test_n=args.test_n)

        state = _openai_run_poll(judge_config)
        if state is None:
            return

        if state["status"] == "failed":
            print(f"\n  BATCH FAILED: {state.get('error', 'unknown')}")
            sys.exit(1)

        if state["status"] == "collecting":
            _openai_run_collect(judge_config,
                                no_segments=args.no_segments,
                                force=args.force)


def run_local(args, judge_config):
    """Run the local vLLM provider for the current config."""
    if args.dry_run:
        _run_dry_run(judge_config, args)
        return

    if args.merge_shards:
        _local_run_merge_shards(judge_config,
                                no_segments=args.no_segments)
        return

    # Cache check for unsharded runs
    if (args.shard_index is None and not args.force
            and _cache_exists(JUDGE_PARQUET_PATH)):
        print(f"  SKIP (cached): {JUDGE_PARQUET_PATH.name}")
        print("  Use --force to recompute.")
        return

    _local_run_inference(
        judge_config,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        no_segments=args.no_segments,
        force=args.force,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified LLM-as-a-judge per-segment classification "
                    "of CoT reasoning patterns",
    )

    # Required
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["gemini", "openai", "local"],
        help="Inference provider.",
    )
    # Config selection
    config_group = parser.add_argument_group("config selection")
    config_group.add_argument(
        "--config",
        type=str,
        nargs="*",
        default=None,
        help="One or more model config names "
             "(e.g. pipeline/code/deepseek-r1-7b).",
    )
    config_group.add_argument(
        "--all",
        action="store_true",
        help="Process all configs (pipeline + eval-only).",
    )

    # Batch API ops (gemini/openai)
    batch_group = parser.add_argument_group("batch API (gemini/openai)")
    batch_group.add_argument(
        "--submit",
        action="store_true",
        help="Submit batch only (returns immediately).",
    )
    batch_group.add_argument(
        "--status",
        action="store_true",
        help="Show status of all active/completed batches.",
    )
    batch_group.add_argument(
        "--collect",
        action="store_true",
        help="Collect results from a completed batch.",
    )
    batch_group.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        help="Override poll interval in seconds.",
    )

    # Behavior
    behavior_group = parser.add_argument_group("behavior")
    behavior_group.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if cached output exists.",
    )
    behavior_group.add_argument(
        "--no-segments",
        action="store_true",
        help="Skip saving segment-level parquet.",
    )
    behavior_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without running inference.",
    )
    behavior_group.add_argument(
        "--test-n",
        type=int,
        default=None,
        metavar="N",
        help="Process only N segments (for testing).",
    )

    # Local provider
    local_group = parser.add_argument_group("local provider")
    local_group.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="HuggingFace model ID for local judge.",
    )
    local_group.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Shard index (0-based).",
    )
    local_group.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards.",
    )
    local_group.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge shard outputs into final parquets.",
    )
    local_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Segments per checkpoint batch.",
    )
    local_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of GPU memory to use.",
    )
    local_group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism.",
    )
    local_group.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length.",
    )

    # Path overrides
    override_group = parser.add_argument_group("path overrides")
    override_group.add_argument(
        "--traces-dir",
        type=str,
        default=None,
        help="Override cot_traces base directory. Looks for "
             "{traces-dir}/{domain}/{model}/cot_traces.jsonl.",
    )
    override_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override judge output base directory. Outputs go to "
             "{output-dir}/{domain}/{model}/llm_judge/{slug}/.",
    )

    return parser.parse_args()


def _validate_provider_args(args):
    """Validate that provider-specific args are not mixed."""
    if args.provider != "local":
        local_only = ["judge_model", "shard_index", "num_shards",
                      "merge_shards", "batch_size",
                      "gpu_memory_utilization", "tensor_parallel_size",
                      "max_model_len"]
        for attr in local_only:
            val = getattr(args, attr, None)
            if val is not None and val is not False:
                print(f"WARNING: --{attr.replace('_', '-')} is only used "
                      f"with --provider local (ignored).")

    if args.provider == "local":
        if args.submit or args.collect or args.status:
            print("ERROR: --submit/--collect/--status are only for "
                  "batch API providers (gemini/openai).")
            sys.exit(1)
        if args.shard_index is not None and args.num_shards is None:
            print("ERROR: --shard-index requires --num-shards.")
            sys.exit(1)
        if args.num_shards is not None and args.shard_index is None:
            if not args.merge_shards and not args.dry_run:
                print("ERROR: --num-shards requires --shard-index "
                      "(or --merge-shards / --dry-run).")
                sys.exit(1)


def main():
    args = parse_args()
    _validate_provider_args(args)

    # --status: no config required
    if args.status:
        if args.provider == "gemini":
            _gemini_run_status()
        elif args.provider == "openai":
            _openai_run_status()
        return

    # Resolve config list
    if args.all:
        config_names = list_configs(include_eval_only=True)
    elif args.config is not None:
        config_names = args.config
    else:
        print("ERROR: --config or --all is required (except for --status).")
        sys.exit(1)

    if not config_names:
        print("ERROR: No configs found.")
        sys.exit(1)

    judge_config = load_judge_config()

    # Apply CLI overrides
    if args.poll_interval is not None:
        if args.provider == "gemini":
            judge_config["batch"]["poll_interval"] = args.poll_interval
        elif args.provider == "openai":
            judge_config["batch_openai"]["poll_interval"] = args.poll_interval

    if args.provider == "local":
        if args.judge_model is not None:
            judge_config["judge_local"]["model"] = args.judge_model
        if args.batch_size is not None:
            judge_config["judge_local"]["batch_size"] = args.batch_size
        if args.gpu_memory_utilization is not None:
            judge_config["judge_local"]["gpu_memory_utilization"] = (
                args.gpu_memory_utilization)
        if args.tensor_parallel_size is not None:
            judge_config["judge_local"]["tensor_parallel_size"] = (
                args.tensor_parallel_size)
        if args.max_model_len is not None:
            judge_config["judge_local"]["max_model_len"] = args.max_model_len

    # Multi-config path (batch API providers)
    if len(config_names) > 1 and args.provider in ("gemini", "openai"):
        if args.dry_run:
            for cn in config_names:
                try:
                    ctx = _make_ctx(cn, judge_config, args)
                except Exception as exc:
                    print(f"  WARNING: Could not load '{cn}': {exc}")
                    continue
                _set_globals_from_ctx(ctx)
                print(f"\n--- {ctx['model_name']}/{ctx['domain']} ---")
                if not ctx["traces_path"].exists():
                    print(f"  SKIP: Traces not found: {ctx['traces_path']}")
                    continue
                _run_dry_run(judge_config, args)
            return

        if args.submit:
            submit_fn = (_gemini_run_submit if args.provider == "gemini"
                         else _openai_run_submit)
            for cn in config_names:
                try:
                    ctx = _make_ctx(cn, judge_config, args)
                except Exception as exc:
                    print(f"  WARNING: Could not load '{cn}': {exc}")
                    continue
                _set_globals_from_ctx(ctx)
                label = f"{ctx['model_name']}/{ctx['domain']}"
                print(f"\n--- {label} ---")
                if not ctx["traces_path"].exists():
                    print(f"  SKIP: Traces not found: {ctx['traces_path']}")
                    continue
                if (not args.force
                        and _cache_exists(ctx["judge_parquet_path"])):
                    print(f"  SKIP (cached): "
                          f"{ctx['judge_parquet_path'].name}")
                    continue
                try:
                    submit_fn(judge_config,
                              force=args.force, test_n=args.test_n)
                except Exception as exc:
                    print(f"  ERROR: {exc}")
            return

        if args.collect:
            collect_fn = (_gemini_collect_with_ctx
                          if args.provider == "gemini"
                          else _openai_collect_with_ctx)
            for cn in config_names:
                try:
                    ctx = _make_ctx(cn, judge_config, args)
                except Exception as exc:
                    print(f"  WARNING: Could not load '{cn}': {exc}")
                    continue
                label = f"{ctx['model_name']}/{ctx['domain']}"
                print(f"\n--- {label} ---")
                try:
                    collect_fn(
                        ctx, judge_config,
                        no_segments=args.no_segments,
                        force=args.force,
                    )
                except Exception as exc:
                    print(f"  ERROR: {exc}")
            return

        # Default: full pipeline with concurrent polling
        _run_multi(config_names, args, judge_config)
        return

    # Single-config path (or multi-config for local provider)
    for config_name in config_names:
        try:
            ctx = _make_ctx(config_name, judge_config, args)
        except Exception as exc:
            print(f"  WARNING: Could not load '{config_name}': {exc}")
            continue

        _set_globals_from_ctx(ctx)

        # Determine provider label
        provider_labels = {
            "gemini": "Gemini",
            "openai": "OpenAI",
            "local": "Local vLLM",
        }

        print("=" * 70)
        print(f"  IRT Latent Difficulty -- LLM Judge "
              f"({provider_labels[args.provider]})")
        print("=" * 70)
        print(f"  Model:    {MODEL_NAME}")
        print(f"  Domain:   {DOMAIN}")
        print(f"  Config:   {config_name}")
        print(f"  Output:   {JUDGE_OUTPUT_DIR}")
        print(f"  Force:    {args.force}")
        if args.provider == "local":
            judge_model = judge_config["judge_local"]["model"]
            print(f"  Judge:    {judge_model}")
            if args.shard_index is not None:
                print(f"  Shard:    {args.shard_index}/{args.num_shards}")
        print()

        if not TRACES_PATH.exists():
            print(f"  SKIP: Traces not found: {TRACES_PATH}")
            print("  Generate traces first, then run the judge.")
            continue

        if (not args.force and not args.collect
                and not args.merge_shards
                and _cache_exists(JUDGE_PARQUET_PATH)):
            print(f"  SKIP (cached): {JUDGE_PARQUET_PATH.name}")
            print("  Use --force to recompute.")
            continue

        t_total = time.time()

        if args.provider == "gemini":
            run_gemini(args, judge_config)
        elif args.provider == "openai":
            run_openai(args, judge_config)
        elif args.provider == "local":
            run_local(args, judge_config)

        elapsed = time.time() - t_total
        print(f"\n{'=' * 70}")
        print(f"  Done in {_fmt_elapsed(elapsed)}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
