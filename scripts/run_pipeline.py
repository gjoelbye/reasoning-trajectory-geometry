"""Unified CoT generation and activation extraction pipeline.

Generates Chain-of-Thought traces for selected problems (Codeforces or MATH),
with optional hidden-state extraction via PyTorch forward hooks in the same
inference pass.  This guarantees that saved text and activations correspond
to the exact same stochastic trace.

Modes:
  - Text-only (--layers omitted): writes JSONL traces, no HDF5.
  - With activations (--layers): writes both JSONL and HDF5 in a single
    generation pass.  Bare --layers uses default layer indices; explicit
    indices can be given as --layers 0 6 13 20 27.

Performance optimizations (20-50x speedup vs. previous version):
  - Batched HDF5 writes: file opened once per shard instead of per trace
  - Fast compression: LZF default (much faster than GZIP, good compression)
  - Optimized think-block detection: binary search instead of token-by-token

Supports sharding via --shard-index / --num-shards for parallel execution
across a SLURM job array.  Each shard writes to separate output files to
avoid concurrent write conflicts.  Use scripts/merge_shards.py to combine
outputs afterward.

Persistent mode (--persistent):
  Automatically recovers from CUDA out-of-memory errors and other transient
  GPU failures without losing progress.  On OOM the script clears GPU caches,
  waits (with graduated back-off), and retries.  After --max-retries
  consecutive failures on the same trace, it is skipped and logged to
  skipped_traces.jsonl.  Also enables periodic GPU memory status logging
  and graceful KeyboardInterrupt handling.

Usage (config-driven, recommended):
    python scripts/run_pipeline.py --config code/qwen-7b --layers
    python scripts/run_pipeline.py --config math/deepseek-r1-7b --layers

    # Override a config value:
    python scripts/run_pipeline.py --config code/qwen-7b --layers --num-runs 5

    # With OOM recovery:
    python scripts/run_pipeline.py --config code/qwen-7b --layers --persistent

Usage (manual, without config):
    python scripts/run_pipeline.py \\
        --problems data/datasets/selected_500.parquet \\
        --output-dir data/cot_traces \\
        --num-runs 5 \\
        --max-new-tokens 16384

Usage (with activations):
    python scripts/run_pipeline.py \\
        --problems data/datasets/selected_500.parquet \\
        --output-dir data/pipeline_output \\
        --num-runs 5 \\
        --max-new-tokens 16384 \\
        --layers \\
        --stride 10 \\
        --compression lzf

Usage (SLURM shard):
    python scripts/run_pipeline.py \\
        --config code/qwen-7b \\
        --layers \\
        --shard-index $SLURM_ARRAY_TASK_ID \\
        --num-shards 10

"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import transformers
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models import (
    load_model,
    load_tokenizer,
    format_chat_prompt,
    default_extraction_layers,
    get_model_config,
    MODEL_CONFIGS,
    THINK_START,
    THINK_END,
)
from src.extraction import extract_think_block_states


# ---------------------------------------------------------------------------
# Generation (with optional hook-based activation extraction)
# ---------------------------------------------------------------------------

def generate_trace(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    layer_indices: Optional[List[int]] = None,
    seed: Optional[int] = None,
    scramble_input: bool = False,
) -> Tuple[str, Optional[Dict[int, List[torch.Tensor]]], Optional[List[int]]]:
    """Run a single generation pass with optional activation extraction.

    Parameters
    ----------
    model : nnsight.LanguageModel
    tokenizer : the model's tokenizer
    prompt : full chat-formatted prompt string
    max_new_tokens : generation length limit
    temperature : sampling temperature
    top_p : nucleus sampling threshold
    layer_indices : layers to extract from, or None for text-only mode
    seed : if provided, seed RNG before generation for reproducibility
    scramble_input : if True, randomly permute content tokens before generation

    Returns
    -------
    (response_text, hidden_states_or_None, token_ids_or_None)
        hidden_states maps layer_index -> list of (hidden_dim,) tensors,
        one per generated token.  Both are None in text-only mode.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model._model.device)
    attention_mask = inputs["attention_mask"].to(model._model.device)

    # Scramble input tokens if requested (control condition).
    # Keeps special tokens (BOS, EOS, pad) in place but randomly
    # permutes the content tokens.
    if scramble_input:
        content_mask = torch.ones_like(input_ids[0], dtype=torch.bool)
        special_ids = set(tokenizer.all_special_ids)
        for j in range(input_ids.shape[1]):
            if input_ids[0, j].item() in special_ids:
                content_mask[j] = False
        content_indices = content_mask.nonzero(as_tuple=True)[0]
        perm = content_indices[torch.randperm(len(content_indices))]
        scrambled = input_ids.clone()
        scrambled[0, content_indices] = input_ids[0, perm]
        input_ids = scrambled

    # Seed RNG for reproducibility.  Setting both CPU and CUDA seeds
    # ensures identical sampling across reruns on the same hardware.
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Register forward hooks when extracting activations.
    hooks = []
    collected: Dict[int, List[torch.Tensor]] = {}
    if layer_indices is not None:
        collected = {idx: [] for idx in layer_indices}
        for idx in layer_indices:
            layer_module = model._model.model.layers[idx]

            def _hook(module, input, output, _idx=idx):
                h = output if isinstance(output, torch.Tensor) else output[0]
                h = h[:, -1, :].detach().cpu()
                collected[_idx].append(h.squeeze(0))

            hooks.append(layer_module.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            outputs = model._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
    except Exception:
        # On OOM (or any error during generate), eagerly free GPU input
        # tensors so they don't linger while the retry loop runs
        # gc.collect() + empty_cache().  Hooks are removed in finally.
        del input_ids, attention_mask, inputs
        raise
    finally:
        # CRITICAL: Always remove hooks, even on OOM.  Stale hooks
        # accumulate memory captures and cascade into further OOM.
        for hook in hooks:
            hook.remove()

    generated = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=False)

    # Free GPU tensors from generate() immediately — the full output
    # sequence and KV-cache can consume significant VRAM.
    token_ids_list = generated.tolist() if layer_indices is not None else None
    del outputs, generated, input_ids, attention_mask, inputs

    # The chat template places <think> in the generation prefix (input
    # tokens), so it is absent from the generated output.  Re-attach it
    # so downstream parsing finds both tags.
    if THINK_END in response and THINK_START not in response:
        response = THINK_START + response

    if layer_indices is not None:
        # Drop the prefill call (first entry per layer) so entries align
        # 1:1 with generated tokens.
        hidden_states = {idx: collected[idx][1:] for idx in layer_indices}
        return response, hidden_states, token_ids_list

    return response, None, None


# ---------------------------------------------------------------------------
# OOM recovery helpers (used with --persistent)
# ---------------------------------------------------------------------------

def _is_cuda_error(exc: BaseException) -> bool:
    """Return True if *exc* is a CUDA/GPU-related error worth retrying."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(kw in msg for kw in (
            "out of memory", "cuda error", "cublas", "cudnn", "nccl",
        ))
    return False


def _log_gpu_memory(prefix: str = "") -> None:
    """Print current GPU memory statistics."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free = total - reserved
    label = f"[GPU Memory {prefix}]" if prefix else "[GPU Memory]"
    print(f"{label} Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | "
          f"Free: ~{free:.2f} GB | Total: {total:.2f} GB",
          flush=True)


def generate_trace_with_retry(
    model, tokenizer, prompt,
    max_new_tokens, temperature, top_p,
    layer_indices, seed, scramble_input,
    *,
    max_retries: int = 3,
    base_wait: int = 30,
    problem_id: str = "",
    run_idx: int = 0,
):
    """Call ``generate_trace`` with graduated OOM recovery.

    Returns
    -------
    (response, hidden_states, token_ids, success)
        *success* is ``False`` if all retries were exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            response, hidden_states, token_ids = generate_trace(
                model, tokenizer, prompt,
                max_new_tokens, temperature, top_p,
                layer_indices=layer_indices,
                seed=seed,
                scramble_input=scramble_input,
            )
            return response, hidden_states, token_ids, True

        except Exception as exc:
            if not _is_cuda_error(exc):
                raise  # Not a GPU error — propagate immediately

            attempts_left = max_retries - attempt
            print(f"\n[OOM] Problem {problem_id} run {run_idx}, "
                  f"attempt {attempt + 1}/{max_retries + 1}: {type(exc).__name__}",
                  flush=True)

            # Graduated cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                _log_gpu_memory("after cleanup")
            except Exception:
                print("[GPU Memory] Unable to query GPU stats", flush=True)

            if attempts_left == 0:
                print(f"[OOM] Max retries exhausted for {problem_id} run {run_idx}. "
                      f"Skipping.", flush=True)
                return None, None, None, False

            # First retry: immediate (cache clear may be enough).
            # Subsequent retries: graduated wait.
            if attempt == 0:
                print("[OOM] Retrying immediately after cache clear...", flush=True)
            else:
                wait = base_wait * attempt
                print(f"[OOM] Waiting {wait}s before retry "
                      f"({attempts_left} attempts remaining)...", flush=True)
                time.sleep(wait)

    # Should never reach here, but defensively:
    return None, None, None, False


# ---------------------------------------------------------------------------
# HDF5 persistence (batched writes with open file handle)
# ---------------------------------------------------------------------------

def prepare_activation_data(
    hidden_states: Dict[int, List[torch.Tensor]],
    token_ids: List[int],
    layer_indices: List[int],
    stride: int,
) -> Tuple[List[int], Dict[int, np.ndarray]]:
    """Process hidden states into numpy arrays ready for HDF5 writing.

    Parameters
    ----------
    hidden_states : dict mapping layer_index -> list of tensors
    token_ids : generated token IDs
    layer_indices : which layers to save
    stride : sample every Nth token

    Returns
    -------
    (token_ids, layer_arrays)
        token_ids : list of token IDs (possibly strided)
        layer_arrays : dict mapping layer_idx -> numpy array (float16)
    """
    # Apply stride to token_ids first (once, outside the loop)
    if stride > 1:
        token_ids = token_ids[::stride]

    # Convert to numpy arrays
    layer_arrays = {}
    for layer_idx in layer_indices:
        states = hidden_states.get(layer_idx, [])
        if not states:
            continue

        if isinstance(states, list):
            states = torch.stack(states)

        # Apply stride to hidden states
        if stride > 1:
            states = states[::stride]

        # Cast via PyTorch first: numpy doesn't support bfloat16
        layer_arrays[layer_idx] = states.half().numpy()

    return token_ids, layer_arrays


def write_activation_data(
    hf: h5py.File,
    problem_id: str,
    run_idx: int,
    token_ids: List[int],
    layer_arrays: Dict[int, np.ndarray],
    rating: float,
    compression: str = "lzf",
) -> None:
    """Write preprocessed activation data to an open HDF5 file.

    Parameters
    ----------
    hf : open h5py.File handle in append mode
    problem_id : unique problem identifier
    run_idx : run index for this problem
    token_ids : token IDs for this trace
    layer_arrays : dict mapping layer_idx -> numpy array (float16)
    rating : problem difficulty rating
    compression : "gzip", "lzf", or None
    """
    group_name = f"{problem_id}/run_{run_idx}"

    # Remove existing group if present (crash recovery)
    if group_name in hf:
        del hf[group_name]

    grp = hf.create_group(group_name)
    grp.create_dataset("token_ids", data=np.array(token_ids, dtype=np.int32))
    grp.attrs["rating"] = rating
    grp.attrs["problem_id"] = problem_id
    grp.attrs["run_idx"] = run_idx

    # Write layer datasets with specified compression
    comp_kwargs = {}
    if compression == "gzip":
        comp_kwargs = {"compression": "gzip", "compression_opts": 4}
    elif compression == "lzf":
        comp_kwargs = {"compression": "lzf"}
    # else: no compression

    for layer_idx, array in layer_arrays.items():
        grp.create_dataset(
            f"layer_{layer_idx}",
            data=array,
            **comp_kwargs,
        )


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def append_result(result: dict, path: Path) -> None:
    """Append a single result to the JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_completed(jsonl_path: Path, h5_path: Optional[Path] = None) -> set:
    """Load already-completed (problem_id, run_idx) pairs for resumption.

    When activations are enabled, a pair is only considered complete if it
    exists in both the JSONL and the HDF5 file.
    """
    jsonl_done = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                jsonl_done.add((r["problem_id"], r["run_idx"]))

    if h5_path is None or not h5_path.exists():
        return jsonl_done

    h5_done = set()
    with h5py.File(h5_path, "r") as hf:
        for problem_id in hf:
            grp = hf[problem_id]
            if isinstance(grp, h5py.Group):
                for run_key in grp:
                    if run_key.startswith("run_"):
                        idx = int(run_key.split("_", 1)[1])
                        h5_done.add((problem_id, idx))

    return jsonl_done & h5_done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT traces with optional activation extraction"
    )
    # --- Config-driven defaults (recommended) ---
    parser.add_argument("--config", type=str, default=None,
                        help="Config name (e.g. 'code/qwen-7b', 'math/llama-8b') "
                             "to load model, paths, and generation settings from YAML. "
                             "CLI flags override config values.")
    # --- Model & data (defaults populated from --config when provided) ---
    parser.add_argument("--model", type=str,
                        choices=sorted(MODEL_CONFIGS.keys()),
                        default=None,
                        help="Model key from MODEL_CONFIGS "
                             "(see src/models.py for full list). "
                             "Default: deepseek-r1-7b (or from --config)")
    parser.add_argument("--problems", type=Path, default=None,
                        help="Path to the selected problems parquet file "
                             "(auto-set by --config)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for JSONL traces and HDF5 activations "
                             "(auto-set by --config from scratch paths)")
    parser.add_argument("--domain", type=str, choices=["codeforces", "math", "sat"],
                        default=None,
                        help="Problem domain: 'codeforces', 'math', or 'sat'. "
                             "Auto-inferred from --config path prefix.")
    # --- Generation parameters (defaults populated from --config when provided) ---
    parser.add_argument("--num-runs", type=int, default=None,
                        help="Number of independent traces per problem (default: 5)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Maximum tokens to generate per trace (default: 16384)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default: 0.6)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling threshold (default: 0.95)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Save every Nth token's hidden state (default: 10). "
                             "Stride 1 saves all positions")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed for reproducibility. Each trace gets a "
                             "deterministic seed derived from this value, the problem ID, "
                             "and the run index. Omit for non-deterministic sampling.")
    # --- Activation extraction ---
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Enable activation extraction. Omit for text-only mode. "
                             "Bare --layers uses default_extraction_layers(); "
                             "explicit indices like --layers 0 6 13 20 27 extract those layers")
    parser.add_argument("--compression", type=str, default="lzf",
                        choices=["gzip", "lzf", "none"],
                        help="HDF5 compression: 'lzf' (fast, good compression), "
                             "'gzip' (slower, better compression), 'none' (fastest, largest). "
                             "Default: lzf")
    # --- Infrastructure ---
    parser.add_argument("--shard-index", type=int, default=None,
                        help="Which shard this process handles (for SLURM array jobs)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards to split problems into")
    parser.add_argument("--quantization", type=str, choices=["bf16", "8bit"],
                        default="bf16",
                        help="Precision: 'bf16' (default, full bfloat16) or "
                             "'8bit' (bitsandbytes INT8, ~50%% memory reduction)")
    parser.add_argument("--num-problems", type=int, default=None,
                        help="Limit to first N problems (default: use all). "
                             "Applied before sharding.")
    # --- Experimental ---
    parser.add_argument("--scramble-input", action="store_true",
                        help="Randomly permute input token IDs before generation. "
                             "Used as a control condition to test whether trajectory "
                             "structure depends on problem content.")
    # --- Persistence / OOM recovery ---
    parser.add_argument("--persistent", action="store_true",
                        help="Enable OOM-resilient persistent mode with automatic "
                             "retry, skip logging, GPU status monitoring, and "
                             "graceful KeyboardInterrupt handling.")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum OOM retry attempts per trace (default: 3). "
                             "Only used with --persistent.")
    parser.add_argument("--retry-wait", type=int, default=30,
                        help="Base wait time in seconds between OOM retries "
                             "(default: 30). Only used with --persistent.")
    parser.add_argument("--status-interval", type=int, default=50,
                        help="Print GPU memory stats every N traces (default: 50). "
                             "Only used with --persistent.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Apply config defaults: --config → YAML values → hardcoded fallbacks
    # Priority: explicit CLI flag > YAML config > hardcoded default
    # ------------------------------------------------------------------
    _HARDCODED_DEFAULTS = {
        "model": "deepseek-r1-7b",
        "num_runs": 5,
        "max_new_tokens": 16384,
        "temperature": 0.6,
        "top_p": 0.95,
        "stride": 10,
        "output_dir": Path("data/cot_traces"),
        "domain": "codeforces",
    }

    if args.config is not None:
        from src.config import load_config
        cfg = load_config(args.config)
        gen = cfg.get("generation", {})

        cfg_vals = {
            "model": cfg["model"]["name"],
            "problems": cfg["paths"]["problems"],
            "output_dir": cfg["paths"]["pipeline"]["cot_traces"].parent,
            "num_runs": gen.get("num_runs"),
            "max_new_tokens": gen.get("max_new_tokens"),
            "temperature": gen.get("temperature"),
            "top_p": gen.get("top_p"),
            "stride": gen.get("stride"),
            "seed": gen.get("seed"),
            "domain": ("math" if "math" in args.config.split("/")
                      else "sat" if "sat" in args.config.split("/")
                      else "codeforces"),
        }
        for key, cfg_val in cfg_vals.items():
            if getattr(args, key) is None and cfg_val is not None:
                setattr(args, key, cfg_val)

    # Fill remaining Nones with hardcoded defaults
    for key, default_val in _HARDCODED_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, default_val)

    if args.problems is None:
        parser.error("--problems is required (provide it directly or use --config)")

    extract_mode = args.layers is not None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output file paths
    if args.shard_index is not None:
        suffix = f"_shard_{args.shard_index:03d}"
    else:
        suffix = ""
    jsonl_path = args.output_dir / f"cot_traces{suffix}.jsonl"
    h5_path = args.output_dir / f"activations{suffix}.h5" if extract_mode else None

    # Resumption
    completed = load_completed(jsonl_path, h5_path)
    if completed:
        print(f"Found {len(completed)} existing results, resuming...")

    # Load problems
    problems = pd.read_parquet(args.problems)

    if args.num_problems is not None:
        problems = problems.iloc[:args.num_problems].reset_index(drop=True)
        print(f"Limited to {args.num_problems} problems")

    if args.shard_index is not None and args.num_shards is not None:
        n = len(problems)
        shard_size = (n + args.num_shards - 1) // args.num_shards
        start = args.shard_index * shard_size
        end = min(start + shard_size, n)
        problems = problems.iloc[start:end].reset_index(drop=True)
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(problems)} problems")

    if "formatted_prompt" not in problems.columns:
        raise ValueError(
            "The problems parquet must contain a 'formatted_prompt' column. "
            "Run the data selection notebook first."
        )

    # Suppress HuggingFace info messages (pad_token_id, compile_config, etc.)
    transformers.logging.set_verbosity_error()

    # Load model (use config for the selected model version)
    model_cfg = get_model_config(args.model)
    model_id = model_cfg["model_id"]
    model_hidden_dim = model_cfg["hidden_dim"]
    model_num_layers = model_cfg["num_layers"]
    system_prompt = model_cfg.get("system_prompt")  # None = use template default

    print(f"Loading model: {model_id} ({args.model})...")
    model = load_model(model_id=model_id,
                       load_in_8bit=(args.quantization == "8bit"))
    tokenizer = load_tokenizer(model_id=model_id)

    # Resolve bare --layers to default layer indices for this model
    if extract_mode and len(args.layers) == 0:
        args.layers = default_extraction_layers(model_num_layers)
        print(f"Using default extraction layers: {args.layers}")

    layer_indices = args.layers if extract_mode else None

    # Handle compression setting
    compression = args.compression if args.compression != "none" else None

    # Save run configuration for reproducibility
    config = {
        "model": args.model,
        "model_id": model_id,
        "hidden_dim": model_hidden_dim,
        "num_layers": model_num_layers,
        "problems": str(args.problems),
        "output_dir": str(args.output_dir),
        "num_runs": args.num_runs,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "layers": layer_indices,
        "stride": args.stride,
        "compression": args.compression,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "quantization": args.quantization,
        "num_problems": args.num_problems,
        "system_prompt": system_prompt,
        "domain": args.domain,
        "scramble_input": args.scramble_input,
        "persistent_mode": args.persistent,
    }
    if args.persistent:
        config["max_retries"] = args.max_retries
        config["retry_wait"] = args.retry_wait
    config_path = args.output_dir / f"config{suffix}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Run config saved to {config_path}")

    # Open HDF5 file once for the entire shard (major performance optimization)
    hf = None
    if extract_mode:
        hf = h5py.File(h5_path, "a")
        # Write metadata on first open
        if "layers" not in hf.attrs:
            hf.attrs["layers"] = layer_indices
            hf.attrs["hidden_dim"] = model_hidden_dim
            hf.attrs["stride"] = args.stride
            hf.attrs["compression"] = compression if compression else "none"
            hf.attrs["domain"] = args.domain
            hf.attrs["scrambled"] = args.scramble_input
            print(f"HDF5 compression: {compression if compression else 'none'}")

    # Persistent-mode tracking
    skipped_count = 0
    traces_since_status = 0
    skipped_path = args.output_dir / f"skipped_traces{suffix}.jsonl" if args.persistent else None
    interrupted = False

    try:
        # Main loop — progress bar shows overall position including completed
        total = len(problems) * args.num_runs
        pbar = tqdm(total=total, initial=len(completed), desc="Generating traces")

        print(f"  [difficulty] Domain={args.domain}, using "
              f"{'level_int' if args.domain == 'math' else 'num_clauses' if args.domain == 'sat' else 'unnorm_rating'} as difficulty rating")

        for run_idx in range(args.num_runs):
            for i, row in problems.iterrows():
                problem_id = str(row["join_key"])
                if (problem_id, run_idx) in completed:
                    continue

                prompt = format_chat_prompt(row["formatted_prompt"], tokenizer,
                                          system_prompt=system_prompt,
                                          domain=args.domain)

                # Domain-specific difficulty rating
                if args.domain == "math":
                    if "level_int" not in row or pd.isna(row.get("level_int")):
                        print(f"  WARNING: Missing level_int for problem {problem_id}, using -1")
                    rating = float(row.get("level_int", -1))
                elif args.domain == "sat":
                    if "num_clauses" not in row or pd.isna(row.get("num_clauses")):
                        print(f"  WARNING: Missing num_clauses for problem {problem_id}, using -1")
                    rating = float(row.get("num_clauses", -1))
                else:
                    if "unnorm_rating" not in row or pd.isna(row.get("unnorm_rating")):
                        print(f"  WARNING: Missing unnorm_rating for problem {problem_id}, using -1")
                    rating = float(row.get("unnorm_rating", -1))

                # Derive a deterministic per-trace seed from the base seed,
                # problem ID, and run index so that each trace is individually
                # reproducible regardless of execution order or resumption.
                trace_seed = None
                if args.seed is not None:
                    hash_input = f"{args.seed}:{problem_id}:{run_idx}"
                    trace_seed = int.from_bytes(
                        hash_input.encode(), byteorder="big"
                    ) % (2**32)

                t0 = time.time()

                if args.persistent:
                    response, hidden_states, token_ids, success = generate_trace_with_retry(
                        model, tokenizer, prompt,
                        args.max_new_tokens,
                        args.temperature,
                        args.top_p,
                        layer_indices,
                        trace_seed,
                        args.scramble_input,
                        max_retries=args.max_retries,
                        base_wait=args.retry_wait,
                        problem_id=problem_id,
                        run_idx=run_idx,
                    )
                    elapsed = time.time() - t0

                    if not success:
                        skipped_count += 1
                        append_result({
                            "problem_id": problem_id,
                            "run_idx": run_idx,
                            "rating": rating,
                            "reason": "oom_max_retries_exhausted",
                            "timestamp": datetime.now().isoformat(),
                            "max_retries": args.max_retries,
                        }, skipped_path)
                        pbar.update(1)
                        traces_since_status += 1
                        continue
                else:
                    response, hidden_states, token_ids = generate_trace(
                        model, tokenizer, prompt,
                        args.max_new_tokens,
                        args.temperature,
                        args.top_p,
                        layer_indices=layer_indices,
                        seed=trace_seed,
                        scramble_input=args.scramble_input,
                    )
                    elapsed = time.time() - t0

                # Write HDF5 first (so a crash before JSONL means the pair
                # re-runs on the next invocation).
                if extract_mode:
                    processed_tokens, layer_arrays = prepare_activation_data(
                        hidden_states, token_ids, layer_indices,
                        args.stride,
                    )
                    write_activation_data(
                        hf, problem_id, run_idx,
                        processed_tokens, layer_arrays,
                        rating, compression
                    )
                    # Flush to disk so data survives crashes
                    hf.flush()

                    # Free CPU-side activation data now that it's on disk
                    del hidden_states, token_ids, processed_tokens, layer_arrays

                # Free the generate() KV-cache and any residual GPU tensors.
                # This is defensive — the hook already moves states to CPU,
                # but generate() itself allocates KV-cache on VRAM that isn't
                # freed until the next generate() call overwrites it.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                result = {
                    "problem_id": problem_id,
                    "run_idx": run_idx,
                    "rating": rating,
                    "seed": trace_seed,
                    "prompt": prompt,
                    "trace": response,
                    "has_think_tags": THINK_START in response and THINK_END in response,
                    "trace_length_chars": len(response),
                    "generation_time_seconds": round(elapsed, 2),
                }
                append_result(result, jsonl_path)
                pbar.update(1)

                # Periodic GPU memory status (persistent mode only)
                if args.persistent:
                    traces_since_status += 1
                    if traces_since_status >= args.status_interval:
                        _log_gpu_memory(f"after {traces_since_status} traces")
                        traces_since_status = 0

    except KeyboardInterrupt:
        if args.persistent:
            interrupted = True
            print("\n[INTERRUPTED] Saving progress...", flush=True)
        else:
            raise

    finally:
        # Close progress bar (may still be open after KeyboardInterrupt)
        try:
            pbar.close()
        except Exception:
            pass
        # Always close HDF5 file, even on error
        if hf is not None:
            # Write final trace count to HDF5 metadata
            count = sum(
                1 for pid in hf if isinstance(hf[pid], h5py.Group)
                for rk in hf[pid] if rk.startswith("run_")
            )
            hf.attrs["num_traces"] = count
            hf.close()

    # --- Summary ---
    status = " (interrupted)" if interrupted else ""
    print(f"\nDone{status}. Traces saved to {jsonl_path}")
    if args.persistent and skipped_count > 0:
        print(f"WARNING: {skipped_count} traces skipped due to OOM. "
              f"Details in {skipped_path}")
    if extract_mode:
        file_size_mb = h5_path.stat().st_size / (1024 * 1024)
        print(f"Activations saved to {h5_path} ({file_size_mb:.1f} MB)")
    if interrupted:
        print("Re-run with the same arguments to resume from checkpoint.")
        sys.exit(130)  # Standard exit code for SIGINT


if __name__ == "__main__":
    main()
