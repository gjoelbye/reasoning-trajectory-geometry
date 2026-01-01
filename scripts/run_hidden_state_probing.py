"""GPU-based hidden-state probing: prompt-stage, random-init baseline,
and length probing.

This script performs probing experiments.  All modes **require a GPU** and the
model loaded in memory.

Modes:

  - **prompt**: Extract hidden state at the last prompt token for each layer,
    then probe for difficulty.  Tests whether difficulty is encoded *before*
    generation begins (i.e. from the prompt alone).

  - **random-baseline**: Load a randomly initialized model of the same
    architecture (no pretrained weights), repeat the prompt-stage extraction,
    and probe.  If the pretrained model has higher R², the encoding reflects
    learned representations, not trivial architectural properties.

  - **length**: Predict log(mean reasoning_length) from prompt-stage hidden states
    (GPU, reuses prompt-stage extraction).

  - **all**: Run prompt, length, and random-baseline sequentially.

Usage
-----
    # Prompt-stage probing (requires GPU + model loaded)
    python scripts/run_hidden_state_probing.py \\
        --config code/deepseek-r1-14b --mode prompt

    # Random-init baseline (requires GPU)
    python scripts/run_hidden_state_probing.py \\
        --config code/deepseek-r1-14b --mode random-baseline

    # Length probing (GPU, prompt-stage extraction)
    python scripts/run_hidden_state_probing.py \\
        --config code/deepseek-r1-14b --mode length

    # All modes
    python scripts/run_hidden_state_probing.py \\
        --config code/deepseek-r1-14b --mode all

    # With pooled IRT difficulty
    python scripts/run_hidden_state_probing.py \\
        --config code/deepseek-r1-14b --mode prompt \\
        --pooled-irt-path data/results/pooled_irt/code/pooled_difficulties.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _cache_exists(*paths):
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def _load_difficulties(problems_df, domain, pooled_irt_path=None):
    """Load difficulty values as a Series indexed by problem_id."""
    if pooled_irt_path:
        print(f"  [difficulty] Using POOLED IRT from {pooled_irt_path}")
        pooled = pd.read_parquet(pooled_irt_path)
        return pooled.set_index("item_id")["difficulty"]

    if domain == "math":
        print("  [difficulty] Using MATH levels (level_int) — no pooled IRT path provided")
        return problems_df.set_index("join_key")["level_int"].astype(float)
    elif domain == "sat":
        print("  [difficulty] Using SAT num_clauses")
        return problems_df.set_index("join_key")["num_clauses"].astype(float)
    else:
        print("  [difficulty] Using Glicko-2 ratings (unnorm_rating)")
        return problems_df.set_index("join_key")["unnorm_rating"]


def _format_prompts(problems_df, domain):
    """Format prompts for each problem."""
    from src.models import format_prompt, format_math_generation_prompt

    prompts = []
    problem_ids = []

    for idx, row in problems_df.iterrows():
        if domain == "math":
            prompt = format_math_generation_prompt(
                row.get("problem", row.get("problem_text", ""))
            )
            pid = row["join_key"]
        elif domain == "sat":
            from src.models import format_sat_generation_prompt
            prompt = format_sat_generation_prompt(
                row.get("formatted_prompt", row.get("scenario", ""))
            )
            pid = row["join_key"]
        else:
            prompt = format_prompt(row.get("description", ""))
            pid = row["join_key"]
        prompts.append(prompt)
        problem_ids.append(str(pid))

    return prompts, problem_ids


# ---------------------------------------------------------------------------
# Probing modes
# ---------------------------------------------------------------------------

def run_prompt_probing(model, tokenizer, problems_df, difficulties,
                       layer_indices, domain, output_dir, n_folds=5,
                       force=False, difficulty_suffix="native"):
    """Extract prompt-stage hidden states and probe for difficulty."""
    output_path = output_dir / f"prompt_stage_probes_{difficulty_suffix}.parquet"
    if not force and _cache_exists(output_path):
        print(f"  SKIP (cached): {output_path.name}")
        print("  Use --force to recompute.")
        return pd.read_parquet(output_path)

    from src.hidden_state_probing import (
        extract_prompt_hidden_states,
        probe_prompt_stage,
    )

    prompts, problem_ids = _format_prompts(problems_df, domain)

    # Filter to problems with valid difficulties
    valid_mask = pd.Series(problem_ids).isin(difficulties.index)
    prompts_valid = [p for p, v in zip(prompts, valid_mask) if v]
    pids_valid = [p for p, v in zip(problem_ids, valid_mask) if v]
    diffs_valid = difficulties.reindex(pids_valid).values

    print(f"  Extracting prompt-stage hidden states for {len(prompts_valid)} problems...")
    print(f"  Layers: {layer_indices}")

    layer_states = extract_prompt_hidden_states(
        model, tokenizer, prompts_valid, layer_indices,
    )

    print(f"  Probing for difficulty at each layer...")
    probe_results = probe_prompt_stage(
        layer_states, diffs_valid, n_folds=n_folds,
    )

    # Build output DataFrame
    rows = []
    for layer_idx, result in probe_results.items():
        rows.append({
            "layer": layer_idx,
            "r2_mean": result["r2_mean"],
            "r2_std": result["r2_std"],
            "mse_mean": result["mse_mean"],
            "spearman_rho": result["spearman_rho"],
            "spearman_p": result["spearman_p"],
            "n_problems": len(diffs_valid),
        })

    df = pd.DataFrame(rows)
    output_path = output_dir / f"prompt_stage_probes_{difficulty_suffix}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  -> {output_path}")

    # Print summary
    if not df.empty:
        peak = df.loc[df["r2_mean"].idxmax()]
        print(f"  Peak R² = {peak['r2_mean']:.4f} at layer {int(peak['layer'])}")

    return df


def run_random_baseline(model_id, problems_df, difficulties,
                        layer_indices, domain, output_dir, n_folds=5,
                        force=False, difficulty_suffix="native"):
    """Load randomly initialized model and probe — baseline control."""
    output_path = output_dir / f"random_init_baseline_{difficulty_suffix}.parquet"
    if not force and _cache_exists(output_path):
        print(f"  SKIP (cached): {output_path.name}")
        print("  Use --force to recompute.")
        return pd.read_parquet(output_path)

    print("  Loading randomly initialized model...")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import torch

    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create model with random weights
    with torch.no_grad():
        random_model = AutoModelForCausalLM.from_config(config)
        random_model = random_model.half().cuda()

    # We need to wrap this in an nnsight-compatible way
    # Since nnsight may not handle random models well, we use a simpler approach
    from src.hidden_state_probing import probe_prompt_stage

    prompts, problem_ids = _format_prompts(problems_df, domain)

    valid_mask = pd.Series(problem_ids).isin(difficulties.index)
    prompts_valid = [p for p, v in zip(prompts, valid_mask) if v]
    pids_valid = [p for p, v in zip(problem_ids, valid_mask) if v]
    diffs_valid = difficulties.reindex(pids_valid).values

    print(f"  Extracting hidden states from random model for {len(prompts_valid)} problems...")

    # Manual extraction without nnsight
    layer_states = {idx: [] for idx in layer_indices}

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    prompt_iter = enumerate(prompts_valid)
    if tqdm is not None:
        prompt_iter = tqdm(prompt_iter, total=len(prompts_valid),
                           desc="  Extracting random-init states", unit="prompt")

    for _i, prompt in prompt_iter:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].cuda()

        with torch.no_grad():
            outputs = random_model(input_ids, output_hidden_states=True)

        for idx in layer_indices:
            # HuggingFace hidden_states includes embedding at [0],
            # so transformer block idx output is at [idx + 1]
            hf_idx = idx + 1
            if hf_idx < len(outputs.hidden_states):
                h = outputs.hidden_states[hf_idx][0, -1, :].cpu().numpy()
            else:
                h = np.zeros(config.hidden_size, dtype=np.float32)
            layer_states[idx].append(h)

    layer_states_np = {idx: np.stack(vecs) for idx, vecs in layer_states.items()}

    # Probe
    probe_results = probe_prompt_stage(
        layer_states_np, diffs_valid, n_folds=n_folds,
    )

    rows = []
    for layer_idx, result in probe_results.items():
        rows.append({
            "layer": layer_idx,
            "r2_mean": result["r2_mean"],
            "r2_std": result["r2_std"],
            "mse_mean": result["mse_mean"],
            "spearman_rho": result["spearman_rho"],
            "spearman_p": result["spearman_p"],
            "n_problems": len(diffs_valid),
        })

    df = pd.DataFrame(rows)
    output_path = output_dir / f"random_init_baseline_{difficulty_suffix}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  -> {output_path}")

    if not df.empty:
        peak = df.loc[df["r2_mean"].idxmax()]
        print(f"  Random baseline peak R² = {peak['r2_mean']:.4f} at layer {int(peak['layer'])}")

    # Clean up
    del random_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cot_analysis(cfg):
    """Load lightweight subset of cot_analysis.parquet."""
    path = cfg["paths"]["analysis"]["cot_analysis"]
    return pd.read_parquet(
        path,
        columns=["problem_id", "run_idx", "correct",
                 "trace_length_chars", "reasoning_end_frac"],
    )


# ---------------------------------------------------------------------------
# Length probing (GPU, prompt-stage extraction)
# ---------------------------------------------------------------------------

def run_length_probing(model, tokenizer, problems_df, cfg,
                       layer_indices, domain, output_dir,
                       n_folds=5, force=False):
    """Predict log(mean reasoning_length) from prompt-stage hidden states.

    reasoning_length = trace_length_chars * reasoning_end_frac, which works
    for all models (not just reasoning models with <think> tags).

    Requires model loaded (GPU).  Reuses prompt-stage extraction.

    Output: length_probes.parquet
    """
    output_path = output_dir / "length_probes.parquet"
    if not force and _cache_exists(output_path):
        print(f"  SKIP (cached): {output_path.name}")
        print("  Use --force to recompute.")
        return pd.read_parquet(output_path)

    from src.hidden_state_probing import (
        extract_prompt_hidden_states,
        probe_prompt_stage,
    )
    from scipy.stats import spearmanr

    # Compute y = log(mean reasoning_length per problem) from cot_analysis
    cot = _load_cot_analysis(cfg)
    cot["problem_id"] = cot["problem_id"].astype(str)
    cot["reasoning_length"] = cot["trace_length_chars"] * cot["reasoning_end_frac"]

    mean_length = (
        cot.groupby("problem_id")["reasoning_length"]
        .mean()
        .rename("mean_reasoning_length")
    )
    # Drop problems with zero or missing length
    mean_length = mean_length[mean_length > 0]
    log_length = np.log(mean_length).rename("log_mean_reasoning_length")

    prompts, problem_ids = _format_prompts(problems_df, domain)

    # Filter to problems with valid log-length targets
    valid_mask = pd.Series(problem_ids).isin(log_length.index)
    prompts_valid = [p for p, v in zip(prompts, valid_mask) if v]
    pids_valid = [p for p, v in zip(problem_ids, valid_mask) if v]
    y = log_length.reindex(pids_valid).values

    if len(prompts_valid) == 0:
        print("  SKIP: no problems with reasoning_length > 0")
        return pd.DataFrame()

    print(f"  Extracting prompt-stage hidden states for {len(prompts_valid)} problems...")
    print(f"  Layers: {layer_indices}")

    layer_states = extract_prompt_hidden_states(
        model, tokenizer, prompts_valid, layer_indices,
    )

    # Probe each layer (regular KFold — problem-level, no repeated measures)
    from src.probing import train_regression_probe

    rows = []
    for layer_idx in layer_indices:
        X = layer_states[layer_idx]
        if len(X) < n_folds * 2:
            continue

        probe = train_regression_probe(
            X, y, n_folds=n_folds,
        )
        rho, rho_p = spearmanr(y, probe["oof_predictions"])

        rows.append({
            "layer": layer_idx,
            "r2_mean": probe["r2_mean"],
            "r2_std": probe["r2_std"],
            "mse_mean": probe["mse_mean"],
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "n_problems": len(y),
        })
        print(f"    layer={layer_idx:3d}  R2={probe['r2_mean']:.4f}  "
              f"rho={rho:.4f}")

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  -> {output_path}")

    if not df.empty:
        peak = df.loc[df["r2_mean"].idxmax()]
        print(f"  Peak R2 = {peak['r2_mean']:.4f} at layer {int(peak['layer'])}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU-based hidden-state probing (prompt-stage and random baseline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Model config name (e.g. 'code/deepseek-r1-14b').",
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["prompt", "random-baseline", "length", "all"],
        help="Which probing mode to run.",
    )
    parser.add_argument(
        "--pooled-irt-path", type=str, default=None,
        help="Path to pooled_difficulties.parquet for difficulty target.",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of CV folds for Ridge probe (default: 5).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if cached outputs exist.",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true",
        help="Load model in 8-bit quantization (halves VRAM usage). "
             "Required for 32B+ models on 80GB GPUs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from src.config import load_config
    from src.models import default_extraction_layers

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    model_id = cfg["model"]["model_id"]
    num_layers = cfg["model"]["num_layers"]
    problems_path = cfg["paths"]["problems"]

    # Domain auto-detection
    if "math/" in args.config:
        domain = "math"
    elif "sat/" in args.config:
        domain = "sat"
    else:
        domain = "codeforces"

    # Output directory — data/{domain}/{model}/probing/
    # cot_analysis path is .../cot_analysis/cot_analysis.parquet; go up two levels
    analysis_base = cfg["paths"]["analysis"]["cot_analysis"].parent.parent
    output_dir = analysis_base / "probing"

    # Layer indices
    layer_indices = default_extraction_layers(num_layers)

    print("=" * 70)
    print("  Hidden-State Probing (GPU)")
    print("=" * 70)
    print(f"  Model:    {model_name} ({model_id})")
    print(f"  Domain:   {domain}")
    print(f"  Mode:     {args.mode}")
    print(f"  Layers:   {layer_indices}")
    print(f"  Output:   {output_dir}")
    print()

    t0 = time.time()

    # Determine difficulty suffix
    difficulty_suffix = "pooled" if args.pooled_irt_path else "native"

    # Load problems and difficulties
    problems = pd.read_parquet(problems_path)
    difficulties = _load_difficulties(problems, domain, args.pooled_irt_path)
    print(f"  Loaded {len(difficulties)} difficulty values (source: {difficulty_suffix})")

    # Modes that need the model loaded (prompt, length)
    if args.mode in ("prompt", "length", "all"):
        # Load model once for all GPU modes
        from src.models import load_model
        model = load_model(model_id, load_in_8bit=args.load_in_8bit)
        tokenizer = model.tokenizer

        if args.mode in ("prompt", "all"):
            print("\n" + "-" * 70)
            print("  Prompt-Stage Probing")
            print("-" * 70)

            prompt_df = run_prompt_probing(
                model, tokenizer, problems, difficulties,
                layer_indices, domain, output_dir,
                n_folds=args.n_folds,
                force=args.force,
                difficulty_suffix=difficulty_suffix,
            )

            # Surface-feature baseline comparison
            if not prompt_df.empty:
                print("\n  Running surface-feature baseline comparison...")
                try:
                    from src.probing import compute_surface_features, train_regression_probe
                    from src.models import format_prompt, format_math_generation_prompt

                    if domain == "math":
                        problems["formatted_prompt"] = problems.apply(
                            lambda row: format_math_generation_prompt(
                                row.get("problem", row.get("problem_text", ""))
                            ), axis=1,
                        )
                    elif domain == "sat":
                        from src.models import format_sat_generation_prompt
                        problems["formatted_prompt"] = problems.apply(
                            lambda row: format_sat_generation_prompt(
                                row.get("formatted_prompt", row.get("scenario", ""))
                            ), axis=1,
                        )
                    else:
                        problems["formatted_prompt"] = problems.apply(
                            lambda row: format_prompt(row.get("description", "")), axis=1,
                        )

                    X_surface = compute_surface_features(problems, "formatted_prompt")

                    problem_ids = problems["join_key"].values
                    valid_mask = pd.Series(problem_ids).isin(difficulties.index)
                    X_valid = X_surface[valid_mask]
                    y_valid = difficulties.reindex(
                        pd.Series(problem_ids)[valid_mask]
                    ).values

                    if len(X_valid) >= 10:
                        surface_result = train_regression_probe(X_valid, y_valid, n_folds=5)

                        peak = prompt_df.loc[prompt_df["r2_mean"].idxmax()]
                        print(f"  Surface-feature baseline R² = {surface_result['r2_mean']:.4f} "
                              f"(±{surface_result['r2_std']:.4f})")
                        print(f"  Prompt-stage peak R² = {peak['r2_mean']:.4f} "
                              f"(improvement: {peak['r2_mean'] - surface_result['r2_mean']:.4f})")

                        comparison = {
                            "surface_r2_mean": surface_result["r2_mean"],
                            "surface_r2_std": surface_result["r2_std"],
                            "prompt_stage_peak_r2": float(peak["r2_mean"]),
                            "prompt_stage_peak_layer": int(peak["layer"]),
                        }
                        # Re-save prompt probes parquet with surface baseline as metadata
                        import pyarrow as pa
                        import pyarrow.parquet as pq
                        probes_path = output_dir / f"prompt_stage_probes_{difficulty_suffix}.parquet"
                        if probes_path.exists():
                            table = pq.read_table(probes_path)
                            metadata = table.schema.metadata or {}
                            metadata[b"surface_baseline"] = json.dumps(comparison).encode()
                            table = table.replace_schema_metadata(metadata)
                            pq.write_table(table, probes_path)
                            print(f"  Surface baseline embedded in {probes_path.name}")
                except Exception as e:
                    print(f"  WARNING: Surface-feature baseline failed: {e}")

        if args.mode in ("length", "all"):
            print("\n" + "-" * 70)
            print("  Length Probing (log mean reasoning length)")
            print("-" * 70)

            length_df = run_length_probing(
                model, tokenizer, problems, cfg,
                layer_indices, domain, output_dir,
                n_folds=args.n_folds,
                force=args.force,
            )

        # Clean up model to free GPU memory
        del model, tokenizer
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    if args.mode in ("random-baseline", "all"):
        print("\n" + "-" * 70)
        print("  Random-Init Baseline")
        print("-" * 70)

        random_df = run_random_baseline(
            model_id, problems, difficulties,
            layer_indices, domain, output_dir,
            n_folds=args.n_folds,
            force=args.force,
            difficulty_suffix=difficulty_suffix,
        )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Hidden-state probing complete in {_fmt_elapsed(elapsed)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
