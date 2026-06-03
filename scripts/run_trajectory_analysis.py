"""Run trajectory analysis and hidden-state probing.

Stages 03 and 04 from the analysis pipeline.  Both stages operate on
pre-computed activations (HDF5) and CoT analysis outputs, and do NOT
require IRT model fitting.

Usage
-----
    python scripts/run_trajectory_analysis.py --config pipeline/code/deepseek-r1-7b
    python scripts/run_trajectory_analysis.py --config pipeline/code/deepseek-r1-7b --stages 03
    python scripts/run_trajectory_analysis.py --config pipeline/code/deepseek-r1-7b --stages 04 --workers 12
    python scripts/run_trajectory_analysis.py --config pipeline/code/deepseek-r1-7b --stages 03 04 \
        --pooled-irt-path data/results/pooled_irt/code/pooled_difficulties.parquet
    python scripts/run_trajectory_analysis.py --config pipeline/code/deepseek-r1-7b --stages 03 04 --force

Stages
------
    03  Activation trajectory metrics, random-walk baselines, probing,
        trajectory-difficulty correlations, CoT-difficulty correlations
    04  Hidden-state probing (generation-stage, CPU — reads HDF5)

Difficulty source is controlled by ``--pooled-irt-path``: when provided,
difficulty-dependent outputs use pooled IRT (suffix ``_pooled``); when
absent, native difficulty is used (suffix ``_native``).

Stage 03 and 04 depend on Stage 02 outputs (cot_analysis.parquet).
"""

# --- Suppress BLAS multi-threading BEFORE numpy/torch are imported ----------
# ProcessPoolExecutor uses fork() on Linux; forked children inherit the
# parent's already-initialized BLAS.  Setting env vars *before* the first
# numpy import ensures BLAS starts with 1 thread in both parent and children.
import os as _os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")
del _os, _var

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.parallel_workers import (
    _worker_init,
    _worker_init_blas2,
    compute_baseline_worker,
    compute_metrics_and_traces_worker,
    compute_metrics_worker,
    compute_probe_worker,
)
from src.probing import compute_surface_features, train_binned_difficulty_probe

# ---------------------------------------------------------------------------
# Path configuration — set from --config in main()
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Module-level globals; set by main() from the loaded config.
DOMAIN             = "codeforces"
MODEL_NAME         = None
TRACES_PATH        = None
PROBLEMS_PATH      = None
ACTIVATIONS_PATH   = None
COT_ANALYSIS_PATH  = None
TRAJECTORY_PATH    = None
PROBE_RESULTS_PATH = None
TRACES_OUTPUT_PATH = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _cache_exists(*paths: Path) -> bool:
    """Return True if all *paths* exist and are non-empty files."""
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def _ensure_dir(path: Path) -> None:
    """Create parent directories for *path* if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as ``M:SS`` or ``H:MM:SS``."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _compat_columns(df):
    """Rename legacy think_length -> reasoning_length if needed."""
    if "think_length" in df.columns and "reasoning_length" not in df.columns:
        df = df.rename(columns={"think_length": "reasoning_length"})
    return df


def _load_difficulties(problems_df, domain, pooled_irt_path=None):
    """Load difficulty values as a Series indexed by problem_id.

    Parameters
    ----------
    problems_df : DataFrame
        Problems dataframe with ``join_key`` column.
    domain : str
        ``"math"`` or ``"codeforces"``.
    pooled_irt_path : str or Path, optional
        Path to ``pooled_difficulties.parquet``.  When provided, pooled IRT
        difficulty is used; otherwise native difficulty (MATH levels or
        Glicko-2 ratings).

    Returns
    -------
    pd.Series
        Difficulty values indexed by problem id.
    """
    if pooled_irt_path:
        print(f"  [difficulty] Using POOLED IRT from {pooled_irt_path}")
        pooled = pd.read_parquet(pooled_irt_path)
        return pooled.set_index("item_id")["difficulty"]

    if domain == "math":
        print("  [difficulty] Using MATH levels (level_int)")
        return problems_df.set_index("join_key")["level_int"].astype(float)
    elif domain == "sat":
        print("  [difficulty] Using SAT num_clauses")
        return problems_df.set_index("join_key")["num_clauses"].astype(float)
    else:
        print("  [difficulty] Using Glicko-2 ratings (unnorm_rating)")
        return problems_df.set_index("join_key")["unnorm_rating"]


# ---------------------------------------------------------------------------
# Stage 03: Activation trajectory analysis
# ---------------------------------------------------------------------------

def run_stage_03(n_workers: int, force: bool, pooled_irt_path=None) -> None:
    """Stage 03: Activation trajectory analysis.

    Steps:
      1. Compute trajectory metrics (parallel, chunked from HDF5)
      2. Compute random-walk baselines (parallel)
      3. Compute z-scores (vectorized)
      4. Run probing experiments (parallel)
      5. Save trajectory_summaries.parquet and probe_results_{suffix}.parquet
      6. Compute trajectory-difficulty and CoT-difficulty correlations
      7. Embed correlations metadata into trajectory_summaries.parquet
    """
    difficulty_suffix = "pooled" if pooled_irt_path else "native"

    print("\n" + "=" * 70)
    print(f"  STAGE 03: Activation Trajectory Analysis (difficulty: {difficulty_suffix})")
    print("=" * 70)
    t0 = time.time()

    # Dependency check
    if not COT_ANALYSIS_PATH.exists():
        print(f"  ERROR: Required input {COT_ANALYSIS_PATH} not found.")
        print("  Run stage 02 first.")
        return

    if not ACTIVATIONS_PATH.exists():
        print(f"  ERROR: Required input {ACTIVATIONS_PATH} not found.")
        print("  Run the GPU pipeline first (scripts/run_pipeline.py).")
        return

    # Size each chunk to ~2x worker count: enough to keep all workers busy
    # while limiting pickle serialization cost (~10 MB per item).
    CHUNK_SIZE = max(n_workers * 4, 16)

    # Probe output path depends on difficulty source
    actual_probe_path = (PROBE_RESULTS_PATH.parent / f"probe_results_{difficulty_suffix}.parquet"
                         if PROBE_RESULTS_PATH else None)

    traj_cache_ok = not force and _cache_exists(TRAJECTORY_PATH)
    probe_cache_ok = not force and actual_probe_path and _cache_exists(actual_probe_path)
    traces_cache_ok = (not force and TRACES_OUTPUT_PATH
                       and _cache_exists(TRACES_OUTPUT_PATH))

    # Check whether correlations metadata already exists in the base file
    correlations_key = f"correlations_{difficulty_suffix}"

    def _has_metadata_key(path, key):
        if not path or not path.exists():
            return False
        import pyarrow.parquet as pq
        meta = pq.read_schema(path).metadata or {}
        return key.encode() in meta

    traj_summary_ok = not force and _has_metadata_key(TRAJECTORY_PATH, correlations_key)

    # When TRACES_OUTPUT_PATH is configured, traj cache is only fully OK
    # if traces are also cached (they require the same HDF5 pass)
    need_hdf5_pass = not traj_cache_ok or (TRACES_OUTPUT_PATH and not traces_cache_ok)

    if traj_cache_ok and probe_cache_ok and traj_summary_ok and (
        not TRACES_OUTPUT_PATH or traces_cache_ok
    ):
        probe_name = actual_probe_path.name if actual_probe_path else "N/A"
        traces_name = TRACES_OUTPUT_PATH.name if TRACES_OUTPUT_PATH else "N/A"
        print(f"  SKIP (cached): {TRAJECTORY_PATH.name}, {probe_name}, "
              f"{traces_name}, {correlations_key} metadata")
        print("  Use --force to recompute.")
        return

    problems = pd.read_parquet(PROBLEMS_PATH)
    cot_analysis = _compat_columns(pd.read_parquet(COT_ANALYSIS_PATH))

    # Build reasoning boundary lookup for trajectory truncation
    if "reasoning_end_frac" in cot_analysis.columns:
        boundary_lookup = dict(zip(
            zip(cot_analysis["problem_id"].astype(str),
                cot_analysis["run_idx"].astype(int)),
            cot_analysis["reasoning_end_frac"].values,
        ))
        n_truncated = sum(1 for v in boundary_lookup.values() if v < 1.0)
        print(f"  Reasoning boundaries: {n_truncated}/{len(boundary_lookup)} "
              f"traces will be truncated")
    else:
        boundary_lookup = None
        print("  No reasoning_end_frac in cot_analysis — using full trajectories")

    hf = h5py.File(ACTIVATIONS_PATH, "r")
    try:
        layers = list(hf.attrs["layers"])
        hidden_dim = int(hf.attrs["hidden_dim"])
        n_traces = int(hf.attrs["num_traces"])

        # ---- Step 1: Trajectory metrics (and traces if configured) ----
        compute_traces = TRACES_OUTPUT_PATH is not None and need_hdf5_pass
        if traj_cache_ok and not compute_traces:
            print("[03.1] Loading cached trajectory summaries...")
            traj_df = pd.read_parquet(TRAJECTORY_PATH)
            print(f"  Loaded {len(traj_df)} rows")
        else:
            print(f"[03.1] Computing trajectory metrics ({n_workers} workers)...")

            def _load_items_from_hdf5(hf_handle, layer_list, boundaries=None):
                """Generator yielding (states_np, metadata) from HDF5.

                Runs in the main process — HDF5 handles are not safe for
                concurrent reads, so data is pre-loaded here.

                Parameters
                ----------
                boundaries : dict, optional
                    Mapping ``(problem_id, run_idx) -> reasoning_end_frac``.
                    When provided, states are truncated to the reasoning phase.
                """
                for problem_id in hf_handle.keys():
                    problem_grp = hf_handle[problem_id]
                    for run_key in problem_grp:
                        run_grp = problem_grp[run_key]
                        run_idx = int(run_key.split("_")[1])
                        rating = float(run_grp.attrs.get("rating", -1))

                        # Pre-read all layers for this run to improve HDF5
                        # read locality (reduces group traversal / seek overhead)
                        layer_data = {}
                        for layer_idx in layer_list:
                            ds_name = f"layer_{layer_idx}"
                            if ds_name in run_grp:
                                layer_data[layer_idx] = run_grp[ds_name][:].astype(
                                    np.float32
                                )

                        for layer_idx, states_np in layer_data.items():
                            n_total = states_np.shape[0]
                            if n_total < 3:
                                continue

                            # Truncate to reasoning phase
                            if boundaries is not None:
                                frac = boundaries.get(
                                    (str(problem_id), run_idx), 1.0
                                )
                                end_tok = max(
                                    3, min(int(frac * n_total), n_total)
                                )
                                states_np = states_np[:end_tok]
                            else:
                                frac = 1.0
                                end_tok = n_total

                            meta = {
                                "problem_id": problem_id,
                                "run_idx": run_idx,
                                "layer": layer_idx,
                                "rating": rating,
                                "reasoning_n_tokens": end_tok,
                                "reasoning_end_frac": frac,
                            }
                            yield (states_np, meta)

            expected_total = n_traces * len(layers)
            summaries = []
            trace_rows = [] if compute_traces else None
            item_gen = _load_items_from_hdf5(hf, layers, boundaries=boundary_lookup)

            worker_fn = compute_metrics_and_traces_worker if compute_traces else compute_metrics_worker
            desc = "  Trajectory metrics+traces" if compute_traces else "  Trajectory metrics"

            with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
                pbar = tqdm(
                    desc=desc,
                    total=expected_total,
                    unit="item",
                )
                while True:
                    chunk = list(islice(item_gen, CHUNK_SIZE))
                    if not chunk:
                        break
                    results = list(pool.map(worker_fn, chunk, chunksize=1))
                    if compute_traces:
                        for scalar_dict, traces_dict in results:
                            summaries.append(scalar_dict)
                            trace_rows.append(traces_dict)
                    else:
                        summaries.extend(results)
                    pbar.update(len(results))
                    del chunk, results
                pbar.close()

            traj_df = pd.DataFrame(summaries)
            print(f"  Computed {len(traj_df)} trajectory summaries")

            # Save trajectory traces if computed
            if compute_traces and trace_rows:
                import pyarrow as pa
                import pyarrow.parquet as pq

                stride = int(hf.attrs.get("stride", 10))

                pid_arr = pa.array([r["problem_id"] for r in trace_rows], type=pa.string())
                ridx_arr = pa.array([r["run_idx"] for r in trace_rows], type=pa.int64())
                layer_arr = pa.array([r["layer"] for r in trace_rows], type=pa.int64())
                nsteps_arr = pa.array([r["n_steps"] for r in trace_rows], type=pa.int64())
                vel_arr = pa.array([r["velocity"].tolist() for r in trace_rows],
                                   type=pa.list_(pa.float32()))
                curv_arr = pa.array([r["curvature"].tolist() for r in trace_rows],
                                    type=pa.list_(pa.float32()))
                cos_arr = pa.array([r["cosine_turn"].tolist() for r in trace_rows],
                                   type=pa.list_(pa.float32()))

                table = pa.table({
                    "problem_id": pid_arr, "run_idx": ridx_arr, "layer": layer_arr,
                    "n_steps": nsteps_arr, "velocity": vel_arr,
                    "curvature": curv_arr, "cosine_turn": cos_arr,
                })
                meta = table.schema.metadata or {}
                meta[b"stride"] = json.dumps(stride).encode()
                meta[b"model_name"] = MODEL_NAME.encode()
                table = table.replace_schema_metadata(meta)

                _ensure_dir(TRACES_OUTPUT_PATH)
                pq.write_table(table, TRACES_OUTPUT_PATH)
                print(f"  Saved trajectory traces: {TRACES_OUTPUT_PATH}")
                del trace_rows

            # ---- Step 2: Random-walk baselines ----
            n_baseline_workers = min(n_workers, max(1, os.cpu_count() // 2))
            print(f"[03.2] Computing random-walk baselines ({n_baseline_workers} workers, 2 BLAS threads each)...")
            unique_lengths = sorted(traj_df["num_steps"].unique())
            baseline_args = [(length, hidden_dim) for length in unique_lengths]

            with ProcessPoolExecutor(max_workers=n_baseline_workers, initializer=_worker_init_blas2) as pool:
                baseline_results = list(tqdm(
                    pool.map(compute_baseline_worker, baseline_args),
                    total=len(unique_lengths),
                    desc="  Random-walk baselines",
                ))
            null_cache = dict(baseline_results)

            # ---- Step 3: Z-scores (vectorized) ----
            print("[03.3] Computing z-scores...")
            null_rows = [{"num_steps": length, **stats_dict}
                         for length, stats_dict in null_cache.items()]
            null_df = pd.DataFrame(null_rows)

            null_cols = [c for c in null_df.columns if c != "num_steps"]
            traj_df.drop(
                columns=[c for c in null_cols if c in traj_df.columns],
                inplace=True,
                errors="ignore",
            )
            traj_df = traj_df.merge(null_df, on="num_steps", how="left")

            def _safe_zscore(obs, null_mean, null_std):
                return np.where(null_std > 1e-12, (obs - null_mean) / null_std, 0.0)

            traj_df["curvature_zscore"] = _safe_zscore(
                traj_df["curvature_mean"],
                traj_df["curvature_mean_null_mean"],
                traj_df["curvature_mean_null_std"],
            )
            traj_df["directness_zscore"] = _safe_zscore(
                traj_df["directness"],
                traj_df["directness_null_mean"],
                traj_df["directness_null_std"],
            )

            # Drop intermediate null columns
            null_merge_cols = [c for c in traj_df.columns
                              if c.endswith("_null_mean") or c.endswith("_null_std")]
            traj_df.drop(columns=null_merge_cols, inplace=True)

            # Save trajectory summaries
            _ensure_dir(TRAJECTORY_PATH)
            traj_df.to_parquet(TRAJECTORY_PATH, index=False)
            print(f"  Saved {TRAJECTORY_PATH}")
    finally:
        hf.close()

    # Load difficulty values once for steps 4 and 7
    need_difficulties = not probe_cache_ok or not traj_summary_ok
    diff_series = None
    if need_difficulties:
        diff_series = _load_difficulties(problems, DOMAIN, pooled_irt_path)

    # ---- Step 4: Probing ----
    if probe_cache_ok:
        print(f"[03.4] SKIP probing (cached): {actual_probe_path.name}")
    else:
        print(f"[03.4] Running probing experiments ({n_workers} workers, "
              f"difficulty: {difficulty_suffix})...")
        from sklearn.preprocessing import StandardScaler

        # Merge difficulty into trajectory data
        traj_with_meta = traj_df.copy()
        traj_with_meta["difficulty"] = traj_with_meta["problem_id"].map(diff_series)
        traj_with_meta = traj_with_meta.dropna(subset=["difficulty"])
        difficulty_col = "difficulty"

        feature_cols = [
            "directness", "curvature_mean", "twonn_dim", "pca_dim_90",
        ]

        all_feature_sets = {
            "raw": feature_cols,
        }

        N_PERMUTATIONS = 100

        # Build probe task arguments
        probe_args = []
        for layer in layers:
            layer_data = traj_with_meta[traj_with_meta["layer"] == layer].dropna(
                subset=[difficulty_col]
            )
            layer_merged = layer_data.merge(
                cot_analysis[["problem_id", "run_idx", "correct"]],
                on=["problem_id", "run_idx"],
                how="inner",
            )
            if len(layer_merged) < 10:
                continue

            y_rating = layer_merged[difficulty_col].values.astype(float)
            y_correct = layer_merged["correct"].astype(int).values
            majority_baseline = float(max(y_correct.mean(), 1 - y_correct.mean()))

            groups = layer_merged["problem_id"].values

            for feat_name, feat_cols_list in all_feature_sets.items():
                available = [c for c in feat_cols_list if c in layer_merged.columns]
                if not available:
                    continue
                X = layer_merged[available].values

                probe_args.append((
                    layer, feat_name, X,
                    y_rating, y_correct,
                    majority_baseline, N_PERMUTATIONS, groups,
                ))

        print(f"  Running {len(probe_args)} probe combinations "
              f"({len(layers)} layers x {len(all_feature_sets)} feature sets)...")

        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
            probe_results_list = list(tqdm(
                pool.map(compute_probe_worker, probe_args),
                total=len(probe_args),
                desc="  Probing",
            ))

        probe_results = pd.DataFrame(probe_results_list)

        # ---- Step 5: Surface-feature baseline probing ----
        print("[03.5] Computing surface-feature baseline probes...")
        prompt_col = "formatted_prompt" if "formatted_prompt" in problems.columns else None
        if prompt_col is not None:
            surface_X = compute_surface_features(problems, prompt_column=prompt_col)
            # Align surface features with difficulty series
            problem_ids = problems["join_key"].values
            valid_mask = pd.Series(problem_ids).isin(diff_series.index)
            X_valid = surface_X[valid_mask]
            surface_y = diff_series.reindex(pd.Series(problem_ids)[valid_mask]).values

            # Ridge regression on surface features
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_score
            scaler = StandardScaler()
            X_surf_scaled = scaler.fit_transform(X_valid)
            ridge = Ridge(alpha=1.0)
            r2_scores = cross_val_score(ridge, X_surf_scaled, surface_y, cv=5, scoring="r2")
            surface_r2 = float(r2_scores.mean())
            print(f"  Surface-feature baseline R2: {surface_r2:.4f}")

            # Add to probe_results as a special row per layer
            surface_rows = [
                {
                    "layer": layer,
                    "features": "surface_baseline",
                    "r2": surface_r2,
                    "r2_perm_mean": 0.0,
                    "r2_perm_std": 0.0,
                    "r2_pvalue": 0.0,
                    "accuracy": float("nan"),
                    "majority_baseline": float("nan"),
                    "roc_auc": float("nan"),
                }
                for layer in layers
            ]
            probe_results = pd.concat(
                [probe_results, pd.DataFrame(surface_rows)], ignore_index=True
            )
        else:
            print("  SKIP: No formatted_prompt column in problems DataFrame")

        # ---- Step 6: Discretised difficulty bin classification ----
        print("[03.6] Computing binned difficulty classification probes...")
        binned_results = []
        for layer in layers:
            layer_data = traj_with_meta[traj_with_meta["layer"] == layer].dropna(
                subset=[difficulty_col]
            )
            if len(layer_data) < 20:
                continue
            y_diff = layer_data[difficulty_col].values.astype(float)
            layer_groups = layer_data["problem_id"].values
            # Use raw trajectory features
            available = [c for c in feature_cols if c in layer_data.columns]
            if not available:
                continue
            X = layer_data[available].values
            try:
                binned = train_binned_difficulty_probe(X, y_diff, n_bins=5, n_folds=5, groups=layer_groups)
                binned_results.append({
                    "layer": layer,
                    "features": "raw_binned",
                    "bin_accuracy": binned["accuracy_mean"],
                    "bin_accuracy_std": binned["accuracy_std"],
                    "chance_level": binned["chance_level"],
                })
            except Exception as e:
                print(f"    Layer {layer}: binned probe failed ({e})")

        if binned_results:
            binned_df = pd.DataFrame(binned_results)
            # Merge binned accuracy into probe_results
            probe_results = probe_results.merge(
                binned_df[["layer", "bin_accuracy", "bin_accuracy_std", "chance_level"]],
                on="layer",
                how="left",
            )
            best_bin = binned_df.loc[binned_df["bin_accuracy"].idxmax()]
            print(f"  Best binned accuracy: {best_bin['bin_accuracy']:.4f} "
                  f"(layer {int(best_bin['layer'])}, chance={best_bin['chance_level']:.4f})")

        _ensure_dir(actual_probe_path)
        probe_results.to_parquet(actual_probe_path, index=False)
        print(f"  Saved {actual_probe_path}")

        # Print summary
        for feat_name in all_feature_sets:
            sub = probe_results[probe_results["features"] == feat_name]
            if sub.empty:
                continue
            print(f"\n  --- {feat_name.upper()} features ---")
            for _, row in sub.iterrows():
                sig = "*" if row["r2_pvalue"] < 0.05 else "ns"
                print(f"    Layer {int(row['layer']):2d}: "
                      f"R2={row['r2']:.4f} (perm={row['r2_perm_mean']:.4f} {sig}), "
                      f"Acc={row['accuracy']:.4f}, AUC={row['roc_auc']:.4f}")

    # ---- Step 7: Trajectory-difficulty and CoT-difficulty correlations ----
    if not traj_summary_ok:
        print(f"[03.7] Computing difficulty correlations ({difficulty_suffix})...")
        from scipy import stats as sp_stats

        # -- Trajectory-difficulty correlations --
        traj_corr_df = traj_df.copy()
        traj_corr_df["difficulty"] = traj_corr_df["problem_id"].map(diff_series)
        traj_valid = traj_corr_df.dropna(subset=["difficulty"])

        metrics = ["directness", "curvature_mean", "twonn_dim", "pca_dim_90"]
        traj_corrs = {}
        for metric in metrics:
            if metric not in traj_valid.columns:
                continue
            valid = traj_valid.dropna(subset=[metric])
            if len(valid) >= 3:
                rho, p = sp_stats.spearmanr(valid["difficulty"], valid[metric])
                traj_corrs[metric] = {
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "n": len(valid),
                }
                print(f"    {metric}: rho = {rho:.4f} (p={p:.4e})")

        # -- CoT pattern-difficulty correlations --
        cot_corrs = {}
        if COT_ANALYSIS_PATH.exists():
            cot_df = cot_analysis.copy()
            pid_col = "problem_id" if "problem_id" in cot_df.columns else "join_key"
            cot_df["difficulty"] = cot_df[pid_col].map(diff_series)
            cot_valid = cot_df.dropna(subset=["difficulty"])

            pattern_cols = [c for c in cot_valid.columns
                            if c.startswith("pattern_") or c in (
                                "backtracking_norm", "verification_norm",
                                "strategy_shift_norm", "uncertainty_norm",
                                "correction_norm", "exploration_norm",
                                "repetition_score", "reasoning_length",  # reasoning-phase length (chars)
                            )]

            for col in pattern_cols:
                if col not in cot_valid.columns:
                    continue
                valid = cot_valid.dropna(subset=[col])
                if len(valid) >= 3 and valid[col].nunique() > 1:
                    rho, p = sp_stats.spearmanr(valid["difficulty"], valid[col])
                    cot_corrs[col] = {
                        "spearman_rho": float(rho),
                        "spearman_p": float(p),
                        "n": len(valid),
                    }

            if cot_corrs:
                top_patterns = sorted(cot_corrs.items(),
                                       key=lambda x: abs(x[1]["spearman_rho"]),
                                       reverse=True)[:5]
                print("  Top CoT-difficulty correlations:")
                for name, corr in top_patterns:
                    print(f"    {name}: rho = {corr['spearman_rho']:.4f}")

        # Embed correlations metadata into the base trajectory_summaries.parquet
        correlations_data = {
            "trajectory_correlations": traj_corrs,
            "cot_pattern_correlations": cot_corrs,
            "difficulty_source": difficulty_suffix,
            "n_items_with_difficulty": int(diff_series.notna().sum()),
        }
        if pooled_irt_path:
            correlations_data["pooled_irt_path"] = str(pooled_irt_path)

        import pyarrow.parquet as pq
        table = pq.read_table(TRAJECTORY_PATH)
        existing_meta = table.schema.metadata or {}
        existing_meta[correlations_key.encode()] = json.dumps(correlations_data).encode()
        table = table.replace_schema_metadata(existing_meta)
        pq.write_table(table, TRAJECTORY_PATH)
        print(f"  Embedded {correlations_key} metadata in {TRAJECTORY_PATH.name}")
    else:
        print(f"[03.7] SKIP correlations (cached): {correlations_key} in "
              f"{TRAJECTORY_PATH.name}")

    elapsed = time.time() - t0
    print(f"\n  Stage 03 complete in {_fmt_elapsed(elapsed)}")


# ---------------------------------------------------------------------------
# Stage 04: Hidden-state probing (generation-stage, CPU)
# ---------------------------------------------------------------------------

def run_stage_04(n_workers: int, force: bool, pooled_irt_path=None) -> None:
    """Stage 04: Hidden-state probing of raw generation-stage activations.

    THE HERO EXPERIMENT.  For each (layer, generation-progress) cell in a
    grid, fits a Ridge regression probe predicting difficulty from the raw
    hidden-state vector.  Produces a layer x generation-progress R2 heatmap.

    Steps:
      1. Load difficulty values (native or pooled via --pooled-irt-path)
      2. Build probing heatmap (layer x position grid)
      3. Build correctness-split heatmaps
      4. Extract difficulty direction from peak-R2 cell
      5. Surface-feature baseline comparison
      6. Save outputs
    """
    difficulty_suffix = "pooled" if pooled_irt_path else "native"

    print("\n" + "=" * 70)
    print(f"  STAGE 04: Hidden-State Probing (Generation-Stage, difficulty: {difficulty_suffix})")
    print("=" * 70)
    t0 = time.time()

    # Resolve output paths with correct suffix
    analysis_base = COT_ANALYSIS_PATH.parent.parent
    probing_dir = analysis_base / "probing"
    heatmap_path = probing_dir / f"generation_heatmap_{difficulty_suffix}.parquet"
    directions_path = probing_dir / f"difficulty_directions_{difficulty_suffix}.npz"

    # Cache check
    if not force and _cache_exists(heatmap_path):
        print(f"  SKIP (cached): {heatmap_path.name}")
        print("  Use --force to recompute.")
        return

    # Dependency check
    if ACTIVATIONS_PATH is None:
        print("  ERROR: Stage 04 requires paths.pipeline.activations "
              "(not available in eval_only configs)")
        return
    if not ACTIVATIONS_PATH.exists():
        print(f"  ERROR: Activations file not found: {ACTIVATIONS_PATH}")
        return

    # ---- Step 1: Load difficulty values ----
    print(f"[04.1] Loading difficulty values (source: {difficulty_suffix})...")
    problems = pd.read_parquet(PROBLEMS_PATH)
    diff_series = _load_difficulties(problems, DOMAIN, pooled_irt_path)
    print(f"  Loaded {len(diff_series)} difficulty values")

    # ---- Step 2: Determine layers and extract states ONCE ----
    print("[04.2] Reading HDF5 metadata and extracting states...")
    with h5py.File(ACTIVATIONS_PATH, "r") as hf:
        layer_indices = list(hf.attrs.get("layers", [0]))
    print(f"  Layers: {layer_indices}")

    from src.hidden_state_probing import (
        build_probing_heatmap,
        build_probing_heatmap_by_correctness,
        extract_generation_stage_states,
        extract_peak_direction,
    )

    # Build reasoning boundary lookup for trajectory truncation
    boundary_lookup = None
    if COT_ANALYSIS_PATH.exists():
        import pyarrow.parquet as pq
        _pf = pq.ParquetFile(COT_ANALYSIS_PATH)
        _available_cols = [f.name for f in _pf.schema_arrow]
        if "reasoning_end_frac" in _available_cols:
            _cot_for_bounds = pd.read_parquet(
                COT_ANALYSIS_PATH,
                columns=["problem_id", "run_idx", "reasoning_end_frac"],
            )
            boundary_lookup = dict(zip(
                zip(_cot_for_bounds["problem_id"].astype(str),
                    _cot_for_bounds["run_idx"].astype(int)),
                _cot_for_bounds["reasoning_end_frac"].values,
            ))
            n_truncated = sum(1 for v in boundary_lookup.values() if v < 1.0)
            print(f"  Reasoning boundaries: {n_truncated}/{len(boundary_lookup)} "
                  f"traces will be truncated")
            del _cot_for_bounds
        else:
            print("  No reasoning_end_frac in cot_analysis — using full trajectories")
        del _pf

    # Single HDF5 read — reused by heatmap, correctness split, and direction
    _states, _metadata = extract_generation_stage_states(
        ACTIVATIONS_PATH, layer_indices, n_positions=10,
        reasoning_boundaries=boundary_lookup,
    )

    # ---- Step 3: Build probing heatmap ----
    print("[04.3] Building probing heatmap (layer x generation-progress)...")
    heatmap_df = build_probing_heatmap(
        hdf5_path=ACTIVATIONS_PATH,
        difficulties=diff_series,
        layer_indices=layer_indices,
        n_positions=10,
        n_folds=5,
        n_permutations=100,
        n_workers=n_workers,
        _states=_states,
        _metadata=_metadata,
    )

    if heatmap_df.empty:
        print("  WARNING: Heatmap is empty (no valid traces). Stopping Stage 04.")
        return

    heatmap_df["subset"] = "all"
    all_heatmaps = [heatmap_df]

    peak = heatmap_df.loc[heatmap_df["r2_mean"].idxmax()]
    print(f"  Peak R2 = {peak['r2_mean']:.4f} at layer {int(peak['layer'])}, "
          f"position {int(peak['position_idx'])}/{10} "
          f"({peak['position_frac']:.0%} through generation)")

    # ---- Step 4: Correctness interaction ----
    print("[04.4] Building correctness-split heatmaps...")
    if COT_ANALYSIS_PATH.exists():
        correctness_df = pd.read_parquet(
            COT_ANALYSIS_PATH, columns=["problem_id", "run_idx", "correct"]
        )

        heatmaps_by_corr = build_probing_heatmap_by_correctness(
            hdf5_path=ACTIVATIONS_PATH,
            difficulties=diff_series,
            correctness=correctness_df,
            layer_indices=layer_indices,
            n_positions=10,
            n_workers=n_workers,
            _states=_states,
            _metadata=_metadata,
        )

        for label, hm in heatmaps_by_corr.items():
            if not hm.empty:
                hm["subset"] = label
                all_heatmaps.append(hm)
                peak_corr = hm.loc[hm["r2_mean"].idxmax()]
                print(f"  {label}: peak R2 = {peak_corr['r2_mean']:.4f}")
    else:
        print("  SKIP correctness split (correctness results not available)")

    # ---- Step 5: Extract difficulty direction ----
    print("[04.5] Extracting difficulty direction from peak cell...")
    try:
        direction_result = extract_peak_direction(
            heatmap_df=heatmap_df,
            hdf5_path=ACTIVATIONS_PATH,
            difficulties=diff_series,
            _states=_states,
            _metadata=_metadata,
        )
        _ensure_dir(directions_path)
        np.savez_compressed(
            directions_path,
            direction=direction_result["direction"],
            peak_layer=direction_result["peak_layer"],
            peak_position_idx=direction_result["peak_position_idx"],
            peak_position_frac=direction_result["peak_position_frac"],
            peak_r2=direction_result["peak_r2"],
            projection_mean=direction_result["projection_mean"],
            projection_std=direction_result["projection_std"],
        )
        print(f"  Direction: layer {direction_result['peak_layer']}, "
              f"pos {direction_result['peak_position_idx']}, "
              f"R2 = {direction_result['peak_r2']:.4f}")
        print(f"  Projection stats: mean={direction_result['projection_mean']:.4f}, "
              f"std={direction_result['projection_std']:.4f}")
        print(f"  -> {directions_path}")
    except Exception as e:
        print(f"  WARNING: Could not extract direction: {e}")

    # ---- Step 6: Surface-feature baseline ----
    print("[04.6] Running surface-feature baseline comparison...")
    surface_baseline = None
    try:
        from src.probing import compute_surface_features, train_regression_probe
        from src.models import format_prompt, format_math_generation_prompt

        # Format prompts for surface features
        if DOMAIN == "math":
            problems["formatted_prompt"] = problems.apply(
                lambda row: format_math_generation_prompt(
                    row.get("problem", row.get("problem_text", ""))
                ), axis=1,
            )
        else:
            problems["formatted_prompt"] = problems.apply(
                lambda row: format_prompt(row.get("description", "")), axis=1,
            )

        X_surface = compute_surface_features(problems, "formatted_prompt")

        # Align with difficulty series
        problem_ids = problems["join_key"].values

        valid_mask = pd.Series(problem_ids).isin(diff_series.index)
        X_valid = X_surface[valid_mask]
        y_valid = diff_series.reindex(pd.Series(problem_ids)[valid_mask]).values

        if len(X_valid) >= 10:
            surface_result = train_regression_probe(X_valid, y_valid, n_folds=5)
            print(f"  Surface-feature baseline R2 = {surface_result['r2_mean']:.4f} "
                  f"(+/-{surface_result['r2_std']:.4f})")
            print(f"  Hidden-state peak R2 = {peak['r2_mean']:.4f} "
                  f"(improvement: {peak['r2_mean'] - surface_result['r2_mean']:.4f})")

            surface_baseline = {
                "surface_r2_mean": surface_result["r2_mean"],
                "surface_r2_std": surface_result["r2_std"],
                "hidden_state_peak_r2": float(peak["r2_mean"]),
                "hidden_state_peak_layer": int(peak["layer"]),
                "hidden_state_peak_position": int(peak["position_idx"]),
            }
    except Exception as e:
        print(f"  WARNING: Surface-feature baseline failed: {e}")
        surface_baseline = None

    # ---- Step 7: Save merged heatmap with subset column ----
    merged_heatmap = pd.concat(all_heatmaps, ignore_index=True)
    _ensure_dir(heatmap_path)

    # Embed surface baseline as parquet metadata
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(merged_heatmap)
    metadata = table.schema.metadata or {}
    if surface_baseline is not None:
        metadata[b"surface_baseline"] = json.dumps(surface_baseline).encode()
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, heatmap_path)
    print(f"  Saved merged heatmap ({len(merged_heatmap)} rows, "
          f"subsets: {merged_heatmap['subset'].unique().tolist()})")
    print(f"  -> {heatmap_path}")

    elapsed = time.time() - t0
    print(f"\n  Stage 04 complete in {_fmt_elapsed(elapsed)}")


# ---------------------------------------------------------------------------
# CLI and main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trajectory analysis and hidden-state probing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["03", "04"],
        default=["03", "04"],
        help="Which stages to run (default: 03). "
             "Stage 03 = trajectory metrics + probing + correlations. "
             "Stage 04 = generation-stage hidden-state probing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes "
             "(default: cpu_count - 1).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cached output files already exist.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config: name (e.g. 'pipeline/code/deepseek-r1-7b') or path to YAML. "
             "Sets all input/output paths for this model. "
             "Available: see configs/ directory.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["codeforces", "math", "sat"],
        default=None,
        help="Problem domain: 'codeforces', 'math', or 'sat'. "
             "Auto-detected from config path if not specified "
             "(e.g. 'math/deepseek-r1-7b' -> math).",
    )
    parser.add_argument(
        "--pooled-irt-path",
        type=str,
        default=None,
        help="Path to pooled_difficulties.parquet. When provided, "
             "Stage 03 probing and Stage 04 heatmap use pooled IRT "
             "difficulty (output suffix: _pooled). Without this, native "
             "difficulty is used (suffix: _native).",
    )
    args = parser.parse_args()

    # Auto-detect domain from config path if not explicitly set
    if args.domain is None:
        if "math/" in args.config:
            args.domain = "math"
        elif "sat/" in args.config:
            args.domain = "sat"
        else:
            args.domain = "codeforces"

    if args.workers is None:
        args.workers = max(1, (os.cpu_count() or 4) - 1)

    return args


def main():
    args = parse_args()

    # Load config and set path globals
    global DOMAIN, MODEL_NAME, TRACES_PATH, PROBLEMS_PATH, ACTIVATIONS_PATH
    global COT_ANALYSIS_PATH
    global TRAJECTORY_PATH, PROBE_RESULTS_PATH, TRACES_OUTPUT_PATH

    DOMAIN = args.domain
    cfg = load_config(args.config)
    MODEL_NAME         = cfg["model"]["name"]
    PROBLEMS_PATH      = cfg["paths"]["problems"]
    TRACES_PATH        = cfg["paths"]["pipeline"]["cot_traces"]
    ACTIVATIONS_PATH   = cfg["paths"]["pipeline"].get("activations")     # None for eval_only
    COT_ANALYSIS_PATH  = cfg["paths"]["analysis"]["cot_analysis"]
    TRAJECTORY_PATH    = cfg["paths"]["analysis"].get("trajectories")    # None for eval_only
    PROBE_RESULTS_PATH = cfg["paths"]["analysis"].get("probe_results")   # None for eval_only
    TRACES_OUTPUT_PATH = cfg["paths"]["analysis"].get("trajectory_traces")  # None if not configured

    # Resolve pooled IRT path to absolute before any use
    pooled_irt_path = None
    if args.pooled_irt_path:
        pooled_irt_path = Path(args.pooled_irt_path)
        if not pooled_irt_path.is_absolute():
            pooled_irt_path = PROJECT_ROOT / pooled_irt_path
        if not pooled_irt_path.exists():
            print(f"ERROR: Pooled IRT file not found: {pooled_irt_path}")
            sys.exit(1)

    difficulty_label = "pooled" if pooled_irt_path else "native"
    print("=" * 70)
    print("  IRT Latent Difficulty — Trajectory Analysis Pipeline")
    print("=" * 70)
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Domain:     {DOMAIN}")
    print(f"  Config:     {args.config}")
    print(f"  Stages:     {', '.join(args.stages)}")
    print(f"  Difficulty: {difficulty_label}")
    print(f"  Workers:    {args.workers}")
    print(f"  Force:      {args.force}")
    print(f"  Traces:     {TRACES_PATH}")
    print(f"  Activ.:     {ACTIVATIONS_PATH or '(not configured)'}")
    if pooled_irt_path:
        print(f"  Pooled IRT: {pooled_irt_path}")
    print()

    # Validate that required input files exist
    required_inputs = [PROBLEMS_PATH]
    if "03" in args.stages:
        if ACTIVATIONS_PATH is None:
            print("ERROR: Stage 03 requires paths.pipeline.activations "
                  "(not available in eval_only configs)")
            sys.exit(1)
        required_inputs.append(ACTIVATIONS_PATH)
    if "04" in args.stages:
        if ACTIVATIONS_PATH is None:
            print("ERROR: Stage 04 requires paths.pipeline.activations "
                  "(not available in eval_only configs)")
            sys.exit(1)
        required_inputs.append(ACTIVATIONS_PATH)

    for p in required_inputs:
        if not p.exists():
            print(f"ERROR: Required input file not found: {p}")
            sys.exit(1)

    t_total = time.time()

    # Run stages in dependency order
    if "03" in args.stages:
        run_stage_03(args.workers, args.force, pooled_irt_path=pooled_irt_path)

    if "04" in args.stages:
        run_stage_04(args.workers, args.force, pooled_irt_path=pooled_irt_path)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  All stages complete in {_fmt_elapsed(elapsed)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
