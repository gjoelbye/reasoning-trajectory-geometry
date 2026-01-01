"""Hidden-state probing for difficulty encoding.

This module implements the project's **hero experiment**: probing raw hidden-
state vectors at different (layer, generation-progress) positions to determine
where and when LLMs encode problem difficulty during chain-of-thought
reasoning.

Two probing modes are supported:

1. **Generation-stage probing** (CPU-only): reads pre-extracted hidden states
   from HDF5 activation files.  For each (layer, position_fraction) cell in a
   grid, fits a Ridge regression probe predicting difficulty from the raw
   hidden-state vector.  Produces a *layer × generation-progress* R² heatmap.
   Hidden states are **averaged across runs per problem** before probing
   (sharpens signal), and **trace length is residualized** from both X and y
   to control for the length confound.

2. **Prompt-stage probing** (requires GPU + model): runs a single forward pass
   per problem to extract the hidden state at the last prompt token.  This
   tests whether difficulty is already encoded *before* generation begins.

The module **reuses** existing infrastructure rather than duplicating logic:

- ``src.probing.train_regression_probe()`` for Ridge + cross-validation
- ``src.probing.selectivity_index()`` for control comparisons
- ``src.probing.compute_surface_features()`` for surface-feature baselines
- ``src.extraction.subsample_positions()`` for uniform position sampling
- ``src.extraction.extract_hidden_states_single_forward()`` for prompt-stage GPU extraction

Usage
-----
    from src.hidden_state_probing import (
        build_probing_heatmap,
        extract_peak_direction,
        extract_prompt_hidden_states,
        probe_prompt_stage,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Generation-stage extraction from HDF5
# ---------------------------------------------------------------------------

def extract_generation_stage_states(
    hdf5_path: str | Path,
    layer_indices: List[int],
    n_positions: int = 10,
    min_tokens: int = 3,
    reasoning_boundaries: Optional[Dict[Tuple[str, int], float]] = None,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], pd.DataFrame]:
    """Extract hidden states at evenly-spaced generation positions from HDF5.

    For each trace (problem_id, run_idx), samples ``n_positions`` evenly-
    spaced token positions using ``subsample_positions()``.  For each
    (layer, position_index) pair, collects the hidden-state vector.

    Parameters
    ----------
    hdf5_path : str or Path
        Path to the HDF5 activations file.
    layer_indices : list of int
        Which layers to extract (e.g. ``[0, 6, 13, 20, 27]``).
    n_positions : int
        Number of evenly-spaced positions to sample per trace (default 10).
    min_tokens : int
        Skip traces with fewer than this many tokens.
    reasoning_boundaries : dict, optional
        Mapping ``(problem_id, run_idx) -> reasoning_end_frac`` (0–1).
        When provided, tokens are truncated to the reasoning phase before
        position sampling.

    Returns
    -------
    states : dict[(layer_idx, position_idx) -> ndarray(n_traces, hidden_dim)]
        Hidden-state arrays indexed by (layer, position_index).
        ``position_idx`` ranges from 0 to ``n_positions - 1``.
    metadata : DataFrame
        One row per trace with columns [problem_id, run_idx, trace_length].
        Row order matches the first axis of each ``states`` array.
    """
    import h5py
    from src.extraction import subsample_positions

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    states: Dict[Tuple[int, int], list] = {}
    for layer_idx in layer_indices:
        for pos_idx in range(n_positions):
            states[(layer_idx, pos_idx)] = []

    meta_rows = []

    with h5py.File(hdf5_path, "r") as hf:
        problem_keys = list(hf.keys())
        n_problems = len(problem_keys)

        prob_iter = problem_keys
        if tqdm is not None:
            prob_iter = tqdm(prob_iter, desc="  Reading HDF5", unit="problem")

        for problem_id in prob_iter:
            problem_grp = hf[problem_id]
            for run_key in problem_grp:
                run_grp = problem_grp[run_key]
                run_idx = int(run_key.split("_")[1])

                # Check that we have at least one layer dataset
                first_layer = f"layer_{layer_indices[0]}"
                if first_layer not in run_grp:
                    continue

                n_tokens = run_grp[first_layer].shape[0]
                if n_tokens < min_tokens:
                    continue

                # Truncate to reasoning phase if boundaries provided
                if reasoning_boundaries is not None:
                    frac = reasoning_boundaries.get(
                        (str(problem_id), run_idx), 1.0
                    )
                    n_tokens = max(
                        min_tokens, min(int(frac * n_tokens), n_tokens)
                    )

                # Get evenly-spaced positions
                positions = subsample_positions(
                    n_tokens, n_samples=n_positions,
                    include_first_last=True,
                )
                # Ensure exactly n_positions (pad or trim)
                if len(positions) > n_positions:
                    positions = positions[:n_positions]
                elif len(positions) < n_positions:
                    positions = positions + [positions[-1]] * (n_positions - len(positions))

                meta_rows.append({
                    "problem_id": str(problem_id),
                    "run_idx": run_idx,
                    "trace_length": n_tokens,
                })

                # Build sorted unique indices for efficient HDF5 reads.
                # Reading only the needed rows avoids loading the full
                # (n_tokens, hidden_dim) array which can be hundreds of MB.
                sorted_unique = sorted(set(positions))
                pos_to_local = {p: i for i, p in enumerate(sorted_unique)}

                for layer_idx in layer_indices:
                    ds_name = f"layer_{layer_idx}"
                    if ds_name not in run_grp:
                        import warnings
                        hidden_dim = run_grp[first_layer].shape[1]
                        warnings.warn(
                            f"Missing {ds_name} for {problem_id}/{run_key}; "
                            f"filling with NaN (will be excluded from averages)"
                        )
                        for pos_idx in range(n_positions):
                            states[(layer_idx, pos_idx)].append(
                                np.full(hidden_dim, np.nan, dtype=np.float32)
                            )
                        continue

                    # Fancy-index: read only the ~10 rows we need
                    selected = run_grp[ds_name][sorted_unique].astype(np.float32)
                    for pos_idx, tok_pos in enumerate(positions):
                        states[(layer_idx, pos_idx)].append(
                            selected[pos_to_local[tok_pos]]
                        )

    print(f"  HDF5: {n_problems} problems, {len(meta_rows)} traces extracted")

    # Stack into arrays
    stacked: Dict[Tuple[int, int], np.ndarray] = {}
    for key, arrays in states.items():
        if arrays:
            stacked[key] = np.stack(arrays, axis=0)
        else:
            stacked[key] = np.empty((0, 0), dtype=np.float32)

    if meta_rows:
        metadata = pd.DataFrame(meta_rows)
    else:
        metadata = pd.DataFrame(columns=["problem_id", "run_idx", "trace_length"])
    return stacked, metadata


def average_states_by_problem(
    states: Dict[Tuple[int, int], np.ndarray],
    metadata: pd.DataFrame,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], pd.DataFrame]:
    """Average hidden states across runs for each problem.

    Groups traces by ``problem_id`` and averages the hidden-state vectors
    within each group.  This reduces the effective sample size to the
    number of unique problems but sharpens the signal by averaging away
    within-problem noise.

    Parameters
    ----------
    states : dict[(layer_idx, position_idx) -> ndarray(n_traces, hidden_dim)]
        Hidden-state arrays from ``extract_generation_stage_states()``.
    metadata : DataFrame
        One row per trace with columns [problem_id, run_idx, trace_length].
        Row order matches the first axis of each ``states`` array.

    Returns
    -------
    states_avg : dict[(layer_idx, position_idx) -> ndarray(n_problems, hidden_dim)]
        Averaged hidden-state arrays, one row per unique problem.
    metadata_avg : DataFrame
        One row per problem with columns [problem_id, trace_length].
        ``trace_length`` is the mean across runs.
    """
    problem_ids = sorted(metadata["problem_id"].unique())
    pid_to_idx = {pid: i for i, pid in enumerate(problem_ids)}

    # Build group indices: map each problem to its row indices in metadata
    groups: Dict[str, list] = {pid: [] for pid in problem_ids}
    for row_idx, pid in enumerate(metadata["problem_id"].values):
        groups[pid].append(row_idx)

    n_problems = len(problem_ids)
    states_avg: Dict[Tuple[int, int], np.ndarray] = {}

    for key, arr in states.items():
        hidden_dim = arr.shape[1] if arr.ndim == 2 else 0
        avg = np.empty((n_problems, hidden_dim), dtype=np.float32)
        for pid in problem_ids:
            idx = groups[pid]
            avg[pid_to_idx[pid]] = np.nanmean(arr[idx], axis=0)
        states_avg[key] = avg

    # Build metadata_avg: one row per problem, mean trace_length
    meta_rows = []
    for pid in problem_ids:
        idx = groups[pid]
        mean_length = metadata.iloc[idx]["trace_length"].mean()
        meta_rows.append({
            "problem_id": pid,
            "trace_length": mean_length,
        })
    metadata_avg = pd.DataFrame(meta_rows)

    return states_avg, metadata_avg


# ---------------------------------------------------------------------------
# Internal: SVD-accelerated permutation testing
# ---------------------------------------------------------------------------

def _permutation_test_svd(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    n_folds: int,
    n_permutations: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> List[float]:
    """Fast permutation testing via SVD precomputation.

    Instead of fitting a full Ridge regression for each permutation,
    precompute the SVD of X_train once per fold. Each permuted y then
    costs only a matrix-vector product (~3000x faster).

    The alpha should be fixed (e.g. from the real fit's RidgeCV).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold, KFold

    if groups is not None:
        cv = GroupKFold(n_splits=n_folds)
        splits = list(cv.split(X, y, groups))
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(cv.split(X, y))

    # Precompute SVD and projection matrices per fold (expensive, done once)
    fold_cache = []
    for train_idx, test_idx in splits:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        U, s, Vt = np.linalg.svd(X_train, full_matrices=False)
        d_alpha = s / (s ** 2 + alpha)

        # Precompute X_test @ V so each permutation is just dot products
        XtV = X_test @ Vt.T  # (n_test, k)

        fold_cache.append({
            "U": U, "d_alpha": d_alpha, "XtV": XtV,
            "train_idx": train_idx, "test_idx": test_idx,
        })

    rng = np.random.RandomState(seed)
    perm_r2s = []

    # Pre-compute group-level permutation mapping if needed.
    # Permute which label each group gets, not individual observations.
    if groups is not None:
        unique_groups = np.unique(groups)
        group_labels = np.array([y[groups == g][0] for g in unique_groups])

    for _ in range(n_permutations):
        if groups is not None:
            perm_labels = rng.permutation(group_labels)
            group_to_perm = dict(zip(unique_groups, perm_labels))
            y_perm = np.array([group_to_perm[g] for g in groups])
        else:
            y_perm = rng.permutation(y)
        fold_r2s = []

        for fc in fold_cache:
            y_train = y_perm[fc["train_idx"]]
            y_test = y_perm[fc["test_idx"]]

            # Ridge prediction via SVD: ŷ = XtV @ (d_alpha * (U^T y_train))
            coeff = fc["d_alpha"] * (fc["U"].T @ y_train)
            y_pred = fc["XtV"] @ coeff

            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            fold_r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)

        perm_r2s.append(float(np.mean(fold_r2s)))

    return perm_r2s


# ---------------------------------------------------------------------------
# Internal: probe a single (layer, position) cell
# ---------------------------------------------------------------------------

def _probe_single_cell(
    X: np.ndarray,
    y: np.ndarray,
    layer_idx: int,
    pos_idx: int,
    n_positions: int,
    n_folds: int,
    alpha: float,
    seed: int,
    n_permutations: int,
    groups: Optional[np.ndarray] = None,
    trace_lengths: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Fit Ridge probe for one (layer, position) cell. Returns result dict or None."""
    from src.probing import train_regression_probe
    from scipy.stats import spearmanr

    n_unique = len(np.unique(groups)) if groups is not None else len(y)
    if n_unique < n_folds * 2:
        return None

    # Length residualization: regress out trace_length from X and y
    if trace_lengths is not None:
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        L = trace_lengths.reshape(-1, 1)
        lr.fit(L, X); X = X - lr.predict(L)
        lr.fit(L, y.reshape(-1, 1)); y = y - lr.predict(L).ravel()

    probe_result = train_regression_probe(
        X, y, n_folds=n_folds, alpha=alpha, seed=seed, groups=groups,
    )

    rho, rho_p = spearmanr(y, probe_result["oof_predictions"])

    row = {
        "layer": layer_idx,
        "position_idx": pos_idx,
        "position_frac": pos_idx / max(1, n_positions - 1),
        "r2_mean": probe_result["r2_mean"],
        "r2_std": probe_result["r2_std"],
        "mse_mean": probe_result["mse_mean"],
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "n_traces": len(X),
        "n_problems": int(n_unique),
        "_model": probe_result["model"],
    }

    if n_permutations > 0:
        # Extract alpha from real fit for permutations (standard practice:
        # Cule & De Iorio 2011).  Avoids re-selecting alpha 100 times.
        fitted_reg = probe_result["model"].named_steps["reg"]
        perm_alpha = getattr(fitted_reg, "alpha_", None) or alpha or 1.0

        perm_r2s = _permutation_test_svd(
            X, y, alpha=perm_alpha, n_folds=n_folds,
            n_permutations=n_permutations, seed=seed, groups=groups,
        )
        row["r2_perm_mean"] = float(np.mean(perm_r2s))
        row["r2_perm_std"] = float(np.std(perm_r2s))
        row["r2_perm_p"] = float(
            np.mean(np.array(perm_r2s) >= probe_result["r2_mean"])
        )

    return row


# ---------------------------------------------------------------------------
# Shared-memory globals for fork-based multiprocessing
# ---------------------------------------------------------------------------
# On Linux (fork), child processes inherit the parent's memory via
# copy-on-write.  By setting these globals before Pool creation,
# workers can read the large hidden-state arrays without serialization.

_shared_states: Optional[Dict] = None
_shared_valid_indices: Optional[list] = None
_shared_y: Optional[np.ndarray] = None
_shared_trace_lengths: Optional[np.ndarray] = None
_shared_groups: Optional[np.ndarray] = None


def _probe_cell_worker(args: tuple) -> Optional[dict]:
    """Multiprocessing worker that reads X from shared memory."""
    (layer_idx, pos_idx, n_positions,
     n_folds, alpha, seed, n_permutations) = args
    global _shared_states, _shared_valid_indices, _shared_y
    global _shared_trace_lengths, _shared_groups
    X = _shared_states[(layer_idx, pos_idx)][_shared_valid_indices]
    return _probe_single_cell(
        X, _shared_y, layer_idx, pos_idx, n_positions,
        n_folds, alpha, seed, n_permutations,
        groups=_shared_groups,
        trace_lengths=_shared_trace_lengths,
    )


# ---------------------------------------------------------------------------
# Generation-stage probing heatmap (THE HERO EXPERIMENT)
# ---------------------------------------------------------------------------

def _prepare_probing_data(
    states: Dict[Tuple[int, int], np.ndarray],
    metadata: pd.DataFrame,
    difficulties: np.ndarray | pd.Series,
    problem_ids: Optional[List[str]] = None,
    n_folds: int = 5,
) -> Tuple[pd.DataFrame, list]:
    """Assign difficulties to metadata and compute valid trace indices.

    Returns (metadata_with_difficulty, valid_indices) or raises if insufficient data.
    """
    if isinstance(difficulties, pd.Series):
        diff_lookup = difficulties.to_dict()
    else:
        if problem_ids is None:
            raise ValueError(
                "problem_ids must be provided when difficulties is an ndarray"
            )
        diff_lookup = dict(zip(problem_ids, difficulties))

    metadata = metadata.copy()
    metadata["difficulty"] = metadata["problem_id"].map(diff_lookup)
    n_before = len(metadata)
    metadata = metadata.dropna(subset=["difficulty"])
    n_dropped = n_before - len(metadata)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped}/{n_before} traces with missing difficulty")

    if len(metadata) < n_folds * 2:
        import warnings
        warnings.warn(
            f"Only {len(metadata)} traces with valid difficulties "
            f"(need at least {n_folds * 2})."
        )
        return metadata, []

    return metadata, metadata.index.tolist()



def build_probing_heatmap(
    hdf5_path: str | Path,
    difficulties: np.ndarray | pd.Series,
    layer_indices: List[int],
    n_positions: int = 10,
    n_folds: int = 5,
    alpha: float = None,
    seed: int = 42,
    problem_ids: Optional[List[str]] = None,
    n_permutations: int = 0,
    n_workers: int = 1,
    _states: Optional[Dict] = None,
    _metadata: Optional[pd.DataFrame] = None,
    reasoning_boundaries: Optional[Dict[Tuple[str, int], float]] = None,
) -> pd.DataFrame:
    """Build the layer x generation-progress probing R^2 heatmap.

    **This is the project's hero experiment.**  For each cell in a
    (layer, position_fraction) grid, fits a Ridge regression probe
    predicting difficulty from the raw hidden-state vector.

    Hidden states are **averaged across runs per problem** before probing
    (sharpens signal by averaging within-problem noise, effective n =
    number of unique problems).  **Trace length is residualized** from
    both X and y before fitting to control for the length confound.

    Parameters
    ----------
    hdf5_path : str or Path
        Path to the HDF5 activations file.
    difficulties : array-like
        Difficulty values indexed by problem_id.  If a pandas Series,
        the index is used as problem IDs.  If an ndarray, ``problem_ids``
        must be provided.
    layer_indices : list of int
        Which layers to probe.
    n_positions : int
        Number of evenly-spaced generation positions.
    n_folds : int
        Number of cross-validation folds for Ridge regression.
    alpha : float
        Ridge regularization strength.
    seed : int
        Random seed.
    problem_ids : list of str, optional
        Problem IDs corresponding to ``difficulties`` (if ndarray).
    n_permutations : int
        If > 0, also run this many permutation tests per cell.
    n_workers : int
        Number of parallel workers for probing cells (1 = serial).
    _states : dict, optional
        Pre-extracted states (skip HDF5 read). For internal reuse.
    _metadata : DataFrame, optional
        Pre-extracted metadata (skip HDF5 read). For internal reuse.

    Returns
    -------
    DataFrame with columns [layer, position_idx, position_frac, r2_mean,
    r2_std, mse_mean, n_traces] and optionally [r2_perm_mean, r2_perm_std]
    if permutation tests were run.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    empty_cols = [
        "layer", "position_idx", "position_frac",
        "r2_mean", "r2_std", "mse_mean", "n_traces",
    ]

    # Extract states (or reuse pre-extracted)
    if _states is not None and _metadata is not None:
        states, metadata = _states, _metadata
    else:
        print("  Extracting hidden states from HDF5...")
        states, metadata = extract_generation_stage_states(
            hdf5_path, layer_indices, n_positions=n_positions,
            reasoning_boundaries=reasoning_boundaries,
        )

    if metadata.empty:
        return pd.DataFrame(columns=empty_cols)

    # Assign difficulties to filter valid traces
    metadata_prep, valid_idx_prep = _prepare_probing_data(
        states, metadata, difficulties,
        problem_ids=problem_ids, n_folds=n_folds,
    )
    if not valid_idx_prep:
        return pd.DataFrame(columns=empty_cols)

    # Average hidden states across runs per problem
    states_valid = {k: v[valid_idx_prep] for k, v in states.items()}
    metadata_valid = metadata_prep.loc[valid_idx_prep].reset_index(drop=True)
    states, metadata = average_states_by_problem(states_valid, metadata_valid)

    # metadata now has one row per problem with columns [problem_id, trace_length].
    # All problems have valid difficulty (filtered before averaging).
    if isinstance(difficulties, pd.Series):
        diff_lookup = difficulties.to_dict()
    else:
        diff_lookup = dict(zip(problem_ids, difficulties)) if problem_ids else {}
    metadata["difficulty"] = metadata["problem_id"].map(diff_lookup)
    valid_indices = metadata.index.tolist()

    if len(valid_indices) < n_folds * 2:
        return pd.DataFrame(columns=empty_cols)

    # After averaging, each row is a unique problem — regular KFold
    y = metadata["difficulty"].values
    trace_lengths_for_workers = metadata["trace_length"].values.astype(np.float64)
    n_traces = len(valid_indices)
    n_problems = n_traces
    print(f"  {n_problems} problems (runs averaged, length-residualized)")

    # Build cell list — just (layer, pos) tuples; X comes from shared memory
    cells = [
        (layer_idx, pos_idx)
        for layer_idx in layer_indices
        for pos_idx in range(n_positions)
    ]

    n_cells = len(cells)
    work_label = f"{n_cells} cells"
    if n_permutations > 0:
        work_label += f" x (1 + {n_permutations} permutations)"
    print(f"  Probing {work_label}...")

    use_parallel = n_workers > 1 and n_cells > 1
    results = []

    if use_parallel:
        import os
        import multiprocessing
        from src.parallel_workers import _worker_init

        # Explicitly use "fork" context so child processes inherit the
        # parent's globals (CoW shared memory).  The system default may
        # be "forkserver" or "spawn", which start fresh processes that
        # re-import the module and see _shared_states = None.
        _mp_ctx = multiprocessing.get_context("fork")

        global _shared_states, _shared_valid_indices, _shared_y
        global _shared_trace_lengths, _shared_groups

        # Workers read X from these globals via copy-on-write (no pickling)
        _shared_states = states
        _shared_valid_indices = valid_indices
        _shared_y = y
        _shared_trace_lengths = trace_lengths_for_workers
        _shared_groups = None  # build_probing_heatmap uses regular KFold

        work_args = [
            (layer_idx, pos_idx, n_positions,
             n_folds, alpha, seed, n_permutations)
            for layer_idx, pos_idx in cells
        ]

        # Diagnostic output
        sample_key = next(iter(states))
        x_size_mb = states[sample_key].nbytes / 1e6
        print(f"  Full state array: {x_size_mb:.1f} MB per cell "
              f"({n_traces} traces), shared via CoW")
        print(f"  Workers: {n_workers}, "
              f"estimated ~{n_cells // max(n_workers, 1)} batches")

        # Set BLAS thread limits BEFORE forking so workers inherit them.
        # _worker_init sets these too, but BLAS reads env vars at import
        # time, so setting them after fork (in the child) is too late.
        _blas_vars = [
            "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]
        _saved_env = {v: os.environ.get(v) for v in _blas_vars}
        for v in _blas_vars:
            os.environ[v] = "1"

        try:
            with _mp_ctx.Pool(n_workers, initializer=_worker_init) as pool:
                iterator = pool.imap_unordered(_probe_cell_worker, work_args)
                if tqdm is not None:
                    iterator = tqdm(iterator, total=n_cells,
                                    desc="  Probing cells", unit="cell")
                for result in iterator:
                    if result is not None:
                        results.append(result)
        except (RuntimeError, OSError, BrokenPipeError) as e:
            print(f"\n  ERROR in parallel probing: {type(e).__name__}: {e}")
            print(f"  Falling back to serial execution...")
            results = []
            use_parallel = False
        finally:
            # Clean up shared globals
            _shared_states = None
            _shared_valid_indices = None
            _shared_y = None
            _shared_trace_lengths = None
            _shared_groups = None
            # Restore parent's original env vars
            for v in _blas_vars:
                if _saved_env[v] is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = _saved_env[v]

    if not use_parallel:
        # Serial with progress bar
        iterator = cells
        if tqdm is not None:
            iterator = tqdm(iterator, desc="  Probing cells", unit="cell")
        for layer_idx, pos_idx in iterator:
            X = states[(layer_idx, pos_idx)][valid_indices]
            result = _probe_single_cell(
                X, y, layer_idx, pos_idx, n_positions,
                n_folds, alpha, seed, n_permutations, groups=None,
                trace_lengths=trace_lengths_for_workers,
            )
            if result is not None:
                if tqdm is not None:
                    iterator.set_postfix(
                        layer=layer_idx, pos=pos_idx,
                        R2=f"{result['r2_mean']:.3f}",
                    )
                results.append(result)

    # Drop internal _model key before building DataFrame
    for r in results:
        r.pop("_model", None)

    return pd.DataFrame(results)


def build_probing_heatmap_by_correctness(
    hdf5_path: str | Path,
    difficulties: np.ndarray | pd.Series,
    correctness: pd.DataFrame,
    layer_indices: List[int],
    n_positions: int = 10,
    n_folds: int = 5,
    alpha: float = None,
    seed: int = 42,
    n_workers: int = 1,
    _states: Optional[Dict] = None,
    _metadata: Optional[pd.DataFrame] = None,
    reasoning_boundaries: Optional[Dict[Tuple[str, int], float]] = None,
) -> Dict[str, pd.DataFrame]:
    """Build separate probing heatmaps for correct vs incorrect traces.

    Unlike ``build_probing_heatmap()`` which averages across runs and
    residualizes trace length, this function operates on **individual
    traces** with GroupKFold to enable the correct/incorrect split.

    .. note:: Kept for backward compatibility with ``run_analysis.py``.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    if isinstance(difficulties, pd.Series):
        diff_lookup = difficulties.to_dict()
    else:
        raise ValueError("difficulties must be a pd.Series for correctness split")

    empty = pd.DataFrame(columns=[
        "layer", "position_idx", "position_frac",
        "r2_mean", "r2_std", "mse_mean", "n_traces",
    ])

    if _states is not None and _metadata is not None:
        states, metadata = _states, _metadata.copy()
    else:
        print("  Extracting hidden states from HDF5...")
        states, metadata = extract_generation_stage_states(
            hdf5_path, layer_indices, n_positions=n_positions,
            reasoning_boundaries=reasoning_boundaries,
        )

    if metadata.empty:
        return {"correct": empty, "incorrect": empty}

    metadata["difficulty"] = metadata["problem_id"].map(diff_lookup)
    correctness = correctness.copy()
    correctness["problem_id"] = correctness["problem_id"].astype(str)
    correctness["run_idx"] = correctness["run_idx"].astype(int)
    metadata["problem_id"] = metadata["problem_id"].astype(str)

    metadata = metadata.merge(
        correctness[["problem_id", "run_idx", "correct"]],
        on=["problem_id", "run_idx"],
        how="left",
    )
    metadata = metadata.dropna(subset=["difficulty", "correct"])

    heatmaps = {}
    for label, mask in [("correct", metadata["correct"] == 1),
                         ("incorrect", metadata["correct"] == 0)]:
        sub_indices = metadata.index[mask].tolist()
        if len(sub_indices) < n_folds * 2:
            heatmaps[label] = empty.copy()
            continue

        y = metadata.loc[sub_indices, "difficulty"].values
        groups = metadata.loc[sub_indices, "problem_id"].values
        n_traces_stratum = len(sub_indices)
        n_problems = len(np.unique(groups))

        if n_problems < n_folds:
            heatmaps[label] = empty.copy()
            continue

        print(f"  [{label}] {n_traces_stratum} traces, "
              f"{n_problems} unique problems")

        results = []
        cells = [
            (layer_idx, pos_idx)
            for layer_idx in layer_indices
            for pos_idx in range(n_positions)
        ]
        n_cells = len(cells)
        use_parallel = n_workers > 1 and n_cells > 1

        if use_parallel:
            import os
            import multiprocessing
            from src.parallel_workers import _worker_init

            _mp_ctx = multiprocessing.get_context("fork")

            global _shared_states, _shared_valid_indices, _shared_y
            global _shared_trace_lengths, _shared_groups

            _shared_states = states
            _shared_valid_indices = sub_indices
            _shared_y = y
            _shared_groups = groups
            _shared_trace_lengths = None  # no length residualization

            work_args = [
                (layer_idx, pos_idx, n_positions,
                 n_folds, alpha, seed, 0)
                for layer_idx, pos_idx in cells
            ]

            _blas_vars = [
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
            _saved_env = {v: os.environ.get(v) for v in _blas_vars}
            for v in _blas_vars:
                os.environ[v] = "1"

            try:
                with _mp_ctx.Pool(n_workers, initializer=_worker_init) as pool:
                    iterator = pool.imap_unordered(
                        _probe_cell_worker, work_args)
                    if tqdm is not None:
                        iterator = tqdm(
                            iterator, total=n_cells,
                            desc=f"  {label} probing", unit="cell")
                    for result in iterator:
                        if result is not None:
                            results.append(result)
            except (RuntimeError, OSError, BrokenPipeError) as e:
                print(f"\n  ERROR in parallel probing: "
                      f"{type(e).__name__}: {e}")
                print(f"  Falling back to serial execution...")
                results = []
                use_parallel = False
            finally:
                _shared_states = None
                _shared_valid_indices = None
                _shared_y = None
                _shared_trace_lengths = None
                _shared_groups = None
                for v in _blas_vars:
                    if _saved_env[v] is None:
                        os.environ.pop(v, None)
                    else:
                        os.environ[v] = _saved_env[v]

        if not use_parallel:
            cell_iter = cells
            if tqdm is not None:
                cell_iter = tqdm(cell_iter, desc=f"  {label} probing",
                                 unit="cell")

            for layer_idx, pos_idx in cell_iter:
                X = states[(layer_idx, pos_idx)][sub_indices]
                result = _probe_single_cell(
                    X, y, layer_idx, pos_idx, n_positions,
                    n_folds, alpha, seed, 0, groups=groups,
                )
                if result is not None:
                    results.append(result)

        for r in results:
            r.pop("_model", None)
        heatmaps[label] = pd.DataFrame(results)

    return heatmaps


# ---------------------------------------------------------------------------
# Difficulty direction extraction
# ---------------------------------------------------------------------------

def _extract_difficulty_direction(
    fitted_pipeline,
) -> np.ndarray:
    """Extract the normalized difficulty direction from a fitted Ridge probe.

    Given a fitted sklearn ``Pipeline`` (``StandardScaler`` -> ``Ridge``),
    extracts the Ridge weight vector and transforms it back to the original
    feature space, then normalizes to a unit vector.

    The resulting direction vector can be used for **activation steering**:
    adding ``alpha * direction`` to hidden states shifts the representation
    along the difficulty axis.

    Parameters
    ----------
    fitted_pipeline : sklearn Pipeline
        A fitted pipeline with steps ``("scaler", StandardScaler())``
        and ``("reg", Ridge())``.

    Returns
    -------
    direction : ndarray of shape (hidden_dim,)
        Unit vector in the difficulty direction (original feature space).
    """
    scaler = fitted_pipeline.named_steps["scaler"]
    ridge = fitted_pipeline.named_steps["reg"]

    # Ridge weights are in the scaled space: y_pred = X_scaled @ w + b
    # To get direction in original space: w_orig = w / scale
    w_scaled = ridge.coef_  # (hidden_dim,)
    scale = scaler.scale_   # (hidden_dim,)

    w_original = w_scaled / scale

    # Normalize to unit vector
    norm = np.linalg.norm(w_original)
    if norm < 1e-12:
        import warnings
        warnings.warn("Difficulty direction has near-zero norm. "
                      "The probe may not have learned a meaningful direction.")
        return w_original

    return w_original / norm


def extract_peak_direction(
    heatmap_df: pd.DataFrame,
    hdf5_path: str | Path,
    difficulties: np.ndarray | pd.Series,
    n_folds: int = 5,
    alpha: float = None,
    seed: int = 42,
    _states: Optional[Dict] = None,
    _metadata: Optional[pd.DataFrame] = None,
    reasoning_boundaries: Optional[Dict[Tuple[str, int], float]] = None,
) -> Dict:
    """Extract difficulty direction from the peak-R^2 cell in the heatmap.

    Identifies the (layer, position) with highest R^2, averages hidden
    states across runs per problem, residualizes trace length, refits
    Ridge on the residualized data, and extracts the normalized weight
    vector.

    Parameters
    ----------
    heatmap_df : DataFrame
        Output from ``build_probing_heatmap()``.
    hdf5_path : str or Path
        Path to HDF5 activations.
    difficulties : array-like
        Difficulty values indexed by problem_id.
    n_folds : int
        CV folds for the refit.
    alpha : float
        Ridge regularization.
    seed : int
        Random seed.
    _states : dict, optional
        Pre-extracted states (skip HDF5 read). For internal reuse.
    _metadata : DataFrame, optional
        Pre-extracted metadata (skip HDF5 read). For internal reuse.

    Returns
    -------
    dict with keys: direction (unit vector), peak_layer, peak_position_idx,
    peak_r2, fitted_pipeline.
    """
    from src.probing import train_regression_probe
    from sklearn.linear_model import LinearRegression

    if heatmap_df.empty:
        raise ValueError("Cannot extract direction from empty heatmap.")

    # Find peak R^2 cell
    peak_row = heatmap_df.loc[heatmap_df["r2_mean"].idxmax()]
    peak_layer = int(peak_row["layer"])
    peak_pos = int(peak_row["position_idx"])
    n_positions = int(heatmap_df["position_idx"].max()) + 1

    # Extract or reuse states
    if _states is not None and _metadata is not None:
        states, metadata = _states, _metadata.copy()
    else:
        print("  Extracting hidden states for peak direction...")
        layer_indices = [peak_layer]
        states, metadata = extract_generation_stage_states(
            hdf5_path, layer_indices, n_positions=n_positions,
            reasoning_boundaries=reasoning_boundaries,
        )

    if isinstance(difficulties, pd.Series):
        diff_lookup = difficulties.to_dict()
    else:
        raise ValueError("difficulties must be a pd.Series")

    metadata["difficulty"] = metadata["problem_id"].map(diff_lookup)
    n_before = len(metadata)
    metadata = metadata.dropna(subset=["difficulty"])
    n_dropped = n_before - len(metadata)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped}/{n_before} traces with missing difficulty")

    valid_indices = metadata.index.tolist()

    # Average across runs per problem
    states_valid = {k: v[valid_indices] for k, v in states.items()}
    metadata_valid = metadata.loc[valid_indices].reset_index(drop=True)
    states_avg, metadata_avg = average_states_by_problem(states_valid, metadata_valid)

    # Re-map difficulty after averaging (all problems already had valid
    # difficulty before averaging, so no rows should be dropped here)
    metadata_avg["difficulty"] = metadata_avg["problem_id"].map(diff_lookup)
    assert metadata_avg["difficulty"].notna().all(), (
        "Unexpected NaN difficulties after averaging — all problems were "
        "filtered before averaging"
    )
    avg_indices = metadata_avg.index.tolist()

    key = (peak_layer, peak_pos)
    X = states_avg[key][avg_indices]
    y = metadata_avg["difficulty"].values

    # Residualize trace length from X and y
    trace_lengths = metadata_avg.loc[avg_indices, "trace_length"].values.astype(
        np.float64
    )
    lr = LinearRegression()
    L = trace_lengths.reshape(-1, 1)
    lr.fit(L, X); X = X - lr.predict(L)
    lr.fit(L, y.reshape(-1, 1)); y = y - lr.predict(L).ravel()

    # Fit probe on averaged, length-residualized data
    probe_result = train_regression_probe(
        X, y, n_folds=n_folds, alpha=alpha, seed=seed,
    )

    direction = _extract_difficulty_direction(probe_result["model"])

    # Compute projection statistics for sigma-calibrated steering
    projections = X @ direction
    projection_mean = float(projections.mean())
    projection_std = float(projections.std())

    return {
        "direction": direction,
        "peak_layer": peak_layer,
        "peak_position_idx": peak_pos,
        "peak_position_frac": peak_pos / max(1, n_positions - 1),
        "peak_r2": float(peak_row["r2_mean"]),
        "fitted_pipeline": probe_result["model"],
        "projection_mean": projection_mean,
        "projection_std": projection_std,
    }


# ---------------------------------------------------------------------------
# Prompt-stage probing (requires GPU + model)
# ---------------------------------------------------------------------------

def extract_prompt_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    layer_indices: List[int],
    batch_size: int = 1,
) -> Dict[int, np.ndarray]:
    """Extract hidden states at the last prompt token for each layer.

    Requires a **GPU** and the model loaded in memory.  Uses
    ``extract_hidden_states_single_forward()`` from ``src.extraction`` for
    each prompt.

    Parameters
    ----------
    model : nnsight.LanguageModel
        The loaded model.
    tokenizer : PreTrainedTokenizer
        The model's tokenizer.
    prompts : list of str
        One formatted prompt per problem.
    layer_indices : list of int
        Which layers to extract.
    batch_size : int
        Number of prompts to process at once (default 1).

    Returns
    -------
    dict[layer_idx -> ndarray(n_prompts, hidden_dim)]
        Hidden-state arrays, one vector per prompt per layer.
    """
    import torch
    from src.extraction import extract_hidden_states_single_forward

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    layer_states: Dict[int, list] = {idx: [] for idx in layer_indices}

    prompt_iter = range(0, len(prompts), batch_size)
    if tqdm is not None:
        prompt_iter = tqdm(prompt_iter, desc="  Extracting prompt states", unit="prompt")

    # Disabling autograd is required for 8-bit (bitsandbytes) forwards: without
    # it, every MatMul8bitLt.apply call retains activations for backward and
    # the loop OOMs on 32B models even on an 80 GB H100.
    with torch.no_grad():
        for i in prompt_iter:
            batch = prompts[i:i + batch_size]
            for prompt in batch:
                states = extract_hidden_states_single_forward(
                    model, prompt, layer_indices,
                )
                for idx in layer_indices:
                    h = states[idx][-1]
                    layer_states[idx].append(h.float().numpy())

    return {idx: np.stack(vecs, axis=0) for idx, vecs in layer_states.items()}


def probe_prompt_stage(
    layer_states: Dict[int, np.ndarray],
    difficulties: np.ndarray,
    n_folds: int = 5,
    alpha: float = None,
    seed: int = 42,
) -> Dict[int, Dict]:
    """Run Ridge probes on prompt-stage hidden states per layer.

    Parameters
    ----------
    layer_states : dict[layer_idx -> ndarray(n_prompts, hidden_dim)]
        From ``extract_prompt_hidden_states()``.
    difficulties : ndarray of shape (n_prompts,)
        Difficulty values (same order as prompts).
    n_folds : int
        CV folds.
    alpha : float
        Ridge regularization.
    seed : int
        Random seed.

    Returns
    -------
    dict[layer_idx -> probe_result_dict]
        Each probe_result has keys: r2_mean, r2_std, mse_mean, model,
        spearman_rho, spearman_p.
    """
    from src.probing import train_regression_probe
    from scipy.stats import spearmanr

    results = {}
    for layer_idx, X in layer_states.items():
        if len(X) < n_folds * 2:
            continue

        probe = train_regression_probe(
            X, difficulties,
            n_folds=n_folds, alpha=alpha, seed=seed,
        )

        rho, rho_p = spearmanr(difficulties, probe["oof_predictions"])

        results[layer_idx] = {
            "r2_mean": probe["r2_mean"],
            "r2_std": probe["r2_std"],
            "mse_mean": probe["mse_mean"],
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "model": probe["model"],
        }

    return results
