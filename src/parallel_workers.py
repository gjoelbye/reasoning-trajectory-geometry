"""Worker functions for parallel computation.

These functions are extracted from notebook cells to keep the notebooks clean
and allow reuse.  Compatible with both ThreadPoolExecutor and ProcessPoolExecutor.

IMPORTANT: All worker functions in this module MUST remain top-level module
functions (not closures, lambdas, or nested functions) so they can be pickled
by ProcessPoolExecutor.  Each worker must be fully self-contained — all data
comes through its arguments, with no references to outer-scope variables.

Workers that need extra imports use *deferred* imports inside the function
body so that the module-level import list stays light and pickling always works.
"""

import torch

from src.trajectories import (
    random_walk_baseline,
    trajectory_summary,
    trajectory_traces,
)


# ---------------------------------------------------------------------------
# Worker process initializer
# ---------------------------------------------------------------------------


def _worker_init():
    """Suppress internal BLAS/MKL threading in worker processes.

    When running N worker processes, each inherits multi-threaded BLAS.
    N processes × M internal threads = contention for the same CPU cores.
    Setting threads=1 is optimal when the pool already saturates all cores.
    """
    import os
    import warnings

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    torch.set_num_threads(1)

    # Suppress SymPy's harmless antlr4 warning that fires on import
    # in every worker process when antlr4 is not installed.
    warnings.filterwarnings("ignore", message="antlr4.*not installed")


def _worker_init_blas2():
    """Allow 2 BLAS threads per worker for compute-heavy baselines.

    When there are fewer tasks than workers (e.g. random-walk baselines),
    allowing 2 BLAS threads per worker better utilizes available cores.
    """
    import os
    import warnings

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "2"

    torch.set_num_threads(2)

    warnings.filterwarnings("ignore", message="antlr4.*not installed")


# ---------------------------------------------------------------------------
# Stage-03 workers: trajectory metrics + random-walk baselines
# ---------------------------------------------------------------------------


def compute_metrics_worker(item):
    """Compute all trajectory metrics for a single pre-loaded trace.

    This function must remain a top-level module function (not a closure or
    lambda) so it can be pickled by ProcessPoolExecutor. It must not reference
    any outer-scope variables — all data comes through ``item``.

    Parameters
    ----------
    item : tuple of (numpy.ndarray, dict)
        - states_np: float32 array of shape (T, D), the hidden-state sequence
        - meta: dict with keys 'problem_id', 'run_idx', 'layer', 'rating'

    Returns
    -------
    dict
        Flat dictionary with all trajectory summary metrics plus metadata.
    """
    states_np, meta = item
    states = torch.tensor(states_np)

    summary = trajectory_summary(states)
    summary.update(meta)
    summary["n_tokens"] = states.shape[0]

    return summary


def compute_metrics_and_traces_worker(item):
    """Compute trajectory metrics AND per-step traces for a single trace.

    Calls both ``trajectory_summary()`` and ``trajectory_traces()`` in one
    pass, returning a ``(scalar_dict, traces_dict)`` tuple.

    This function must remain a top-level module function (not a closure or
    lambda) so it can be pickled by ProcessPoolExecutor.

    Parameters
    ----------
    item : tuple of (numpy.ndarray, dict)
        - states_np: float32 array of shape (T, D), the hidden-state sequence
        - meta: dict with keys 'problem_id', 'run_idx', 'layer', 'rating'

    Returns
    -------
    tuple of (dict, dict)
        - scalar_dict: flat dictionary with trajectory summary metrics + metadata
        - traces_dict: per-step traces (velocity, curvature, cosine_turn)
    """
    states_np, meta = item
    states = torch.tensor(states_np)

    summary = trajectory_summary(states)
    summary.update(meta)
    summary["n_tokens"] = states.shape[0]

    traces = trajectory_traces(states)
    traces_out = {
        "problem_id": meta["problem_id"],
        "run_idx": meta["run_idx"],
        "layer": meta["layer"],
        "n_steps": states.shape[0],
        "velocity": traces["velocity"],
        "curvature": traces["curvature"],
        "cosine_turn": traces["cosine_turn"],
    }

    return summary, traces_out


def compute_baseline_worker(args):
    """Compute random-walk baseline for a single trajectory length.

    This function must remain a top-level module function (not a closure or
    lambda) so it can be pickled by ProcessPoolExecutor.

    Parameters
    ----------
    args : tuple of (int, int)
        - length: number of steps in the trajectory
        - hidden_dim: dimensionality of the hidden states

    Returns
    -------
    tuple of (int, dict)
        The length and the baseline statistics dictionary.
    """
    length, hidden_dim = args
    baseline = random_walk_baseline(
        n_steps=length, dim=hidden_dim, n_simulations=100, seed=42
    )
    return length, baseline



# ---------------------------------------------------------------------------
# Stage-02 workers: structural patterns + correctness evaluation
# ---------------------------------------------------------------------------


def compute_patterns_worker(think_text):
    """Compute structural patterns and repetition metrics for one think block.

    Top-level function for ProcessPoolExecutor.  Uses deferred imports.

    Parameters
    ----------
    think_text : str
        The think-block text from a CoT trace.

    Returns
    -------
    dict
        Merged output of ``detect_all_patterns_normalized()`` and
        ``compute_repetition_score()``.
    """
    from src.cot_patterns import detect_all_patterns_normalized, compute_repetition_score

    result = detect_all_patterns_normalized(think_text)
    rep = compute_repetition_score(think_text)
    result.update(rep)
    return result


def evaluate_trace_worker(args):
    """Evaluate a single trace's code against test cases.

    Top-level function for ProcessPoolExecutor.  Uses deferred imports.

    Parameters
    ----------
    args : tuple
        ``(problem_id, run_idx, code, answer_text, trace_text,
        test_cases, test_source)``

        *code* may be ``None``; extraction happens here as fallback.
        *test_cases* must be a plain ``list[dict]`` (no numpy arrays).

    Returns
    -------
    dict
        Keys: ``problem_id, run_idx, passed, total, correct,
        error_type, test_source, code``.
    """
    from src.evaluation import extract_python_code, evaluate_on_tests

    problem_id, run_idx, code, answer_text, trace_text, test_cases, test_source = args

    # Extract code if not provided
    if code is None:
        code = extract_python_code(answer_text) if answer_text else None
        if code is None and trace_text:
            code = extract_python_code(trace_text)

    if not isinstance(code, str) or test_source == "none":
        error = "no_code" if not isinstance(code, str) else "no_tests"
        return {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "passed": 0,
            "total": 0,
            "correct": False,
            "error_type": error,
            "test_source": test_source,
            "code": code,
        }

    passed, total, error_type = evaluate_on_tests(code, test_cases)
    return {
        "problem_id": problem_id,
        "run_idx": run_idx,
        "passed": passed,
        "total": total,
        "correct": passed == total and total > 0,
        "error_type": error_type,
        "test_source": test_source,
        "code": code,
    }


# ---------------------------------------------------------------------------
# Stage-03 worker: probing experiments
# ---------------------------------------------------------------------------


def compute_probe_worker(args):
    """Run probing for a single (layer, feature-set) combination.

    Top-level function for ProcessPoolExecutor.  Uses deferred imports.

    Parameters
    ----------
    args : tuple
        ``(layer, feat_name, X, y_rating, y_correct,
        majority_baseline, n_permutations, groups)``

        ``groups`` is an array of group labels for GroupKFold
        (repeated-measures data) or ``None`` for standard KFold.

    Returns
    -------
    dict
        Probing results for this ``(layer, feature_set)`` combination.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, GroupKFold

    from src.probing import compute_roc_auc, train_regression_probe

    layer, feat_name, X, y_rating, y_correct, majority_baseline, n_permutations, groups = args

    # Ridge regression for rating prediction (with per-fold scaling)
    probe_result = train_regression_probe(X, y_rating, n_folds=5, seed=42, groups=groups, r2_only=True)
    r2_mean = probe_result["r2_mean"]

    # Permutation baseline for R²
    rng = np.random.RandomState(42)
    perm_r2s = []

    # Pre-compute group-level permutation mapping if needed.
    # Each group (problem) has the same difficulty across all its runs,
    # so we permute which difficulty each group gets rather than shuffling
    # individual observations (which would break group structure).
    if groups is not None:
        unique_groups = np.unique(groups)
        # Get one difficulty value per group (all runs share the same value)
        group_difficulties = np.array([y_rating[groups == g][0] for g in unique_groups])

    for _ in range(n_permutations):
        if groups is not None:
            perm_diff = rng.permutation(group_difficulties)
            group_to_perm = dict(zip(unique_groups, perm_diff))
            y_perm = np.array([group_to_perm[g] for g in groups])
        else:
            y_perm = rng.permutation(y_rating)
        perm_result = train_regression_probe(X, y_perm, n_folds=5, seed=42, groups=groups, r2_only=True)
        perm_r2s.append(perm_result["r2_mean"])
    perm_r2_mean = float(np.mean(perm_r2s))
    perm_r2_std = float(np.std(perm_r2s))
    r2_pvalue = float(np.mean([pr >= r2_mean for pr in perm_r2s]))

    # Logistic regression for correctness
    acc_mean = float("nan")
    auc_mean = float("nan")
    if len(np.unique(y_correct)) > 1:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        if groups is not None:
            cv = GroupKFold(n_splits=5)
        else:
            cv = 5
        acc_scores = cross_val_score(lr, X, y_correct, cv=cv, groups=groups, scoring="accuracy")
        acc_mean = float(acc_scores.mean())

        auc_result = compute_roc_auc(X, y_correct, n_folds=5, groups=groups)
        auc_mean = float(auc_result["roc_auc_mean"])

    return {
        "layer": layer,
        "features": feat_name,
        "r2": r2_mean,
        "r2_perm_mean": perm_r2_mean,
        "r2_perm_std": perm_r2_std,
        "r2_pvalue": r2_pvalue,
        "accuracy": acc_mean,
        "majority_baseline": majority_baseline,
        "roc_auc": auc_mean,
    }


# ---------------------------------------------------------------------------
# Stage-02 worker: MATH answer evaluation
# ---------------------------------------------------------------------------


def evaluate_math_trace_worker(args):
    r"""Evaluate a single MATH trace's answer against ground truth.

    Top-level function for ProcessPoolExecutor.  Parallels
    ``evaluate_trace_worker`` but uses ``\boxed{}`` answer extraction
    and symbolic/numerical comparison instead of code execution.

    Parameters
    ----------
    args : tuple
        ``(problem_id, run_idx, answer_text, trace_text,
        ground_truth, math_level)``

    Returns
    -------
    dict
        Keys: ``problem_id, run_idx, correct, predicted_answer,
        ground_truth, math_level, error_type``.
    """
    from src.evaluation import extract_math_answer, evaluate_math_answer

    problem_id, run_idx, answer_text, trace_text, ground_truth, math_level = args

    # Try to extract from answer portion first, then full trace
    predicted = None
    if answer_text:
        predicted = extract_math_answer(answer_text)
    if predicted is None and trace_text:
        predicted = extract_math_answer(trace_text)

    if predicted is None:
        return {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "correct": False,
            "predicted_answer": None,
            "ground_truth": ground_truth,
            "math_level": math_level,
            "error_type": "no_answer",
        }

    correct = evaluate_math_answer(predicted, ground_truth)
    return {
        "problem_id": problem_id,
        "run_idx": run_idx,
        "correct": correct,
        "predicted_answer": predicted,
        "ground_truth": ground_truth,
        "math_level": math_level,
        "error_type": "success" if correct else "wrong_answer",
    }


# ---------------------------------------------------------------------------
# Stage-02 worker: SAT answer evaluation
# ---------------------------------------------------------------------------


def evaluate_sat_trace_worker(args):
    """Evaluate a single SAT trace's answer against ground truth.

    Top-level function for ProcessPoolExecutor.  Parallels
    ``evaluate_math_trace_worker`` but uses SAT/UNSAT label extraction
    instead of symbolic answer comparison.

    Parameters
    ----------
    args : tuple
        ``(problem_id, run_idx, answer_text, trace_text,
        ground_truth_satisfiable, num_clauses)``

    Returns
    -------
    dict
        Keys: ``problem_id, run_idx, correct, predicted_sat_label,
        ground_truth_satisfiable, num_clauses, error_type``.
    """
    from src.evaluation import extract_sat_answer, evaluate_sat_answer

    problem_id, run_idx, answer_text, trace_text, ground_truth_satisfiable, num_clauses = args

    # Try to extract from answer portion first, then full trace
    predicted = None
    if answer_text:
        predicted = extract_sat_answer(answer_text)
    if predicted is None and trace_text:
        predicted = extract_sat_answer(trace_text)

    if predicted is None:
        return {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "correct": False,
            "predicted_sat_label": None,
            "ground_truth_satisfiable": ground_truth_satisfiable,
            "num_clauses": num_clauses,
            "error_type": "no_answer",
        }

    correct = evaluate_sat_answer(predicted, ground_truth_satisfiable)
    return {
        "problem_id": problem_id,
        "run_idx": run_idx,
        "correct": correct,
        "predicted_sat_label": predicted,
        "ground_truth_satisfiable": ground_truth_satisfiable,
        "num_clauses": num_clauses,
        "error_type": "success" if correct else "wrong_answer",
    }
