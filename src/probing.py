"""Linear and MLP probing classifiers for hidden-state analysis.

Probes test whether a given property (correctness, difficulty, stuck state)
is linearly decodable from a model's hidden representations. This module
wraps scikit-learn pipelines with cross-validation to produce reliable
accuracy estimates.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Classification probes
# ---------------------------------------------------------------------------

def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,
    max_iter: int = 2000,
    seed: int = 42,
) -> Dict:
    """Train a logistic-regression probe with stratified cross-validation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, hidden_dim)
    y : ndarray of shape (n_samples,), binary labels
    n_folds : number of CV folds
    C : inverse regularization strength
    max_iter : solver iteration limit
    seed : random seed for reproducibility

    Returns
    -------
    dict with keys: accuracy_mean, accuracy_std, cv_scores, model
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs",
                                   random_state=seed)),
    ])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

    # Fit on full data for the returned model
    pipe.fit(X, y)

    return {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "cv_scores": scores.tolist(),
        "model": pipe,
    }


def compute_roc_auc(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> Dict:
    """Compute ROC-AUC for a logistic-regression probe via cross-validation."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, solver="lbfgs",
                                   random_state=seed)),
    ])

    if groups is not None:
        cv = GroupKFold(n_splits=n_folds)
        scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring="roc_auc")
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

    return {
        "roc_auc_mean": float(scores.mean()),
        "roc_auc_std": float(scores.std()),
        "cv_scores": scores.tolist(),
    }


# ---------------------------------------------------------------------------
# Regression probes
# ---------------------------------------------------------------------------

_DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def train_regression_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    alpha: float = None,
    alphas: tuple = None,
    seed: int = 42,
    r2_only: bool = False,
    groups: Optional[np.ndarray] = None,
) -> Dict:
    """Train a Ridge regression probe with cross-validation.

    Useful for predicting continuous difficulty from hidden states.

    By default uses ``RidgeCV`` with leave-one-out cross-validation to
    automatically select the best regularization strength from a
    logarithmic grid.  This is standard practice in the probing
    literature (Hewitt & Liang 2019) and is analytically computed so
    adds negligible overhead.

    Parameters
    ----------
    X : ndarray of shape (n_samples, hidden_dim)
    y : ndarray of shape (n_samples,), continuous targets
    alpha : float, optional
        If provided, use a fixed Ridge alpha (no internal CV for alpha).
    alphas : tuple of float, optional
        Candidate alphas for RidgeCV.  Default: (0.01, 0.1, ..., 10000).
        Ignored if ``alpha`` is set.
    r2_only : bool
        If True, skip MSE scoring and final fit (faster for permutation
        tests where only R² is needed).
    groups : ndarray, optional
        Group labels for GroupKFold (e.g. problem_ids).  When provided,
        all samples in the same group are kept together in the same fold,
        preventing information leakage from repeated measures.

    Returns
    -------
    dict with keys: r2_mean, r2_std, mse_mean, model, and optionally
    alpha_selected (the alpha chosen by RidgeCV).
    """
    from sklearn.metrics import r2_score, mean_squared_error

    if alpha is not None:
        # Fixed alpha — backward-compatible path
        reg = Ridge(alpha=alpha)
    else:
        # RidgeCV with efficient LOO for alpha selection
        if alphas is None:
            alphas = _DEFAULT_ALPHAS
        reg = RidgeCV(alphas=alphas)

    def _make_pipe():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", reg.__class__(**reg.get_params())),
        ])

    pipe = _make_pipe()

    if groups is not None:
        cv = GroupKFold(n_splits=n_folds)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    if r2_only:
        r2_scores = cross_val_score(
            pipe, X, y, cv=cv, groups=groups, scoring="r2",
        )
        return {
            "r2_mean": float(r2_scores.mean()),
            "r2_std": float(r2_scores.std()),
            "mse_mean": float("nan"),
            "model": None,
        }

    # Single CV loop computing both R² and MSE from the same fold predictions,
    # plus out-of-fold predictions for honest Spearman correlation
    r2_scores = []
    mse_scores = []
    oof_predictions = np.empty(len(y), dtype=np.float64)
    for train_idx, test_idx in cv.split(X, y, groups):
        pipe_fold = _make_pipe()
        pipe_fold.fit(X[train_idx], y[train_idx])
        y_pred = pipe_fold.predict(X[test_idx])
        oof_predictions[test_idx] = y_pred
        r2_scores.append(r2_score(y[test_idx], y_pred))
        mse_scores.append(mean_squared_error(y[test_idx], y_pred))

    pipe.fit(X, y)

    result = {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "mse_mean": float(np.mean(mse_scores)),
        "model": pipe,
        "oof_predictions": oof_predictions,
    }

    # Report which alpha was selected (RidgeCV only)
    fitted_reg = pipe.named_steps["reg"]
    if hasattr(fitted_reg, "alpha_"):
        result["alpha_selected"] = float(fitted_reg.alpha_)

    return result


# ---------------------------------------------------------------------------
# Selectivity
# ---------------------------------------------------------------------------

def selectivity_index(
    probe_accuracy: float,
    control_accuracy: float,
) -> float:
    """Compute selectivity: improvement over a control baseline.

    selectivity = (probe_acc - control_acc) / (1 - control_acc)

    A value of 1.0 means the probe perfectly recovers information that the
    control task cannot. A value of 0.0 means the probe does no better
    than the control.
    """
    denom = 1.0 - control_accuracy
    if denom < 1e-8:
        return 0.0
    return (probe_accuracy - control_accuracy) / denom


# ---------------------------------------------------------------------------
# Surface-feature baselines
# ---------------------------------------------------------------------------

def compute_surface_features(
    problems_df: "pd.DataFrame",
    prompt_column: str = "formatted_prompt",
) -> np.ndarray:
    """Compute surface-level input features for baseline probing.

    These features capture shallow properties of the input (length,
    vocabulary complexity) that a hidden-state probe should ideally
    surpass.  If the probe achieves comparable R-squared to these
    features, it may be picking up surface statistics rather than
    genuine difficulty encoding.

    Features (per problem):
        0. Character length of the prompt
        1. Word count (whitespace-split)
        2. Unique-token ratio (unique words / total words)
        3. Number of numeric literals (digit-only tokens)
        4. Sentence count (period-terminated spans)

    Parameters
    ----------
    problems_df : DataFrame with a prompt text column
    prompt_column : column name containing the formatted prompt text

    Returns
    -------
    ndarray of shape (n_problems, 5)
    """
    import re

    import pandas as _pd

    features = []
    for text in problems_df[prompt_column]:
        if text is None or _pd.isna(text):
            features.append([0, 0, 0.0, 0, 0])
            continue
        text = str(text)
        words = text.split()
        n_words = len(words) if words else 1
        unique_words = len(set(words))
        n_numbers = sum(1 for w in words if re.fullmatch(r"\d+", w))
        n_sentences = max(1, text.count(".") + text.count("?") + text.count("!"))
        features.append([
            len(text),
            n_words,
            unique_words / n_words,
            n_numbers,
            n_sentences,
        ])
    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Binned difficulty probe (used by scripts/run_analysis.py)
# ---------------------------------------------------------------------------

def train_binned_difficulty_probe(
    X: np.ndarray,
    y_continuous: np.ndarray,
    n_bins: int = 5,
    n_folds: int = 5,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> Dict:
    """Discretise difficulty into bins and train a classification probe.

    Provides a robustness check alongside continuous Ridge regression.
    If difficulty is genuinely encoded, classification accuracy should
    exceed chance (1 / n_bins).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y_continuous : ndarray of shape (n_samples,), continuous difficulty
    n_bins : number of equal-frequency bins (default 5)
    n_folds : number of CV folds
    seed : random seed
    groups : ndarray, optional
        Group labels for GroupKFold (e.g. problem_ids).

    Returns
    -------
    dict with keys: accuracy_mean, accuracy_std, cv_scores,
        bin_edges, chance_level
    """
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    # Equal-frequency binning
    y_binned, bin_edges = pd.qcut(
        y_continuous, q=n_bins, labels=False, retbins=True, duplicates="drop"
    )
    y_binned = np.asarray(y_binned, dtype=int)

    actual_bins = len(np.unique(y_binned))
    if actual_bins < 2:
        return {
            "accuracy_mean": float("nan"),
            "accuracy_std": float("nan"),
            "cv_scores": [],
            "bin_edges": bin_edges.tolist(),
            "chance_level": float("nan"),
            "confusion_matrix": [],
        }
    if actual_bins < n_folds:
        n_folds = actual_bins

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs",
            random_state=seed,
        )),
    ])

    if groups is not None:
        cv = GroupKFold(n_splits=n_folds)
        scores = cross_val_score(pipe, X, y_binned, cv=cv, groups=groups, scoring="accuracy")
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(pipe, X, y_binned, cv=cv, scoring="accuracy")

    # Fit on full data for confusion matrix
    pipe.fit(X, y_binned)
    y_pred = pipe.predict(X)

    return {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "cv_scores": scores.tolist(),
        "bin_edges": bin_edges.tolist(),
        "chance_level": 1.0 / actual_bins,
        "confusion_matrix": confusion_matrix(y_binned, y_pred).tolist(),
    }


