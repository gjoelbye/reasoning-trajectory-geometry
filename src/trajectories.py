"""Geometric measures on hidden-state trajectories.

Given a sequence of hidden states h_0, h_1, ..., h_T (one per token or per
reasoning step), this module computes local and global geometric properties
of the trajectory through representation space.

Includes:
- Raw high-dimensional metrics (L2 velocity, curvature, directness).
- Intrinsic dimensionality estimates (TwoNN MLE, PCA-based) that capture
  how many effective dimensions the trajectory occupies.
- Random-walk null baselines for statistical calibration of observed metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import pandas as pd


def l2_velocity(hidden_states: torch.Tensor) -> np.ndarray:
    """L2 distance between consecutive hidden states.

    This is the discrete "speed" of the trajectory: how far the
    representation moves in a single step.

    Parameters
    ----------
    hidden_states : Tensor of shape (T, D)

    Returns
    -------
    ndarray of shape (T-1,)
    """
    diffs = (hidden_states[1:] - hidden_states[:-1]).float()
    return diffs.norm(dim=-1).numpy()


def curvature(hidden_states: torch.Tensor) -> np.ndarray:
    """Menger curvature for consecutive triplets (A, B, C).

    kappa(A,B,C) = 4 * area(ABC) / (|AB| * |BC| * |AC|)

    Scale-invariant curvature with units of 1/length. High curvature
    means the trajectory is turning sharply, which may indicate the model
    is reconsidering or backtracking. Returns one value per interior point.

    Parameters
    ----------
    hidden_states : Tensor of shape (T, D)

    Returns
    -------
    ndarray of shape (T-2,)
    """
    h = hidden_states.float()
    A, B, C = h[:-2], h[1:-1], h[2:]
    AB = B - A
    BC = C - B
    AC = C - A
    ab = AB.norm(dim=-1)
    bc = BC.norm(dim=-1)
    ac = AC.norm(dim=-1)
    # Area via cross product generalised to high-D:
    # |u x v| = sqrt(|u|^2 |v|^2 - (u.v)^2)
    ab_sq = (AB * AB).sum(dim=-1)
    ac_sq = (AC * AC).sum(dim=-1)
    dot_ab_ac = (AB * AC).sum(dim=-1)
    cross_sq = (ab_sq * ac_sq - dot_ab_ac ** 2).clamp(min=0.0)
    area = 0.5 * torch.sqrt(cross_sq)
    denom = (ab * bc * ac).clamp(min=1e-12)
    kappa = 4.0 * area / denom
    return kappa.numpy()


def net_displacement(hidden_states: torch.Tensor) -> float:
    """L2 distance from the first to the last hidden state."""
    return float((hidden_states[-1] - hidden_states[0]).float().norm().item())


def path_length(hidden_states: torch.Tensor) -> float:
    """Total L2 path length of the trajectory."""
    return float(l2_velocity(hidden_states).sum())


def directness(hidden_states: torch.Tensor) -> float:
    """Ratio of net displacement to total path length.

    A value of 1.0 means the trajectory is a straight line. Values near 0
    indicate a highly winding path. Useful for detecting "stuck" states
    where the model moves a lot but makes no net progress.
    """
    pl = path_length(hidden_states)
    if pl < 1e-12:
        return 0.0
    return net_displacement(hidden_states) / pl


# ---------------------------------------------------------------------------
# Intrinsic dimensionality estimates
# ---------------------------------------------------------------------------

def twonn_intrinsic_dimension(hidden_states: torch.Tensor) -> float:
    """Estimate intrinsic dimensionality via the TwoNN method (Facco et al. 2017).

    Computes the ratio mu = r2/r1 for each point (where r1, r2 are the
    distances to the first and second nearest neighbours), then estimates
    the intrinsic dimension as the MLE of the Pareto distribution that
    mu^(d) follows.

    Parameters
    ----------
    hidden_states : Tensor of shape (T, D)
        Sequence of hidden-state vectors. Requires T >= 3.

    Returns
    -------
    float
        Estimated intrinsic dimension. Returns 0.0 if T < 3.
    """
    if hasattr(hidden_states, "float"):
        X = hidden_states.float().cpu().numpy()
    else:
        X = np.asarray(hidden_states, dtype=np.float32)

    T = X.shape[0]
    if T < 3:
        return 0.0

    # Pairwise squared distances
    # Use (a-b)^2 = a^2 + b^2 - 2ab expansion for efficiency
    sq_norms = (X ** 2).sum(axis=1)
    dists_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T
    np.fill_diagonal(dists_sq, np.inf)
    dists_sq = np.clip(dists_sq, 0.0, None)

    # For each point, find the 2 smallest distances
    # np.partition is O(n) per row vs O(n log n) for full sort
    idx = np.argpartition(dists_sq, 2, axis=1)[:, :2]
    r = np.sqrt(np.take_along_axis(dists_sq, idx, axis=1))
    r.sort(axis=1)  # ensure r1 <= r2

    r1 = r[:, 0]
    r2 = r[:, 1]

    # Filter out degenerate points where r1 ~ 0
    valid = r1 > 1e-12
    if valid.sum() < 2:
        return 0.0

    mu = r2[valid] / r1[valid]

    # MLE: d = n / sum(log(mu_i))
    log_mu = np.log(mu)
    log_mu_sum = log_mu.sum()
    if log_mu_sum < 1e-12:
        return 0.0

    return float(len(mu) / log_mu_sum)


def pca_intrinsic_dimension(
    hidden_states: torch.Tensor,
    threshold: float = 0.9,
) -> float:
    """Estimate intrinsic dimensionality via PCA explained variance.

    Computes the number of principal components needed to explain at least
    ``threshold`` fraction of the total variance.

    Parameters
    ----------
    hidden_states : Tensor of shape (T, D)
        Sequence of hidden-state vectors. Requires T >= 3.
    threshold : float
        Fraction of variance to explain (default 0.9 = 90%).

    Returns
    -------
    float
        Number of components for the threshold. Returns 0.0 if T < 3.
    """
    if hasattr(hidden_states, "float"):
        X = hidden_states.float().cpu().numpy()
    else:
        X = np.asarray(hidden_states, dtype=np.float32)

    T = X.shape[0]
    if T < 3:
        return 0.0

    # Center
    X = X - X.mean(axis=0)

    # Gram matrix eigendecomposition: O(T^2*D) matmul + O(T^3) eigvalsh
    # Much faster than SVD when T << D (here T~50-1000, D=3584)
    gram = X @ X.T                                    # (T, T)
    eigvals = np.linalg.eigvalsh(gram)                # ascending order, O(T^3)
    # eigvals = s^2; reverse to descending, clamp negatives
    var = np.maximum(eigvals[::-1], 0.0)
    total_var = var.sum()
    if total_var < 1e-12:
        return 0.0

    cumvar = np.cumsum(var) / total_var
    n_components = int(np.searchsorted(cumvar, threshold) + 1)
    return float(n_components)


def trajectory_summary(hidden_states: torch.Tensor) -> Dict[str, float]:
    """Compute a full suite of summary statistics for a trajectory.

    Returns a flat dictionary suitable for DataFrame construction.

    Parameters
    ----------
    hidden_states : Tensor of shape (T, D)

    Returns
    -------
    dict with keys: curvature_mean, curvature_std, curvature_max,
        directness, num_steps, twonn_dim, pca_dim_90.
    """
    curv = curvature(hidden_states)
    n = len(hidden_states)

    pl = path_length(hidden_states)
    nd = net_displacement(hidden_states)

    return {
        "curvature_mean": float(curv.mean()) if len(curv) > 0 else 0.0,
        "curvature_std": float(curv.std()) if len(curv) > 0 else 0.0,
        "curvature_max": float(curv.max()) if len(curv) > 0 else 0.0,
        "directness": nd / pl if pl > 1e-12 else 0.0,
        "num_steps": n,
        "twonn_dim": twonn_intrinsic_dimension(hidden_states),
        "pca_dim_90": pca_intrinsic_dimension(hidden_states, threshold=0.9),
    }


# ---------------------------------------------------------------------------
# Per-step trajectory traces
# ---------------------------------------------------------------------------

def cosine_turn_angle(hidden_states: torch.Tensor) -> np.ndarray:
    """Cosine similarity between consecutive displacement vectors.

    Measures how much the trajectory "turns" at each interior point.
    A value of 1.0 means the trajectory continues straight; -1.0 means
    a full reversal.

    Parameters
    ----------
    hidden_states : Tensor of shape (T, D)

    Returns
    -------
    ndarray of shape (T-2,)
    """
    h = hidden_states.float()
    delta = h[1:] - h[:-1]                       # (T-1, D)
    d_prev, d_next = delta[:-1], delta[1:]        # (T-2, D)
    dot = (d_prev * d_next).sum(dim=-1)
    norms = d_prev.norm(dim=-1) * d_next.norm(dim=-1)
    cos = dot / norms.clamp(min=1e-12)
    return cos.numpy()


def trajectory_traces(hidden_states: torch.Tensor) -> Dict[str, np.ndarray]:
    """Compute per-step velocity, curvature, and cosine turn angle.

    Computes all three local traces in a single pass, sharing intermediate
    displacement vectors to avoid redundant work.

    Parameters
    ----------
    hidden_states : Tensor of shape (T, D)

    Returns
    -------
    dict with keys:
        velocity : ndarray of shape (T-1,)
        curvature : ndarray of shape (T-2,)
        cosine_turn : ndarray of shape (T-2,)
    """
    h = hidden_states.float()
    # Velocity
    delta = h[1:] - h[:-1]                         # (T-1, D)
    vel = delta.norm(dim=-1).numpy()                # (T-1,)

    if len(h) < 3:
        return {"velocity": vel, "curvature": np.array([]), "cosine_turn": np.array([])}

    # Curvature (same math as curvature() but reuses delta)
    A, B, C = h[:-2], h[1:-1], h[2:]
    AB, AC = B - A, C - A
    ab = AB.norm(dim=-1)
    bc = delta[1:].norm(dim=-1)                     # BC = delta[1:]
    ac = AC.norm(dim=-1)
    ab_sq = (AB * AB).sum(dim=-1)
    ac_sq = (AC * AC).sum(dim=-1)
    dot_ab_ac = (AB * AC).sum(dim=-1)
    cross_sq = (ab_sq * ac_sq - dot_ab_ac ** 2).clamp(min=0.0)
    area = 0.5 * torch.sqrt(cross_sq)
    denom = (ab * bc * ac).clamp(min=1e-12)
    curv = (4.0 * area / denom).numpy()             # (T-2,)

    # Cosine turn angle
    d_prev, d_next = delta[:-1], delta[1:]
    dot_turn = (d_prev * d_next).sum(dim=-1)
    norm_turn = d_prev.norm(dim=-1) * d_next.norm(dim=-1)
    cos_turn = (dot_turn / norm_turn.clamp(min=1e-12)).numpy()  # (T-2,)

    return {"velocity": vel, "curvature": curv, "cosine_turn": cos_turn}


# ---------------------------------------------------------------------------
# Random-walk null baselines
# ---------------------------------------------------------------------------

def random_walk_baseline(
    n_steps: int,
    dim: int,
    n_simulations: int = 100,
    step_std: float = 1.0,
    seed: int = 42,
) -> Dict[str, float]:
    """Generate null distribution statistics for directness and curvature.

    Creates *n_simulations* isotropic random walks in R^dim of length *n_steps*,
    computes directness and curvature for each, and returns mean and std of the
    null distribution for z-score computation.

    Only directness and curvature are baselined; intrinsic dimensionality
    metrics (TwoNN, PCA-ID) are not meaningful under isotropic random walks
    (TwoNN trivially returns ~T-1, making z-scores uninformative) and are
    prohibitively slow to compute (O(T^2 D) pairwise distances per simulation).

    Parameters
    ----------
    n_steps : int
        Number of points in the trajectory (T).
    dim : int
        Dimensionality (e.g. 3584 for DeepSeek-R1-7B).
    n_simulations : int
        Number of random walks to generate (default 100).
    step_std : float
        Standard deviation of each Gaussian step.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: curvature_mean_null_mean, curvature_mean_null_std,
        directness_null_mean, directness_null_std.
    """
    rng = np.random.RandomState(seed)

    curvatures = []
    directnesses = []

    for _ in range(n_simulations):
        steps = rng.randn(n_steps, dim).astype(np.float32) * step_std
        walk = np.cumsum(steps, axis=0)
        states = torch.tensor(walk)

        if n_steps < 3:
            continue

        curv = curvature(states)
        curvatures.append(float(curv.mean()) if len(curv) > 0 else 0.0)

        pl = path_length(states)
        nd = net_displacement(states)
        directnesses.append(nd / pl if pl > 1e-12 else 0.0)

    return {
        "curvature_mean_null_mean": float(np.mean(curvatures)) if curvatures else 0.0,
        "curvature_mean_null_std": float(np.std(curvatures)) if curvatures else 1.0,
        "directness_null_mean": float(np.mean(directnesses)) if directnesses else 0.0,
        "directness_null_std": float(np.std(directnesses)) if directnesses else 1.0,
    }


def random_walk_zscore(
    observed: Dict[str, float],
    null: Dict[str, float],
) -> Dict[str, float]:
    """Compute z-scores for observed trajectory metrics vs random-walk null.

    Parameters
    ----------
    observed : dict from ``trajectory_summary()``
    null : dict from ``random_walk_baseline()``

    Returns
    -------
    dict with keys: curvature_zscore, directness_zscore
    """
    def _z(obs_key, null_mean_key, null_std_key):
        std = null.get(null_std_key, 1.0)
        if std < 1e-12:
            return 0.0
        return (observed.get(obs_key, 0.0) - null.get(null_mean_key, 0.0)) / std

    return {
        "curvature_zscore": _z("curvature_mean", "curvature_mean_null_mean",
                               "curvature_mean_null_std"),
        "directness_zscore": _z("directness", "directness_null_mean",
                                "directness_null_std"),
    }


# ---------------------------------------------------------------------------
# Shuffled-difficulty controls
# ---------------------------------------------------------------------------

def shuffled_difficulty_control(
    traj_df: "pd.DataFrame",
    difficulty_col: str = "difficulty",
    metric_cols: Optional[List[str]] = None,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Test trajectory-difficulty associations against shuffled labels.

    For each metric, computes the observed Spearman correlation with
    difficulty, then generates *n_permutations* null correlations by
    randomly permuting the difficulty column.  The p-value is the fraction
    of permutations with |r| >= the observed |r|.

    Parameters
    ----------
    traj_df : DataFrame with trajectory metrics and a difficulty column
    difficulty_col : column name for difficulty values
    metric_cols : list of metric column names to test.  If *None*, uses
        ``["directness", "curvature_mean", "twonn_dim", "pca_dim_90"]``.
    n_permutations : number of label shuffles (default 1000)
    seed : random seed

    Returns
    -------
    dict mapping each metric to a sub-dict with keys:
        observed_r, perm_mean, perm_std, p_value, permutation_rs
    """
    from scipy.stats import spearmanr

    if metric_cols is None:
        metric_cols = ["directness", "curvature_mean", "twonn_dim", "pca_dim_90"]

    rng = np.random.RandomState(seed)
    difficulty = traj_df[difficulty_col].values

    results = {}
    for col in metric_cols:
        if col not in traj_df.columns:
            continue
        metric_vals = traj_df[col].values
        observed_r, _ = spearmanr(metric_vals, difficulty)

        perm_rs = np.empty(n_permutations)
        for i in range(n_permutations):
            shuffled = rng.permutation(difficulty)
            perm_rs[i], _ = spearmanr(metric_vals, shuffled)

        p_value = float(np.mean(np.abs(perm_rs) >= np.abs(observed_r)))

        results[col] = {
            "observed_r": float(observed_r),
            "perm_mean": float(perm_rs.mean()),
            "perm_std": float(perm_rs.std()),
            "p_value": p_value,
            "permutation_rs": perm_rs,
        }

    return results
