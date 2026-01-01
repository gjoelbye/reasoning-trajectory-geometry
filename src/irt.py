"""Binomial Rasch (1PL) and 2PL IRT model fitting.

Fits Item Response Theory models to aggregated LLM response data using a
binomial likelihood.  Each observation is (model, problem, n_correct, n_total)
rather than individual binary outcomes, which naturally handles multiple runs
per model without hierarchical run-level parameters.

The binomial Rasch (1PL) model fixes discrimination a=1:
    k ~ Binomial(n, sigmoid(theta[model] - b[item]))

The 2PL variant adds item-specific discrimination (for validation only):
    k ~ Binomial(n, sigmoid(exp(log_a[item]) * (theta[model] - b[item])))

MAP estimation with weak Gaussian priors, optimised with Adam.
"""

import torch
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Binomial 1PL (Rasch) MAP estimation
# ---------------------------------------------------------------------------

def _fit_binomial_rasch(
    models: np.ndarray,
    items: np.ndarray,
    n_correct: np.ndarray,
    n_total: np.ndarray,
    n_models: int,
    n_items: int,
    lr: float = 0.05,
    epochs: int = 2000,
    patience: int = 200,
    min_delta: float = 1e-4,
    prior_sigma_ability: float = 2.0,
    prior_sigma_diff: float = 2.0,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Fit a binomial Rasch (1PL) IRT model via MAP estimation with Adam.

    Parameters
    ----------
    models : int array, shape (N,) -- model index per observation
    items : int array, shape (N,) -- item index per observation
    n_correct : int array, shape (N,) -- number correct per observation
    n_total : int array, shape (N,) -- number of trials per observation
    n_models : total number of unique models
    n_items : total number of unique items
    lr : learning rate for Adam
    epochs : maximum number of optimization steps
    patience : early-stop patience (epochs without improvement)
    min_delta : minimum improvement threshold for early stopping
    prior_sigma_ability : std of N(0, sigma^2) prior on theta
    prior_sigma_diff : std of N(0, sigma^2) prior on b
    device : "cpu" or "cuda"
    verbose : print loss every 200 epochs

    Returns
    -------
    dict with keys "ability", "diff", "disc", "final_epoch", "final_loss"
    """
    dev = torch.device(device)

    model_t = torch.tensor(models, dtype=torch.long, device=dev)
    item_t = torch.tensor(items, dtype=torch.long, device=dev)
    k = torch.tensor(n_correct, dtype=torch.float32, device=dev)
    n = torch.tensor(n_total, dtype=torch.float32, device=dev)
    zeros = torch.zeros_like(k)

    theta = torch.nn.Parameter(torch.zeros(n_models, device=dev))
    b = torch.nn.Parameter(torch.zeros(n_items, device=dev))

    optimizer = torch.optim.Adam([theta, b], lr=lr)

    best_loss = float("inf")
    best_theta = theta.detach().clone()
    best_b = b.detach().clone()
    epochs_without_improvement = 0
    final_epoch = epochs

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        logit = theta[model_t] - b[item_t]

        # Binomial NLL: -sum(k * logit - n * log(1 + exp(logit)))
        nll = -(k * logit - n * torch.logaddexp(zeros, logit)).sum()

        prior = (
            0.5 * (theta ** 2).sum() / (prior_sigma_ability ** 2)
            + 0.5 * (b ** 2).sum() / (prior_sigma_diff ** 2)
        )

        loss = nll + prior
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if verbose and (epoch % 200 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>5d}/{epochs}  loss = {current_loss:.4f}")

        if best_loss - current_loss > min_delta:
            best_loss = current_loss
            best_theta = theta.detach().clone()
            best_b = b.detach().clone()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            final_epoch = epoch
            if verbose:
                print(f"  Early stop at epoch {epoch} "
                      f"(no improvement for {patience} epochs, "
                      f"loss = {current_loss:.4f})")
            break
    else:
        final_epoch = epochs

    return {
        "ability": best_theta.cpu().numpy(),
        "diff": best_b.cpu().numpy(),
        "disc": np.ones(n_items),
        "final_epoch": final_epoch,
        "final_loss": best_loss,
    }


# ---------------------------------------------------------------------------
# Binomial 2PL MAP estimation
# ---------------------------------------------------------------------------

def _fit_binomial_2pl(
    models: np.ndarray,
    items: np.ndarray,
    n_correct: np.ndarray,
    n_total: np.ndarray,
    n_models: int,
    n_items: int,
    lr: float = 0.05,
    epochs: int = 2000,
    patience: int = 200,
    min_delta: float = 1e-4,
    prior_sigma_ability: float = 2.0,
    prior_sigma_diff: float = 2.0,
    prior_mu_log_disc: float = 0.0,
    prior_sigma_log_disc: float = 0.5,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Fit a binomial 2PL IRT model via MAP estimation with Adam.

    Parameters
    ----------
    models : int array, shape (N,) -- model index per observation
    items : int array, shape (N,) -- item index per observation
    n_correct : int array, shape (N,) -- number correct per observation
    n_total : int array, shape (N,) -- number of trials per observation
    n_models : total number of unique models
    n_items : total number of unique items
    lr : learning rate for Adam
    epochs : maximum number of optimization steps
    patience : early-stop patience
    min_delta : minimum improvement threshold
    prior_sigma_ability : std of N(0, sigma^2) prior on theta
    prior_sigma_diff : std of N(0, sigma^2) prior on b
    prior_mu_log_disc : mean of N(mu, sigma^2) prior on log(a)
    prior_sigma_log_disc : std of N(mu, sigma^2) prior on log(a)
    device : "cpu" or "cuda"
    verbose : print loss every 200 epochs

    Returns
    -------
    dict with keys "ability", "diff", "disc", "final_epoch", "final_loss"
    """
    dev = torch.device(device)

    model_t = torch.tensor(models, dtype=torch.long, device=dev)
    item_t = torch.tensor(items, dtype=torch.long, device=dev)
    k = torch.tensor(n_correct, dtype=torch.float32, device=dev)
    n = torch.tensor(n_total, dtype=torch.float32, device=dev)
    zeros = torch.zeros_like(k)

    theta = torch.nn.Parameter(torch.zeros(n_models, device=dev))
    b = torch.nn.Parameter(torch.zeros(n_items, device=dev))
    log_a = torch.nn.Parameter(torch.zeros(n_items, device=dev))

    optimizer = torch.optim.Adam([theta, b, log_a], lr=lr)

    best_loss = float("inf")
    best_theta = theta.detach().clone()
    best_b = b.detach().clone()
    best_log_a = log_a.detach().clone()
    epochs_without_improvement = 0
    final_epoch = epochs

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        a = torch.exp(log_a)
        logit = a[item_t] * (theta[model_t] - b[item_t])

        nll = -(k * logit - n * torch.logaddexp(zeros, logit)).sum()

        prior = (
            0.5 * (theta ** 2).sum() / (prior_sigma_ability ** 2)
            + 0.5 * (b ** 2).sum() / (prior_sigma_diff ** 2)
            + 0.5 * ((log_a - prior_mu_log_disc) ** 2).sum()
            / (prior_sigma_log_disc ** 2)
        )

        loss = nll + prior
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if verbose and (epoch % 200 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>5d}/{epochs}  loss = {current_loss:.4f}")

        if best_loss - current_loss > min_delta:
            best_loss = current_loss
            best_theta = theta.detach().clone()
            best_b = b.detach().clone()
            best_log_a = log_a.detach().clone()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            final_epoch = epoch
            if verbose:
                print(f"  Early stop at epoch {epoch} "
                      f"(no improvement for {patience} epochs, "
                      f"loss = {current_loss:.4f})")
            break
    else:
        final_epoch = epochs

    return {
        "ability": best_theta.cpu().numpy(),
        "diff": best_b.cpu().numpy(),
        "disc": torch.exp(best_log_a).cpu().numpy(),
        "final_epoch": final_epoch,
        "final_loss": best_loss,
    }


# ---------------------------------------------------------------------------
# Public fitting interface
# ---------------------------------------------------------------------------

def fit_irt_model(
    response_df: pd.DataFrame,
    model_type: str = "1pl",
    num_epochs: int = 2000,
    patience: int = 200,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """Fit a binomial IRT model to an aggregated response DataFrame.

    Parameters
    ----------
    response_df : DataFrame with columns [model_id, item_id, n_correct, n_total]
    model_type : "1pl" (binomial Rasch) or "2pl" (binomial 2PL)
    num_epochs : maximum number of optimization epochs
    patience : early-stopping patience (epochs without improvement)
    device : "cpu" or "cuda"
    verbose : whether to print training progress

    Returns
    -------
    dict with keys:
        - "ability"      : list of model ability estimates
        - "diff"         : list of item difficulty estimates
        - "disc"         : list of item discrimination estimates
        - "item_ids"     : list of item identifiers (strings)
        - "model_ids"    : list of model identifiers (strings)
        - "subject_ids"  : alias for "model_ids" (backward compat)
        - "model_type"   : "1pl" or "2pl"
        - "final_epoch"  : epoch at which training stopped
        - "final_loss"   : best loss achieved
        - "n_params"     : number of free parameters
    """
    if model_type not in ("1pl", "2pl"):
        raise ValueError(
            f"model_type must be '1pl' or '2pl'; got '{model_type}'."
        )

    df = response_df.copy()
    df["model_id"] = df["model_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    model_ids = sorted(df["model_id"].unique())
    item_ids = sorted(df["item_id"].unique())
    model_map = {s: i for i, s in enumerate(model_ids)}
    item_map = {s: i for i, s in enumerate(item_ids)}

    models = df["model_id"].map(model_map).values.astype(np.int64)
    items = df["item_id"].map(item_map).values.astype(np.int64)
    n_correct = df["n_correct"].values.astype(np.float64)
    n_total = df["n_total"].values.astype(np.float64)

    n_mod = len(model_ids)
    n_item = len(item_ids)

    if verbose:
        total_trials = int(n_total.sum())
        print(
            f"Fitting binomial {model_type.upper()} IRT: {n_mod} models, "
            f"{n_item} items, {len(df)} observations "
            f"({total_trials} total trials)"
        )

    if model_type == "1pl":
        raw = _fit_binomial_rasch(
            models=models,
            items=items,
            n_correct=n_correct,
            n_total=n_total,
            n_models=n_mod,
            n_items=n_item,
            epochs=num_epochs,
            patience=patience,
            device=device,
            verbose=verbose,
        )
        n_params = n_mod + n_item  # theta + b
    else:
        raw = _fit_binomial_2pl(
            models=models,
            items=items,
            n_correct=n_correct,
            n_total=n_total,
            n_models=n_mod,
            n_items=n_item,
            epochs=num_epochs,
            patience=patience,
            device=device,
            verbose=verbose,
        )
        n_params = n_mod + 2 * n_item  # theta + b + log_a

    return {
        "ability": raw["ability"].tolist(),
        "diff": raw["diff"].tolist(),
        "disc": raw["disc"].tolist(),
        "item_ids": item_ids,
        "model_ids": model_ids,
        "subject_ids": model_ids,  # backward compat alias
        "model_type": model_type,
        "final_epoch": raw["final_epoch"],
        "final_loss": raw["final_loss"],
        "n_params": n_params,
    }


def compute_irt_information_criteria(
    params: Dict,
    response_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute AIC and BIC for a fitted binomial IRT model.

    Parameters
    ----------
    params : dict returned by ``fit_irt_model``
    response_df : DataFrame with columns [model_id, item_id, n_correct, n_total]

    Returns
    -------
    dict with keys: aic, bic, n_obs, n_params, neg_log_lik
    """
    n_obs = len(response_df)
    n_params = params.get("n_params", 0)

    df = response_df.copy()
    df["model_id"] = df["model_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    model_map = {s: i for i, s in enumerate(params["model_ids"])}
    item_map = {s: i for i, s in enumerate(params["item_ids"])}

    ability = np.array(params["ability"])
    diff = np.array(params["diff"])
    disc = np.array(params["disc"])

    theta_j = ability[df["model_id"].map(model_map).values]
    b_i = diff[df["item_id"].map(item_map).values]
    a_i = disc[df["item_id"].map(item_map).values]
    k = df["n_correct"].values.astype(np.float64)
    n = df["n_total"].values.astype(np.float64)

    logit = a_i * (theta_j - b_i)
    # Binomial NLL: -sum(k * logit - n * log(1 + exp(logit)))
    nll = -(k * logit - n * np.logaddexp(0, logit)).sum()

    aic = 2 * n_params + 2 * nll
    bic = n_params * np.log(n_obs) + 2 * nll

    return {
        "aic": float(aic),
        "bic": float(bic),
        "n_obs": n_obs,
        "n_params": n_params,
        "neg_log_lik": float(nll),
    }


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

def extract_difficulties(params: Dict) -> pd.DataFrame:
    """Extract item difficulty (and discrimination) from fitted IRT parameters.

    Parameters
    ----------
    params : dict returned by ``fit_irt_model``

    Returns
    -------
    DataFrame with columns [item_id, difficulty] and optionally [discrimination]
    """
    diffs = params.get("diff", [])
    if not diffs:
        import warnings
        warnings.warn("IRT params contain no difficulty estimates ('diff' key is empty)")

    result = pd.DataFrame({
        "item_id": list(params.get("item_ids", [])),
        "difficulty": list(diffs),
    })

    if "disc" in params:
        result["discrimination"] = list(params["disc"])

    return result


def extract_abilities(params: Dict) -> pd.DataFrame:
    """Extract model ability estimates from fitted IRT parameters.

    Handles both the raw format from ``fit_irt_model()`` (flat lists under
    ``"ability"`` and ``"model_ids"``/``"subject_ids"``) and the records
    format saved by scripts (list of dicts under ``"abilities"``).
    """
    # Records format: "abilities" key with list of dicts
    if "abilities" in params and isinstance(params["abilities"], list):
        records = params["abilities"]
        if records and isinstance(records[0], dict):
            return pd.DataFrame(records)

    # Flat format: "ability" and "model_ids"/"subject_ids" keys
    ability = list(params.get("ability", []))
    model_ids = list(params.get(
        "model_ids",
        params.get("subject_ids", range(len(ability))),
    ))
    return pd.DataFrame({"model_id": model_ids, "ability": ability})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compare_irt_to_native_rating(
    irt_difficulties: pd.DataFrame,
    native_ratings: pd.DataFrame,
    irt_id_col: str = "item_id",
    native_id_col: str = "item_id",
    native_rating_col: str = "unnorm_rating",
) -> Dict[str, float]:
    """Compute correlation between IRT difficulty and native Codeforces rating.

    Parameters
    ----------
    irt_difficulties : DataFrame with columns [item_id, difficulty]
    native_ratings : DataFrame with columns [item_id, <native_rating_col>]

    Returns
    -------
    dict with pearson_r, pearson_p, spearman_rho, spearman_p, n
    """
    from scipy.stats import pearsonr, spearmanr

    merged = irt_difficulties.merge(
        native_ratings,
        left_on=irt_id_col,
        right_on=native_id_col,
    )

    if len(merged) < 3:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n": len(merged),
        }

    r_p, p_p = pearsonr(merged["difficulty"], merged[native_rating_col])
    r_s, p_s = spearmanr(merged["difficulty"], merged[native_rating_col])

    return {
        "pearson_r": float(r_p),
        "pearson_p": float(p_p),
        "spearman_rho": float(r_s),
        "spearman_p": float(p_s),
        "n": len(merged),
    }


def compare_irt_to_math_levels(
    irt_difficulties: pd.DataFrame,
    math_problems: pd.DataFrame,
    irt_id_col: str = "item_id",
    math_id_col: str = "item_id",
    math_level_col: str = "level_int",
) -> Dict[str, float]:
    """Compute association between IRT difficulty and ordinal MATH levels.

    Since MATH levels are ordinal (1--5), uses Spearman rank correlation
    and Kruskal-Wallis H-test for group differences.

    Parameters
    ----------
    irt_difficulties : DataFrame with columns [item_id, difficulty]
    math_problems : DataFrame with columns [item_id, <math_level_col>]

    Returns
    -------
    dict with spearman_rho, spearman_p, kruskal_h, kruskal_p,
    mean_irt_by_level (dict mapping level -> mean difficulty), n
    """
    from scipy.stats import spearmanr, kruskal

    merged = irt_difficulties.merge(
        math_problems,
        left_on=irt_id_col,
        right_on=math_id_col,
    )

    if len(merged) < 3:
        return {
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "kruskal_h": float("nan"),
            "kruskal_p": float("nan"),
            "mean_irt_by_level": {},
            "n": len(merged),
        }

    r_s, p_s = spearmanr(merged["difficulty"], merged[math_level_col])

    # Kruskal-Wallis: are IRT difficulties different across MATH levels?
    groups = [
        grp["difficulty"].values
        for _, grp in merged.groupby(math_level_col)
        if len(grp) >= 1
    ]
    if len(groups) >= 2:
        h_stat, h_p = kruskal(*groups)
    else:
        h_stat, h_p = float("nan"), float("nan")

    mean_by_level = (
        merged.groupby(math_level_col)["difficulty"]
        .mean()
        .to_dict()
    )

    return {
        "spearman_rho": float(r_s),
        "spearman_p": float(p_s),
        "kruskal_h": float(h_stat),
        "kruskal_p": float(h_p),
        "mean_irt_by_level": {int(k): float(v) for k, v in mean_by_level.items()},
        "n": len(merged),
    }


def compare_irt_to_clause_count(
    irt_difficulties: pd.DataFrame,
    sat_problems: pd.DataFrame,
    irt_id_col: str = "item_id",
    sat_id_col: str = "item_id",
    clause_col: str = "num_clauses",
) -> Dict[str, float]:
    """Compute correlation between IRT difficulty and SAT clause count.

    Parameters
    ----------
    irt_difficulties : DataFrame with columns [item_id, difficulty]
    sat_problems : DataFrame with columns [item_id, <clause_col>]

    Returns
    -------
    dict with pearson_r, pearson_p, spearman_rho, spearman_p, n
    """
    from scipy.stats import pearsonr, spearmanr

    merged = irt_difficulties.merge(
        sat_problems,
        left_on=irt_id_col,
        right_on=sat_id_col,
    )

    if len(merged) < 3:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n": len(merged),
        }

    r_p, p_p = pearsonr(merged["difficulty"], merged[clause_col])
    r_s, p_s = spearmanr(merged["difficulty"], merged[clause_col])

    return {
        "pearson_r": float(r_p),
        "pearson_p": float(p_p),
        "spearman_rho": float(r_s),
        "spearman_p": float(p_s),
        "n": len(merged),
    }


# ---------------------------------------------------------------------------
# IRT model functions (for plotting and analysis)
# ---------------------------------------------------------------------------

def icc_probability(
    theta: np.ndarray,
    difficulty: float,
    discrimination: float = 1.0,
    guessing: float = 0.0,
) -> np.ndarray:
    """Compute the ICC probability P(correct | theta) under the 3PL model.

    P(theta) = c + (1 - c) / (1 + exp(-a * (theta - b)))

    Parameters
    ----------
    theta : array of ability values
    difficulty : item difficulty parameter (b)
    discrimination : item discrimination parameter (a)
    guessing : pseudo-guessing parameter (c)
    """
    return guessing + (1 - guessing) / (1 + np.exp(-discrimination * (theta - difficulty)))


def item_information(
    theta: np.ndarray,
    difficulty: float,
    discrimination: float = 1.0,
    guessing: float = 0.0,
) -> np.ndarray:
    """Compute the item information function I(theta).

    For the 3PL model:
        I(theta) = a^2 * (P*(theta) - c)^2 / ((1 - c)^2 * P(theta) * (1 - P(theta)))
    where P*(theta) = 1 / (1 + exp(-a(theta - b))).
    """
    z = discrimination * (theta - difficulty)
    p_star = 1.0 / (1.0 + np.exp(-z))
    p = guessing + (1.0 - guessing) * p_star
    q = 1.0 - p
    numerator = (discrimination ** 2) * ((p_star - guessing) ** 2)
    denominator = (1.0 - guessing) ** 2 * p * q + 1e-12
    return numerator / denominator
