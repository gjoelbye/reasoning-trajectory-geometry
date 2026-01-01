"""Unified difficulty interface.

Two explicit difficulty sources:

- **Native**: External ground truth — Glicko-2 for code, MATH levels for
  math, clause count for SAT.
- **Pooled IRT**: Hierarchical 1PL calibrated across models.

All functions return DataFrames with columns ``[item_id, difficulty,
difficulty_source]``.  Downstream code should use ``get_native_difficulty()``
or ``get_pooled_difficulty()`` explicitly.

Usage
-----
    from src.difficulty import (
        get_native_difficulty,
        get_pooled_difficulty,
        assign_difficulty_quintiles,
    )

    # Native difficulty (Glicko-2 for code, MATH levels for math, clause count for SAT)
    df = get_native_difficulty(problems, domain="code")
    df = get_native_difficulty(problems, domain="sat")

    # Pooled IRT difficulty
    df = get_pooled_difficulty("data/results/pooled_irt/code/pooled_difficulties.parquet")

    # Quintile assignment for any continuous difficulty column
    df = assign_difficulty_quintiles(df, difficulty_col="difficulty", n_quintiles=5)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Explicit difficulty API (preferred)
# ---------------------------------------------------------------------------

def get_native_difficulty(
    problems_df: pd.DataFrame,
    domain: str,
    item_id_col: str = "join_key",
) -> pd.DataFrame:
    """Return the native (external ground-truth) difficulty for the domain.

    Parameters
    ----------
    problems_df : DataFrame
        Problem metadata.
    domain : str
        ``"code"`` / ``"codeforces"`` → Glicko-2; ``"math"`` → MATH levels;
        ``"sat"`` / ``"satbench"`` → clause count.
    item_id_col : str
        Column in *problems_df* that serves as the problem identifier.

    Returns
    -------
    DataFrame with columns [item_id, difficulty, difficulty_source].
    ``difficulty_source`` is ``"glicko2"``, ``"math_level"``, or
    ``"clause_count"``.
    """
    domain = _normalize_domain(domain)
    if domain == "code":
        return _glicko2_difficulty(problems_df, item_id_col)
    elif domain == "sat":
        return _clause_count_difficulty(problems_df, item_id_col)
    else:
        return _math_level_difficulty(problems_df, item_id_col)


def get_pooled_difficulty(
    pooled_path: Union[str, Path],
) -> pd.DataFrame:
    """Return pooled IRT difficulty from a parquet file.

    Parameters
    ----------
    pooled_path : str or Path
        Path to ``pooled_difficulties.parquet``.

    Returns
    -------
    DataFrame with columns [item_id, difficulty, difficulty_source].
    ``difficulty_source`` is ``"pooled_irt"``.

    Raises
    ------
    FileNotFoundError
        If *pooled_path* does not exist.
    """
    p = Path(pooled_path)
    if not p.exists():
        raise FileNotFoundError(f"Pooled IRT file not found: {p}")
    pooled_df = pd.read_parquet(p)
    missing = {"item_id", "difficulty"} - set(pooled_df.columns)
    if missing:
        raise ValueError(
            f"Pooled IRT file missing columns: {missing}. "
            f"Found: {list(pooled_df.columns)}"
        )
    result = pooled_df[["item_id", "difficulty"]].copy()
    result["difficulty_source"] = "pooled_irt"
    return result


# ---------------------------------------------------------------------------
# Quintile assignment
# ---------------------------------------------------------------------------

def assign_difficulty_quintiles(
    df: pd.DataFrame,
    difficulty_col: str = "difficulty",
    n_quintiles: int = 5,
    labels: Optional[list] = None,
) -> pd.DataFrame:
    """Assign quintile labels based on a continuous difficulty column.

    Parameters
    ----------
    df : DataFrame
        Must contain *difficulty_col*.
    difficulty_col : str
        Column with continuous difficulty values.
    n_quintiles : int
        Number of bins (default 5).
    labels : list, optional
        Custom labels for quintiles.  If None, uses 1..n_quintiles.

    Returns
    -------
    DataFrame with added ``difficulty_quintile`` column.
    """
    out = df.copy()

    # First pass: compute bin edges (duplicates="drop" may merge bins)
    _, bin_edges = pd.qcut(
        out[difficulty_col], q=n_quintiles, retbins=True, duplicates="drop",
    )
    n_actual_bins = len(bin_edges) - 1

    if labels is None:
        labels = list(range(1, n_actual_bins + 1))
    else:
        labels = list(labels)[:n_actual_bins]

    out["difficulty_quintile"] = pd.qcut(
        out[difficulty_col],
        q=n_quintiles,
        labels=labels,
        duplicates="drop",
    )
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _normalize_domain(domain: str) -> str:
    """Normalize domain string to 'code', 'math', or 'sat'."""
    d = domain.lower().strip()
    if d in ("code", "codeforces", "cf"):
        return "code"
    if d in ("math",):
        return "math"
    if d in ("sat", "satbench"):
        return "sat"
    raise ValueError(f"Unknown domain {domain!r}; expected 'code', 'math', or 'sat'.")


def _glicko2_difficulty(
    problems_df: pd.DataFrame,
    item_id_col: str = "join_key",
) -> pd.DataFrame:
    """Extract Glicko-2 difficulty from Codeforces problem metadata."""
    if "unnorm_rating" not in problems_df.columns:
        raise ValueError(
            "Codeforces problems must have 'unnorm_rating' column "
            "for Glicko-2 difficulty."
        )
    return pd.DataFrame({
        "item_id": problems_df[item_id_col].astype(str),
        "difficulty": problems_df["unnorm_rating"].values,
        "difficulty_source": "glicko2",
    })


def _clause_count_difficulty(
    problems_df: pd.DataFrame,
    item_id_col: str = "join_key",
) -> pd.DataFrame:
    """Extract clause count as continuous difficulty for SAT problems."""
    if "num_clauses" not in problems_df.columns:
        raise ValueError(
            "SAT problems must have 'num_clauses' column "
            "for clause-count difficulty."
        )
    if item_id_col in problems_df.columns:
        item_ids = problems_df[item_id_col].astype(str)
    else:
        item_ids = problems_df.index.astype(str)

    return pd.DataFrame({
        "item_id": item_ids.values,
        "difficulty": problems_df["num_clauses"].values.astype(float),
        "difficulty_source": "clause_count",
    })


def _math_level_difficulty(
    problems_df: pd.DataFrame,
    item_id_col: str = "join_key",
) -> pd.DataFrame:
    """Extract MATH level as ordinal difficulty."""
    if "level_int" not in problems_df.columns:
        raise ValueError(
            "MATH problems must have 'level_int' column "
            "for level-based difficulty."
        )
    # Use join_key if available, otherwise index
    if item_id_col in problems_df.columns:
        item_ids = problems_df[item_id_col].astype(str)
    else:
        item_ids = problems_df.index.astype(str)

    return pd.DataFrame({
        "item_id": item_ids.values,
        "difficulty": problems_df["level_int"].values.astype(float),
        "difficulty_source": "math_level",
    })
