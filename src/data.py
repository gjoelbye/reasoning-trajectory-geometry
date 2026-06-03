"""Dataset loading, merging, and stratified selection.

Supports three problem domains:

**Codeforces** (competitive programming):
  - Easy2Hard-Bench E2H-Codeforces: provides Glicko-2 calibrated difficulty
    ratings (``unnorm_rating`` column on the native Codeforces scale).
  - open-r1/codeforces: provides full problem text, input/output specs,
    sample and official test cases, editorials, and algorithm tags.
  - The merge joins these on (contest_id, problem_index) to produce a unified
    dataset with both calibrated difficulty and full problem content.

**MATH** (mathematical reasoning, Hendrycks et al. 2021):
  - nlile/hendrycks-MATH-benchmark: mirror of the original
    hendrycks/competition_math (taken down via DMCA, January 2025).
    Provides problem text, solution, difficulty level (1--5), and
    problem type/subject (algebra, geometry, etc.).

**SATBench** (logical reasoning via SAT puzzles):
  - LLM4Code/SATBench: SAT puzzle instances encoded as natural-language
    scenarios with variable mappings and logical conditions.  Difficulty
    is stratified by clause count across five bins (4--50).
"""

import pandas as pd
from datasets import load_dataset
from typing import List, Tuple, Optional
from pathlib import Path


# Difficulty quintile bin boundaries on the Codeforces rating scale.
# Five bins spanning 800--2500, each 340 points wide.
DIFFICULTY_BINS = [
    (800, 1140),
    (1140, 1480),
    (1480, 1820),
    (1820, 2160),
    (2160, 2500),
]
PROBLEMS_PER_BIN = 100
TOTAL_PROBLEMS = 500

# MATH dataset constants (Hendrycks et al. 2021).
# Five discrete difficulty levels (integers 1--5).
MATH_DIFFICULTY_LEVELS = [1, 2, 3, 4, 5]
MATH_PROBLEMS_PER_LEVEL = 100
MATH_TOTAL_PROBLEMS = 500

# SATBench constants (logical reasoning via SAT puzzles).
# Five bins based on clause count, spanning 4--50.
SATBENCH_CLAUSE_BINS = [
    (4, 13),    # bin 1
    (13, 22),   # bin 2
    (22, 31),   # bin 3
    (31, 40),   # bin 4
    (40, 50),   # bin 5
]
SATBENCH_PROBLEMS_PER_BIN = 100
SATBENCH_TOTAL_PROBLEMS = 500


# ---------------------------------------------------------------------------
# Dataset loading -- Codeforces
# ---------------------------------------------------------------------------

def load_easy2hard_codeforces() -> pd.DataFrame:
    """Load the E2H-Codeforces subset from Easy2Hard-Bench.

    Concatenates the train and eval splits into a single DataFrame.
    The ``unnorm_rating`` column holds the raw Codeforces-scale difficulty.
    """
    ds = load_dataset("furonghuang-lab/Easy2Hard-Bench", "E2H-Codeforces")
    frames = [ds[split].to_pandas() for split in ds]
    return pd.concat(frames, ignore_index=True)


def load_openr1_codeforces() -> pd.DataFrame:
    """Load the open-r1/codeforces dataset (default config, train split).

    The ``rating`` column is an integer on the native Codeforces scale.
    """
    ds = load_dataset("open-r1/codeforces", name="default", split="train")
    return ds.to_pandas()


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def _make_join_key(contest_id, problem_index) -> pd.Series:
    """Create a join key from contest_id and problem index."""
    return contest_id.astype(str) + "_" + problem_index.astype(str)


def merge_datasets(
    e2h: pd.DataFrame,
    openr1: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Easy2Hard-Bench and open-r1 datasets on contest_id + problem index.

    Brings together E2H's Glicko-2 ratings with open-r1's full problem text,
    test cases, and editorial content.

    Parameters
    ----------
    e2h : DataFrame from ``load_easy2hard_codeforces()``
    openr1 : DataFrame from ``load_openr1_codeforces()``

    Returns
    -------
    Merged DataFrame with columns from both sources. E2H columns are kept
    as-is; open-r1 columns that collide get an ``_openr1`` suffix.
    """
    e2h = e2h.copy()
    openr1 = openr1.copy()

    e2h["join_key"] = _make_join_key(e2h["contest_id"], e2h["problem_index"])
    openr1["join_key"] = _make_join_key(openr1["contest_id"], openr1["index"])

    # Columns to bring from open-r1
    openr1_cols = [
        "join_key", "rating", "description", "input_format", "output_format",
        "examples", "editorial", "tags", "official_tests",
        "executable", "title", "note",
    ]
    # Keep only columns that actually exist in the dataframe
    openr1_cols = [c for c in openr1_cols if c in openr1.columns]

    merged = e2h.merge(
        openr1[openr1_cols],
        on="join_key",
        how="inner",
        suffixes=("", "_openr1"),
    )
    return merged


# ---------------------------------------------------------------------------
# Stratified selection
# ---------------------------------------------------------------------------

def select_stratified_problems(
    df: pd.DataFrame,
    rating_column: str = "unnorm_rating",
    bins: Optional[List[Tuple[int, int]]] = None,
    per_bin: int = PROBLEMS_PER_BIN,
    seed: int = 42,
) -> pd.DataFrame:
    """Select problems stratified by difficulty quintile.

    Samples ``per_bin`` problems from each difficulty bin for a total of
    ``len(bins) * per_bin`` problems. If a bin has fewer than ``per_bin``
    problems, all available problems in that bin are taken.

    Parameters
    ----------
    df : DataFrame with a numeric difficulty column
    rating_column : column name to use for stratification
    bins : list of (low, high) tuples defining the bins
    per_bin : number of problems to sample from each bin
    seed : random seed for reproducible sampling
    """
    if bins is None:
        bins = DIFFICULTY_BINS

    selected = []
    for low, high in bins:
        mask = (df[rating_column] >= low) & (df[rating_column] < high)
        bin_df = df[mask]
        n = min(per_bin, len(bin_df))
        if n > 0:
            sampled = bin_df.sample(n=n, random_state=seed)
            selected.append(sampled)

    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True)


def assign_quintile(
    df: pd.DataFrame,
    rating_column: str = "unnorm_rating",
    bins: Optional[List[Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """Add a ``quintile`` column (1--5) based on difficulty bins."""
    if bins is None:
        bins = DIFFICULTY_BINS
    df = df.copy()
    df["quintile"] = 0
    for i, (low, high) in enumerate(bins, start=1):
        mask = (df[rating_column] >= low) & (df[rating_column] < high)
        df.loc[mask, "quintile"] = i
    return df


# ---------------------------------------------------------------------------
# Problem formatting
# ---------------------------------------------------------------------------

def _is_nonempty_sequence(x) -> bool:
    """Check whether *x* is a non-empty array-like (list or ndarray).

    HuggingFace ``datasets`` converts list columns to numpy object arrays
    when calling ``.to_pandas()``, so a plain ``isinstance(x, list)`` check
    misses them.  This helper accepts both lists and ndarrays.
    """
    if x is None or isinstance(x, (str, float, int, bool)):
        return False
    return hasattr(x, "__len__") and len(x) > 0


def format_problem_text(row: pd.Series) -> str:
    """Format a single problem row into a prompt string.

    Prefers open-r1 ``description`` over E2H ``problem_main`` when available.
    Combines the problem statement, input/output specification, and sample I/O.
    """
    parts = []

    # Problem statement
    description = row.get("description")
    if not description or (isinstance(description, float)):
        description = row.get("problem_main", "")
    if description:
        parts.append(str(description).strip())

    # Input specification
    input_fmt = row.get("input_format")
    if not input_fmt or (isinstance(input_fmt, float)):
        input_fmt = row.get("input_spec", "")
    if input_fmt:
        parts.append(f"\nInput\n{str(input_fmt).strip()}")

    # Output specification
    output_fmt = row.get("output_format")
    if not output_fmt or (isinstance(output_fmt, float)):
        output_fmt = row.get("output_spec", "")
    if output_fmt:
        parts.append(f"\nOutput\n{str(output_fmt).strip()}")

    # Sample I/O from open-r1 examples (list or ndarray of dicts)
    examples = row.get("examples")
    if _is_nonempty_sequence(examples):
        parts.append("\nExamples")
        for ex in examples:
            if isinstance(ex, dict):
                inp = ex.get("input", "")
                out = ex.get("output", "")
                parts.append(f"\nInput\n{inp}\nOutput\n{out}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dataset loading -- MATH
# ---------------------------------------------------------------------------

def load_math_dataset() -> pd.DataFrame:
    """Load the MATH benchmark from HuggingFace.

    Concatenates train and test splits into a single DataFrame.
    Columns include ``problem``, ``level``, ``type``, and ``solution``.
    The ``level`` column is a string like ``"Level 1"``; this function adds
    an integer ``level_int`` column (1--5) for convenience.

    .. note::
       The original ``hendrycks/competition_math`` repository was taken
       down via DMCA in January 2025.  This function loads the equivalent
       ``nlile/hendrycks-MATH-benchmark`` mirror.
    """
    ds = load_dataset("nlile/hendrycks-MATH-benchmark")
    frames = [ds[split].to_pandas() for split in ds]
    df = pd.concat(frames, ignore_index=True)

    # ------ Column compatibility shim ------
    # The mirror uses "subject" instead of "type" and stores level as int.
    # Normalise to match the original schema that downstream code expects.
    if "subject" in df.columns and "type" not in df.columns:
        df.rename(columns={"subject": "type"}, inplace=True)
    if df["level"].dtype != object:
        # Convert integer level → "Level N" string for back-compat
        df["level"] = df["level"].apply(lambda x: f"Level {x}")

    # Parse "Level 1" .. "Level 5" → integer
    df["level_int"] = df["level"].str.extract(r"(\d)").astype(int)
    return df


def _sanitize_math_uid(uid: str) -> str:
    """Convert a MATH ``unique_id`` to an HDF5-safe ``join_key``.

    Replaces ``/`` with ``__`` so the key can be used as an HDF5 group name
    and as a cross-file identifier.

    Example: ``"train/geometry/309.json"`` → ``"train__geometry__309.json"``
    """
    return uid.replace("/", "__")


def select_stratified_math_problems(
    df: pd.DataFrame,
    level_column: str = "level_int",
    per_level: int = MATH_PROBLEMS_PER_LEVEL,
    seed: int = 42,
) -> pd.DataFrame:
    """Select problems stratified by MATH difficulty level.

    Samples ``per_level`` problems from each integer level (1--5).
    If a level has fewer than ``per_level`` problems, all available
    problems in that level are taken.

    A ``join_key`` column is automatically added by sanitising
    ``unique_id`` (``/`` → ``__``) so the key is safe for HDF5 group
    names and cross-file joins.

    Parameters
    ----------
    df : DataFrame with an integer difficulty level column
    level_column : column name for the integer difficulty level
    per_level : number of problems to sample from each level
    seed : random seed for reproducible sampling
    """
    selected = []
    for level in MATH_DIFFICULTY_LEVELS:
        level_df = df[df[level_column] == level]
        n = min(per_level, len(level_df))
        if n > 0:
            sampled = level_df.sample(n=n, random_state=seed)
            selected.append(sampled)

    if not selected:
        return pd.DataFrame()

    result = pd.concat(selected, ignore_index=True)

    # Add join_key from unique_id if not already present
    if "join_key" not in result.columns and "unique_id" in result.columns:
        result["join_key"] = result["unique_id"].apply(_sanitize_math_uid)

    return result


def format_math_prompt(row: pd.Series) -> str:
    r"""Format a MATH problem into a prompt string.

    Instructs the model to think step-by-step and provide the final
    answer in ``\boxed{}`` format, matching the MATH benchmark convention.

    Parameters
    ----------
    row : Series with at least a ``problem`` field.
    """
    problem = str(row.get("problem", "")).strip()
    return (
        "Solve the following math problem. Think step by step, "
        "then provide your final answer in \\boxed{} format.\n\n"
        f"{problem}"
    )


# ---------------------------------------------------------------------------
# Dataset loading -- SATBench
# ---------------------------------------------------------------------------

def load_satbench_dataset() -> pd.DataFrame:
    """Load the SATBench dataset from HuggingFace.

    Returns a DataFrame with columns including ``scenario``,
    ``variable_mapping``, ``conditions``, ``question``,
    ``num_clauses``, ``num_vars``, ``satisfiable``, ``clauses``,
    and ``dims``.

    Falls back to a local parquet cache if available.
    """
    cache_path = Path("data/raw/satbench_train.parquet")
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    ds = load_dataset("LLM4Code/SATBench", split="train")
    return ds.to_pandas()


def select_stratified_satbench_problems(
    df: pd.DataFrame,
    clause_column: str = "num_clauses",
    bins: Optional[List[Tuple[int, int]]] = None,
    per_bin: int = SATBENCH_PROBLEMS_PER_BIN,
    seed: int = 42,
) -> pd.DataFrame:
    """Select problems stratified by clause count, balanced SAT/UNSAT.

    Samples ``per_bin`` problems from each clause bin, targeting
    ~50% SAT and ~50% UNSAT within each bin.  Assigns ``join_key``
    (HDF5-safe) and ``num_clauses_bin`` columns.

    Parameters
    ----------
    df : DataFrame with ``num_clauses`` and ``satisfiable`` columns
    clause_column : column name for clause count
    bins : list of (low, high) tuples defining the bins
    per_bin : number of problems to sample from each bin
    seed : random seed for reproducible sampling
    """
    if bins is None:
        bins = SATBENCH_CLAUSE_BINS

    selected = []
    for bin_idx, (low, high) in enumerate(bins, start=1):
        mask = (df[clause_column] >= low) & (df[clause_column] < high)
        bin_df = df[mask]

        # Balance SAT/UNSAT within each bin
        sat_df = bin_df[bin_df["satisfiable"] == True]
        unsat_df = bin_df[bin_df["satisfiable"] == False]

        half = per_bin // 2
        n_sat = min(half, len(sat_df))
        n_unsat = min(per_bin - n_sat, len(unsat_df))
        # If one side is short, take more from the other
        if n_sat < half:
            n_unsat = min(per_bin - n_sat, len(unsat_df))
        if n_unsat < per_bin - n_sat:
            n_sat = min(per_bin - n_unsat, len(sat_df))

        if n_sat > 0:
            selected.append(sat_df.sample(n=n_sat, random_state=seed))
        if n_unsat > 0:
            selected.append(unsat_df.sample(n=n_unsat, random_state=seed))

    if not selected:
        return pd.DataFrame()

    result = pd.concat(selected, ignore_index=True)
    result["join_key"] = [f"sat_{i}" for i in range(len(result))]
    # Assign bin labels
    result["num_clauses_bin"] = 0
    for bin_idx, (low, high) in enumerate(bins, start=1):
        mask = (result[clause_column] >= low) & (result[clause_column] < high)
        result.loc[mask, "num_clauses_bin"] = bin_idx

    return result


def format_satbench_prompt(row: pd.Series) -> str:
    """Format a SATBench problem into a prompt string.

    Combines the scenario, variable mapping, conditions, and question
    into a structured prompt.  The generation prefix is added separately
    by ``format_sat_generation_prompt()`` in ``src/models.py``.

    Parameters
    ----------
    row : Series with ``scenario``, ``variable_mapping``, ``conditions``,
        and ``question`` fields.
    """
    parts = []

    scenario = str(row.get("scenario", "")).strip()
    if scenario:
        parts.append(scenario)

    var_mapping = row.get("variable_mapping", "")
    if var_mapping:
        parts.append(f"\nVariable meanings:\n{str(var_mapping).strip()}")

    conditions = row.get("conditions", [])
    if _is_nonempty_sequence(conditions):
        cond_lines = [f"{i}. {c}" for i, c in enumerate(conditions, 1)]
        parts.append("\nConditions:\n" + "\n".join(cond_lines))
    elif isinstance(conditions, str) and conditions.strip():
        parts.append(f"\nConditions:\n{conditions}")

    question = str(row.get("question", "")).strip()
    if question:
        parts.append(f"\n{question}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_problems(df: pd.DataFrame, path: Path) -> None:
    """Save a problem DataFrame to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_problems(path: Path) -> pd.DataFrame:
    """Load a previously saved problem DataFrame from parquet."""
    return pd.read_parquet(Path(path))
