"""Run Stage 02 (CoT structural analysis) from the experiment pipeline.

Extracts chain-of-thought structural patterns, evaluates correctness,
and saves results so downstream stages and notebooks can load cached files.

Usage
-----
    python scripts/run_cot_analysis.py --config pipeline/code/deepseek-r1-7b
    python scripts/run_cot_analysis.py --config pipeline/code/qwen-7b --workers 12
    python scripts/run_cot_analysis.py --config pipeline/code/deepseek-r1-7b --force
    python scripts/run_cot_analysis.py --config pipeline/code/deepseek-r1-7b --patterns-only

Stage
-----
    02  CoT structural analysis (patterns, correctness evaluation, repetition)

The ``--patterns-only`` flag recomputes only pattern detection and repetition
metrics from the existing traces, patching them into the existing
``cot_analysis.parquet`` without re-running correctness evaluation.
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
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.cot_patterns import detect_all_patterns_normalized, compute_repetition_score
from src.data import load_problems, _is_nonempty_sequence
from src.evaluation import extract_math_answer, extract_python_code, evaluate_on_tests
from src.config import load_config
from src.models import THINK_END, THINK_START, parse_think_response
from src.parallel_workers import (
    _worker_init,
    compute_patterns_worker,
    evaluate_math_trace_worker,
    evaluate_sat_trace_worker,
    evaluate_trace_worker,
)

# ---------------------------------------------------------------------------
# Path configuration — set from --config in main()
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Module-level globals; set by main() from the loaded config.
DOMAIN             = "codeforces"
MODEL_NAME         = None
TRACES_PATH        = None
PROBLEMS_PATH      = None
COT_ANALYSIS_PATH  = None


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


def _detect_reasoning_boundary(trace_text: str, has_think: bool,
                               domain: str) -> tuple:
    """Detect where reasoning ends and answer/code output begins.

    Uses a character-fraction approximation of the reasoning-phase boundary.

    Parameters
    ----------
    trace_text : str
        Full generation trace.
    has_think : bool
        Whether the trace contains ``<think>`` tags.
    domain : str
        ``"codeforces"`` or ``"math"``.

    Returns
    -------
    (reasoning_end_frac, boundary_type) : (float, str)
        ``reasoning_end_frac`` is the character fraction at which reasoning
        ends (0–1).  ``boundary_type`` is one of ``"think_tag"``,
        ``"code_fence"``, ``"boxed"``, or ``"none"``.
    """
    import re

    total_chars = len(trace_text)
    if total_chars == 0:
        return (1.0, "none")

    # Priority 1: </think> tag (R1-style models)
    if has_think:
        pos = trace_text.rfind("</think>")
        if pos > 0:
            frac = pos / total_chars
            return (frac, "think_tag")

    # Priority 2: domain-specific heuristics
    if domain == "codeforces":
        m = (re.search(r"```[Pp]ython", trace_text)
             or re.search(r"```[Cc]pp", trace_text)
             or re.search(r"```", trace_text))
        if m:
            frac = m.start() / total_chars
            return (frac, "code_fence")

        # XML-style code tags (Claude models)
        for tag in (r"<answer>", r"<solution>", r"<[Pp]ython[^>]*>",
                    r"<code>", r"<code_block[^>]*>", r"<invoke\b"):
            m = re.search(tag, trace_text)
            if m:
                frac = m.start() / total_chars
                return (frac, "xml_tag")

    elif domain == "math":
        pos = trace_text.find("\\boxed{")
        if pos > 0:
            frac = pos / total_chars
            return (frac, "boxed")

    elif domain == "sat":
        # Look for SATISFIABLE/UNSATISFIABLE verdict
        m = re.search(r"(?:UN)?SATISFIABLE", trace_text, re.IGNORECASE)
        if m:
            frac = m.start() / total_chars
            return (frac, "sat_verdict")

    # Fallback: entire trace is reasoning
    return (1.0, "none")


# ---------------------------------------------------------------------------
# Canonical output columns (shared by full and patterns-only modes)
# ---------------------------------------------------------------------------

ANALYSIS_COLS = [
    "problem_id", "run_idx",
    # Domain-specific difficulty columns
    "rating", "quintile", "cf_rating",  # Codeforces
    "math_level",  # MATH
    # Common structural columns
    "has_think", "has_think_tags", "reasoning_length", "answer_length",
    "trace_length_chars",
    "backtrack_count", "verify_count", "strategy_count",
    "hesitation_count", "problem_reread_count", "subgoal_count",
    "backtrack_rate", "verify_rate", "strategy_rate",
    "hesitation_rate", "problem_reread_rate", "subgoal_rate",
    "backtrack_first", "verify_first", "strategy_first",
    "hesitation_first", "problem_reread_first", "subgoal_first",
    "backtrack_last", "verify_last", "strategy_last",
    "hesitation_last", "problem_reread_last", "subgoal_last",
    "backtrack_pos_mean", "verify_pos_mean", "strategy_pos_mean",
    "hesitation_pos_mean", "problem_reread_pos_mean", "subgoal_pos_mean",
    "correct", "passed", "total", "error_type", "test_source",
    "mattr",
    # Reasoning boundary
    "reasoning_end_frac", "reasoning_boundary_type",
    # MATH-specific
    "predicted_answer",
    # SAT-specific
    "num_clauses", "sat_bin", "predicted_sat_label",
]


# ---------------------------------------------------------------------------
# Patterns-only mode
# ---------------------------------------------------------------------------

def run_patterns_only(n_workers: int) -> None:
    """Recompute only pattern + repetition columns in existing parquet.

    Loads ``cot_traces.jsonl`` to extract think text, runs pattern detection
    and repetition scoring in parallel, then rebuilds ``cot_analysis.parquet``
    with exactly the same schema as a fresh Stage 02 run.  Correctness, code
    extraction, and all other non-pattern columns are preserved from the
    existing file.
    """
    print("\n" + "=" * 70)
    print("  STAGE 02-P: Patterns-only recompute")
    print("=" * 70)
    t0 = time.time()

    if not COT_ANALYSIS_PATH.exists():
        print(f"ERROR: Existing parquet not found: {COT_ANALYSIS_PATH}")
        print("  Run full Stage 02 first, then use --patterns-only.")
        sys.exit(1)

    existing = pd.read_parquet(COT_ANALYSIS_PATH)
    print(f"  Loaded existing parquet: {len(existing)} rows, "
          f"{len(existing.columns)} columns")

    # Load traces for think-text extraction
    print("[02-P.1] Loading traces and parsing think blocks...")
    traces = []
    with open(TRACES_PATH) as f:
        for line in f:
            traces.append(json.loads(line))
    traces_df = pd.DataFrame(traces)

    if len(traces_df) != len(existing):
        print(f"ERROR: Row count mismatch: JSONL has {len(traces_df)} rows, "
              f"parquet has {len(existing)} rows")
        sys.exit(1)

    think_texts = []
    for idx, (_, row) in enumerate(traces_df.iterrows()):
        think, answer, _ = parse_think_response(row["trace"])
        think = think.strip()
        if think:
            think_texts.append(think)
        else:
            frac = existing.iloc[idx].get("reasoning_end_frac", 1.0)
            if pd.isna(frac):
                frac = 1.0
            end = int(len(row["trace"]) * float(frac))
            think_texts.append(row["trace"][:end])
    print(f"  Parsed {len(think_texts)} traces")

    # Run pattern detection + repetition (parallel)
    print(f"[02-P.2] Detecting structural patterns ({n_workers} workers)...")
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_worker_init) as pool:
        pattern_results = list(tqdm(
            pool.map(compute_patterns_worker, think_texts, chunksize=20),
            total=len(think_texts),
            desc="  Patterns + repetition",
        ))
    pattern_df = pd.DataFrame(pattern_results)
    print(f"  Computed {len(pattern_df.columns)} pattern/repetition columns")

    # Rebuild: drop ALL old columns that clash with new pattern_df,
    # then concat and filter through ANALYSIS_COLS for a clean schema.
    existing = existing.drop(
        columns=existing.columns.intersection(pattern_df.columns),
        errors="ignore",
    )
    merged = pd.concat([existing, pattern_df], axis=1)
    out_cols = [c for c in ANALYSIS_COLS if c in merged.columns]
    merged = merged[out_cols]

    _ensure_dir(COT_ANALYSIS_PATH)
    merged.to_parquet(COT_ANALYSIS_PATH, index=False)

    elapsed = time.time() - t0
    print(f"\n  Patterns-only complete in {_fmt_elapsed(elapsed)}")
    print(f"  -> {COT_ANALYSIS_PATH}")
    print(f"  Output: {len(merged)} rows, {len(out_cols)} columns")


# ---------------------------------------------------------------------------
# Stage 02: CoT structural analysis
# ---------------------------------------------------------------------------

def run_stage_02(n_workers: int, force: bool) -> None:
    """Stage 02: CoT structural analysis.

    Steps:
      1. Load traces and problems
      2. Parse think blocks
      3. Pattern detection + repetition (parallel)
      4. Code extraction
      5. Correctness evaluation (parallel)
      6. Save cot_analysis.parquet
    """
    print("\n" + "=" * 70)
    print("  STAGE 02: CoT Structural Analysis")
    print("=" * 70)
    t0 = time.time()

    # Cache check
    if not force and _cache_exists(COT_ANALYSIS_PATH):
        print(f"  SKIP (cached): {COT_ANALYSIS_PATH.name}")
        print("  Use --force to recompute.")
        return

    # ---- Step 1: Load data ----
    print("\n[02.1] Loading traces and problems...")
    problems = load_problems(PROBLEMS_PATH)

    traces = []
    with open(TRACES_PATH) as f:
        for line in f:
            traces.append(json.loads(line))
    traces_df = pd.DataFrame(traces)
    print(f"  Loaded {len(traces_df)} traces, {len(problems)} problems")

    # ---- Step 2: Parse think blocks ----
    print("[02.2] Parsing think blocks...")

    def _parse_trace(row):
        think, answer, _ = parse_think_response(row["trace"])
        return pd.Series({
            "think": think.strip(),
            "answer": answer.strip(),
            "reasoning_length": len(think),
            "answer_length": len(answer),
            "has_think": THINK_START in row["trace"],
        })

    parsed = traces_df.apply(_parse_trace, axis=1)
    traces_df = pd.concat([traces_df, parsed], axis=1)

    if "trace_length_chars" not in traces_df.columns:
        traces_df["trace_length_chars"] = traces_df["trace"].apply(len)
    if "has_think_tags" not in traces_df.columns:
        traces_df["has_think_tags"] = traces_df["trace"].apply(
            lambda t: THINK_START in t and THINK_END in t
        )

    # Merge with problem metadata (domain-aware)
    if DOMAIN == "math":
        meta_cols = {"level_int": "math_level"}
        meta_df = problems[["join_key", "level_int"]].rename(
            columns={"join_key": "problem_id", **meta_cols}
        )
        traces_df = traces_df.merge(meta_df, on="problem_id", how="left",
                                    suffixes=("", "_problem"))
    elif DOMAIN == "sat":
        meta_cols = {"num_clauses": "num_clauses", "num_clauses_bin": "sat_bin"}
        meta_keys = [k for k in meta_cols.keys() if k in problems.columns]
        rename_map = {"join_key": "problem_id"}
        rename_map.update({k: meta_cols[k] for k in meta_keys})
        meta_df = problems[["join_key"] + meta_keys].rename(columns=rename_map)
        traces_df = traces_df.merge(meta_df, on="problem_id", how="left",
                                    suffixes=("", "_problem"))
    else:
        traces_df = traces_df.merge(
            problems[["join_key", "quintile", "unnorm_rating"]].rename(
                columns={"join_key": "problem_id", "unnorm_rating": "cf_rating"}
            ),
            on="problem_id",
            how="left",
            suffixes=("", "_problem"),
        )
    print(f"  Parsed {len(traces_df)} traces")

    # Detect reasoning boundary for each trace
    traces_df[["reasoning_end_frac", "reasoning_boundary_type"]] = traces_df.apply(
        lambda row: _detect_reasoning_boundary(
            row["trace"], row["has_think"], DOMAIN
        ),
        axis=1, result_type="expand",
    )
    n_truncated = (traces_df["reasoning_end_frac"] < 1.0).sum()
    print(f"  Reasoning boundaries: {n_truncated}/{len(traces_df)} traces truncated")
    boundary_counts = traces_df["reasoning_boundary_type"].value_counts()
    for btype, count in boundary_counts.items():
        print(f"    {btype}: {count}")

    # For non-R1 models, set reasoning_length to the reasoning portion length
    # (before code fence / \boxed{} / XML tag) so it is consistent with R1.
    non_think = ~traces_df["has_think"]
    n_updated = non_think.sum()
    if n_updated > 0:
        traces_df.loc[non_think, "reasoning_length"] = (
            traces_df.loc[non_think, "trace_length_chars"]
            * traces_df.loc[non_think, "reasoning_end_frac"]
        ).astype(int)
        traces_df.loc[non_think, "answer_length"] = (
            traces_df.loc[non_think, "trace_length_chars"]
            - traces_df.loc[non_think, "reasoning_length"]
        ).astype(int)
        print(f"  Updated reasoning_length for {n_updated} non-R1 traces "
              f"(mean={traces_df.loc[non_think, 'reasoning_length'].mean():.0f} chars)")

    # ---- Step 3: Pattern detection + repetition (parallel) ----
    print(f"[02.3] Detecting structural patterns ({n_workers} workers)...")
    # Use think text for R1 models; truncate trace to reasoning boundary
    # for non-R1 models so patterns reflect reasoning phase only.
    think_texts = [
        think if think else trace[:int(len(trace) * frac)]
        for think, trace, frac in zip(
            traces_df["think"], traces_df["trace"],
            traces_df["reasoning_end_frac"],
        )
    ]

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
        pattern_results = list(tqdm(
            pool.map(compute_patterns_worker, think_texts, chunksize=20),
            total=len(think_texts),
            desc="  Patterns + repetition",
        ))

    pattern_df = pd.DataFrame(pattern_results)
    # Merge pattern columns (drop any pre-existing to avoid conflicts)
    overlap = pattern_df.columns.intersection(traces_df.columns)
    traces_df = pd.concat([
        traces_df.drop(columns=overlap, errors="ignore"),
        pattern_df,
    ], axis=1)
    print(f"  Detected {len(pattern_df.columns)} pattern/repetition columns")

    # ---- Step 4: Code extraction (Codeforces only) ----
    if DOMAIN not in ("math", "sat"):
        print("[02.4] Extracting Python code from traces...")
        traces_df["code"] = traces_df["answer"].apply(extract_python_code)
        no_code_mask = traces_df["code"].isna()
        traces_df.loc[no_code_mask, "code"] = (
            traces_df.loc[no_code_mask, "trace"].apply(extract_python_code)
        )
        has_code = traces_df["code"].notna().sum()
        print(f"  Code extracted: {has_code} / {len(traces_df)} "
              f"({100 * has_code / len(traces_df):.1f}%)")
    else:
        print(f"[02.4] Skipping code extraction ({DOMAIN.upper()} domain)")

    # ---- Step 5: Correctness evaluation ----
    if DOMAIN == "math":
        print(f"[02.5] Evaluating MATH correctness ({n_workers} workers)...")

        # Build MATH evaluation task list
        eval_args = []
        for _, row in traces_df.iterrows():
            problem_id = row["problem_id"]
            # Look up ground truth solution
            orig = problems[problems["join_key"] == problem_id]

            if len(orig) == 0:
                ground_truth = ""
                math_level = -1
            else:
                # Use the short 'answer' field (e.g. "36"), not the full
                # 'solution' write-up.  Fall back to extracting \boxed{}
                # from the solution text if 'answer' is missing.
                ground_truth = str(orig.iloc[0].get("answer", ""))
                if not ground_truth:
                    sol = str(orig.iloc[0].get("solution", ""))
                    ground_truth = extract_math_answer(sol) or sol
                math_level = int(orig.iloc[0].get("level_int", -1))

            eval_args.append((
                problem_id,
                int(row["run_idx"]),
                row.get("answer", ""),
                row.get("trace", ""),
                ground_truth,
                math_level,
            ))

        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
            eval_results = list(tqdm(
                pool.map(evaluate_math_trace_worker, eval_args, chunksize=1),
                total=len(eval_args),
                desc="  MATH correctness evaluation",
            ))

        eval_df = pd.DataFrame(eval_results)
        for col in ["correct", "error_type"]:
            traces_df[col] = eval_df[col].values
        # Add MATH-specific columns
        traces_df["predicted_answer"] = eval_df["predicted_answer"].values
        # Set Codeforces-style columns to defaults for unified schema
        traces_df["passed"] = traces_df["correct"].astype(int)
        traces_df["total"] = 1
        traces_df["test_source"] = "math_symbolic"

    elif DOMAIN == "sat":
        print(f"[02.5] Evaluating SAT correctness ({n_workers} workers)...")

        eval_args = []
        for _, row in traces_df.iterrows():
            problem_id = row["problem_id"]
            orig = problems[problems["join_key"] == problem_id]

            if len(orig) == 0:
                ground_truth_sat = False
                num_clauses = -1
            else:
                ground_truth_sat = bool(orig.iloc[0].get("satisfiable", False))
                num_clauses = int(orig.iloc[0].get("num_clauses", -1))

            eval_args.append((
                problem_id,
                int(row["run_idx"]),
                row.get("answer", ""),
                row.get("trace", ""),
                ground_truth_sat,
                num_clauses,
            ))

        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
            eval_results = list(tqdm(
                pool.map(evaluate_sat_trace_worker, eval_args, chunksize=1),
                total=len(eval_args),
                desc="  SAT correctness evaluation",
            ))

        eval_df = pd.DataFrame(eval_results)
        for col in ["correct", "error_type"]:
            traces_df[col] = eval_df[col].values
        traces_df["predicted_sat_label"] = eval_df["predicted_sat_label"].values
        # Set Codeforces-style columns to defaults for unified schema
        traces_df["passed"] = traces_df["correct"].astype(int)
        traces_df["total"] = 1
        traces_df["test_source"] = "sat_label"

    else:
        print(f"[02.5] Evaluating correctness ({n_workers} workers)...")

        # Build evaluation task list
        eval_args = []
        for _, row in traces_df.iterrows():
            problem_id = row["problem_id"]
            orig = problems[problems["join_key"] == problem_id]
            if len(orig) == 0:
                eval_args.append((
                    problem_id, int(row["run_idx"]),
                    None, "", "", [], "none",
                ))
                continue

            test_cases = orig.iloc[0].get("official_tests", [])
            test_source = "official_tests"
            if not _is_nonempty_sequence(test_cases):
                test_cases = orig.iloc[0].get("examples", [])
                test_source = "examples" if _is_nonempty_sequence(test_cases) else "none"

            # Convert test_cases to plain list of dicts for reliable pickling
            # (HuggingFace datasets may wrap them in numpy object arrays).
            tc_list = []
            if _is_nonempty_sequence(test_cases):
                for tc in test_cases:
                    if isinstance(tc, dict):
                        tc_list.append({
                            "input": str(tc.get("input", "")),
                            "output": str(tc.get("output", "")),
                        })

            eval_args.append((
                problem_id,
                int(row["run_idx"]),
                row.get("code", None),
                row.get("answer", ""),
                row.get("trace", ""),
                tc_list,
                test_source,
            ))

        # Each worker spawns subprocesses internally (I/O-bound waiting).
        # chunksize=1 gives optimal load balancing since tasks vary 1-60s.
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
            eval_results = list(tqdm(
                pool.map(evaluate_trace_worker, eval_args, chunksize=1),
                total=len(eval_args),
                desc="  Correctness evaluation",
            ))

        eval_df = pd.DataFrame(eval_results)
        for col in ["passed", "total", "correct", "error_type", "test_source"]:
            traces_df[col] = eval_df[col].values
        # Update code column (may have been extracted inside worker)
        traces_df["code"] = eval_df["code"].values

    # ---- Step 6: Save outputs ----
    print("[02.6] Saving outputs...")

    analysis_cols = [c for c in ANALYSIS_COLS if c in traces_df.columns]
    _ensure_dir(COT_ANALYSIS_PATH)
    traces_df[analysis_cols].to_parquet(COT_ANALYSIS_PATH, index=False)

    elapsed = time.time() - t0
    correct_rate = traces_df["correct"].mean()
    print(f"\n  Stage 02 complete in {_fmt_elapsed(elapsed)}")
    print(f"  -> {COT_ANALYSIS_PATH}")
    print(f"  Overall correctness: {traces_df['correct'].sum()}/{len(traces_df)} "
          f"({100 * correct_rate:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 02: CoT structural analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
             "(e.g. 'pipeline/math/deepseek-r1-7b' -> math).",
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
        "--patterns-only",
        action="store_true",
        help="Recompute only pattern detection and repetition metrics, "
             "patching them into the existing cot_analysis.parquet. "
             "Skips correctness evaluation and code extraction.",
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
    global DOMAIN, MODEL_NAME, TRACES_PATH, PROBLEMS_PATH
    global COT_ANALYSIS_PATH

    DOMAIN = args.domain
    cfg = load_config(args.config)
    MODEL_NAME        = cfg["model"]["name"]
    PROBLEMS_PATH     = cfg["paths"]["problems"]
    TRACES_PATH       = cfg["paths"]["pipeline"]["cot_traces"]
    COT_ANALYSIS_PATH = cfg["paths"]["analysis"]["cot_analysis"]

    mode = "patterns-only" if args.patterns_only else "full"

    print("=" * 70)
    print("  IRT Latent Difficulty — CoT Analysis (Stage 02)")
    print("=" * 70)
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Domain:   {DOMAIN}")
    print(f"  Config:   {args.config}")
    print(f"  Workers:  {args.workers}")
    print(f"  Mode:     {mode}")
    print(f"  Force:    {args.force}")
    print(f"  Traces:   {TRACES_PATH}")
    print()

    # Validate that required input files exist
    if not TRACES_PATH.exists():
        print(f"ERROR: Required input file not found: {TRACES_PATH}")
        sys.exit(1)
    if not args.patterns_only and not PROBLEMS_PATH.exists():
        print(f"ERROR: Required input file not found: {PROBLEMS_PATH}")
        sys.exit(1)

    t_total = time.time()
    if args.patterns_only:
        run_patterns_only(args.workers)
    else:
        run_stage_02(args.workers, args.force)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  Done in {_fmt_elapsed(elapsed)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
