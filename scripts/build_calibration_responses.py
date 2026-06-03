"""Build ``data/calibration/responses_{code,math,sat}.parquet``.

Reads per-model ``cot_analysis.parquet`` files at
``$IRT_RESULTS_ROOT/<domain>/<model>/cot_analysis/cot_analysis.parquet`` and
aggregates correctness to the IRT response matrix used by ``src.irt``.

Usage
-----
    python scripts/build_calibration_responses.py
    python scripts/build_calibration_responses.py --domains code math
    IRT_RESULTS_ROOT=/path/to/results python scripts/build_calibration_responses.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

HIDDEN_STATE_MODELS = [
    "deepseek-r1-7b", "deepseek-r1-14b", "deepseek-r1-32b",
    "r1-distill-llama-8b", "qwq-32b", "phi-4-reasoning",
    "qwen-7b", "qwen-14b", "qwen-32b", "llama-8b", "phi-4",
]

CALIBRATION_MODELS = [
    "gemma-9b", "phi-3.5", "mistral-7b", "qwen-math-7b",
    "deepseek-7b-chat", "olmo-7b", "qwen2-7b", "zephyr-7b",
    "claude-haiku-4.5", "claude-sonnet-4", "deepseek-v3",
    "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro",
    "gemma-3-27b", "gpt-4o", "gpt-4o-mini", "llama-3.3-70b",
    "mistral-small-24b", "o4-mini", "qwen-2.5-72b",
]

ALL_32 = HIDDEN_STATE_MODELS + CALIBRATION_MODELS


def _resolve(domain: str, model: str, results_roots: list[Path]) -> Path | None:
    """Return the first existing cot_analysis.parquet across the candidate roots."""
    for root in results_roots:
        p = root / domain / model / "cot_analysis" / "cot_analysis.parquet"
        if p.exists():
            return p
    return None


def build(domain: str, results_roots: list[Path]) -> pd.DataFrame:
    """Aggregate per-model `cot_analysis.parquet` into one response matrix.

    `results_roots` is searched in order; the first hit wins per model.
    """
    rows = []
    missing = []
    for model in ALL_32:
        path = _resolve(domain, model, results_roots)
        if path is None:
            missing.append(model)
            continue
        df = pd.read_parquet(path, columns=["problem_id", "correct"])
        agg = (
            df.assign(correct=df["correct"].astype("int64"))
            .groupby("problem_id", as_index=False)
            .agg(n_correct=("correct", "sum"), n_total=("correct", "count"))
        )
        agg.insert(0, "model_id", model)
        rows.append(agg.rename(columns={"problem_id": "item_id"}))
    if missing:
        print(f"[{domain}] missing cot_analysis for {len(missing)} model(s): {', '.join(missing)}")
    out = pd.concat(rows, ignore_index=True)
    out["n_correct"] = out["n_correct"].astype("int32")
    out["n_total"] = out["n_total"].astype("int32")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=["code", "math", "sat"],
                    choices=["code", "math", "sat"])
    ap.add_argument("--results-root", type=Path, action="append",
                    default=None,
                    help="Per-model analysis root. Can be passed multiple times; "
                         "the first matching path wins per model. Default: "
                         "$IRT_RESULTS_ROOT or data/results.")
    ap.add_argument("--out-dir", type=Path, default=Path("data/calibration"),
                    help="Output directory (default: data/calibration)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    roots = args.results_root or [Path(os.environ.get("IRT_RESULTS_ROOT", "data/results"))]
    for dom in args.domains:
        df = build(dom, roots)
        out_path = args.out_dir / f"responses_{dom}.parquet"
        df.to_parquet(out_path, index=False)
        size_kb = out_path.stat().st_size // 1024
        print(
            f"wrote {out_path}: {len(df):,} rows, "
            f"{df['model_id'].nunique()} models, "
            f"{df['item_id'].nunique()} items, {size_kb} KB"
        )


if __name__ == "__main__":
    main()
