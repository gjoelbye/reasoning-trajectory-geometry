"""Pooled cross-model IRT calibration (binomial Rasch 1PL).

Pools correctness results from multiple models and fits a binomial Rasch
(1PL) IRT model.  Each observation is (model, problem, n_correct, n_total),
aggregating runs into binomial counts.  This produces difficulty estimates
that are more reliable than per-model IRT while using a simple, clean model.

Usage
-----
    # Per-domain pooling (primary use case)
    python scripts/run_pooled_irt.py \\
        --auto-discover --domain codeforces --include-eval-only

    # MATH domain
    python scripts/run_pooled_irt.py --auto-discover --domain math --include-eval-only

    # Cross-domain pooling (puts code and math on a common scale)
    python scripts/run_pooled_irt.py --auto-discover --domain cross_domain

    # Explicit config list
    python scripts/run_pooled_irt.py \\
        --configs code/deepseek-r1-14b code/qwen-7b code/llama-8b \\
        --domain codeforces

    # Options
    python scripts/run_pooled_irt.py --auto-discover --domain codeforces --force
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_correctness_for_config(
    cfg: dict,
    model_name: str,
    domain_prefix: str = "",
) -> pd.DataFrame | None:
    """Load correctness results and aggregate to binomial counts.

    Returns a DataFrame with [model_id, item_id, n_correct, n_total, model_name],
    or None if the cot_analysis parquet does not exist.
    """
    cot_path = cfg["paths"]["analysis"]["cot_analysis"]
    if not cot_path.exists():
        return None

    df = pd.read_parquet(cot_path, columns=["problem_id", "run_idx", "correct"])

    # Aggregate per-run binary data to binomial counts per problem
    agg = df.groupby("problem_id").agg(
        n_correct=("correct", "sum"),
        n_total=("correct", "count"),
    ).reset_index()

    return pd.DataFrame({
        "model_id": model_name,
        "item_id": domain_prefix + agg["problem_id"].astype(str),
        "n_correct": agg["n_correct"].astype(int),
        "n_total": agg["n_total"].astype(int),
        "model_name": model_name,
    })


def build_pooled_response_matrix(
    configs: dict[str, dict],
    domain: str,
) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Build the pooled response matrix from all available model configs.

    Returns (pooled_df, loaded_config_names, skipped_config_names).
    """
    all_responses = []
    loaded = []
    skipped = []

    for config_name, cfg in configs.items():
        model_name = cfg["model"]["name"]

        # Determine domain prefix for cross-domain mode
        if domain == "cross_domain":
            if "code/" in config_name:
                prefix = "cf_"
            elif "sat/" in config_name:
                prefix = "sat_"
            else:
                prefix = "math_"
        else:
            prefix = ""

        resp = load_correctness_for_config(cfg, model_name, prefix)
        if resp is not None:
            # For cross-domain, disambiguate model_id from same model across domains
            if domain == "cross_domain":
                if "code/" in config_name:
                    domain_tag = "code"
                elif "sat/" in config_name:
                    domain_tag = "sat"
                else:
                    domain_tag = "math"
                resp["model_id"] = domain_tag + "_" + resp["model_id"]

            all_responses.append(resp)
            loaded.append(config_name)
            print(f"  {config_name} ({model_name}): "
                  f"{resp['item_id'].nunique()} items, "
                  f"{resp['n_total'].sum()} total trials")
        else:
            skipped.append(config_name)
            print(f"  {config_name}: correctness not found, SKIPPING")

    if not all_responses:
        return None, loaded, skipped

    pooled = pd.concat(all_responses, ignore_index=True)
    return pooled, loaded, skipped


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_uninformative_items(
    response_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter items with 0% or 100% pass rate across all models.

    Returns (filtered_response_df, item_stats_df).
    """
    item_stats = response_df.groupby("item_id").agg(
        n_correct_total=("n_correct", "sum"),
        n_trials_total=("n_total", "sum"),
    )
    item_stats["pass_rate"] = (
        item_stats["n_correct_total"] / item_stats["n_trials_total"]
    )

    # How many distinct models solved each item (at least once)
    model_correct = response_df[response_df["n_correct"] > 0]
    n_models_solved = model_correct.groupby("item_id")["model_name"].nunique()
    item_stats["n_models_solved_by"] = n_models_solved.reindex(
        item_stats.index, fill_value=0
    )

    informative = item_stats[
        (item_stats["pass_rate"] > 0.0) & (item_stats["pass_rate"] < 1.0)
    ].index.tolist()

    filtered = response_df[response_df["item_id"].isin(informative)].copy()
    return filtered, item_stats


# ---------------------------------------------------------------------------
# IRT fitting
# ---------------------------------------------------------------------------

def fit_pooled_irt(
    response_df_filtered: pd.DataFrame,
    item_stats: pd.DataFrame,
    max_epochs: int = 5000,
    patience: int = 300,
) -> dict:
    """Fit binomial Rasch (1PL) IRT to the pooled response matrix."""
    from src.irt import fit_irt_model, extract_difficulties, extract_abilities

    n_models = response_df_filtered["model_id"].nunique()
    n_items = response_df_filtered["item_id"].nunique()

    print(f"\n  Fitting binomial Rasch 1PL ({n_models} models, {n_items} items)...")
    params = fit_irt_model(
        response_df_filtered[["model_id", "item_id", "n_correct", "n_total"]],
        model_type="1pl",
        num_epochs=max_epochs,
        patience=patience,
        verbose=True,
    )

    difficulties_informative = extract_difficulties(params)
    abilities = extract_abilities(params)

    # Boundary difficulties for uninformative items
    max_diff = float(difficulties_informative["difficulty"].max())
    min_diff = float(difficulties_informative["difficulty"].min())

    all_fail_items = item_stats[item_stats["pass_rate"] == 0.0].index.tolist()
    all_pass_items = item_stats[item_stats["pass_rate"] == 1.0].index.tolist()

    boundary_rows = []
    for item_id in all_fail_items:
        boundary_rows.append({
            "item_id": item_id, "difficulty": max_diff + 0.5,
            "discrimination": 1.0,
        })
    for item_id in all_pass_items:
        boundary_rows.append({
            "item_id": item_id, "difficulty": min_diff - 0.5,
            "discrimination": 1.0,
        })

    difficulties = pd.concat([
        difficulties_informative,
        pd.DataFrame(boundary_rows),
    ], ignore_index=True)

    print(f"  Total: {len(difficulties)} items "
          f"({len(difficulties_informative)} fitted + {len(boundary_rows)} boundary)")

    return {
        "params": params,
        "preferred": "binomial_1pl",
        "difficulties": difficulties,
        "difficulties_informative": difficulties_informative,
        "abilities": abilities,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_against_native(
    difficulties: pd.DataFrame,
    domain: str,
    problems_path: Path,
) -> dict:
    """Validate IRT difficulties against native ratings (CF), levels (MATH), or clause count (SAT)."""
    from src.irt import compare_irt_to_native_rating, compare_irt_to_math_levels

    problems = pd.read_parquet(problems_path)

    if domain == "math":
        meta = problems[["join_key", "level_int"]].rename(
            columns={"join_key": "item_id"})
        return compare_irt_to_math_levels(difficulties, meta)
    elif domain == "sat":
        from src.irt import compare_irt_to_clause_count
        meta = problems[["join_key", "num_clauses"]].rename(
            columns={"join_key": "item_id"})
        return compare_irt_to_clause_count(difficulties, meta)
    else:
        meta = problems[["join_key", "unnorm_rating"]].rename(
            columns={"join_key": "item_id"})
        return compare_irt_to_native_rating(difficulties, meta)


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    fit_result: dict,
    pooled_resp: pd.DataFrame,
    item_stats: pd.DataFrame,
    validation: dict,
    configs: dict[str, dict],
    loaded: list[str],
    skipped: list[str],
    args: argparse.Namespace,
) -> None:
    """Save all pooled IRT outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params = fit_result["params"]
    abilities_df = fit_result["abilities"]

    # Per-model ability summary (one row per model, no run deviations)
    ability_by_model = {}
    for _, row in abilities_df.iterrows():
        ability_by_model[str(row["model_id"])] = float(row["ability"])

    # 1. pooled_irt_params.json
    irt_output = {
        "model_type": fit_result["preferred"],
        "domain": args.domain,
        "models_included": [configs[n]["model"]["name"] for n in loaded],
        "configs_used": loaded,
        "configs_skipped": skipped,
        "n_models": len(loaded),
        "n_items_total": pooled_resp["item_id"].nunique(),
        "n_items_informative": len(fit_result["difficulties_informative"]),
        "n_items_boundary": (
            len(fit_result["difficulties"])
            - len(fit_result["difficulties_informative"])
        ),
        "n_observations": len(pooled_resp),
        "n_total_trials": int(pooled_resp["n_total"].sum()),
        "difficulties": fit_result["difficulties"].to_dict(orient="records"),
        "difficulties_informative_only": (
            fit_result["difficulties_informative"].to_dict(orient="records")
        ),
        "abilities": abilities_df.to_dict(orient="records"),
        "final_epoch": params["final_epoch"],
        "final_loss": params["final_loss"],
        "model_ability": ability_by_model,
        "validation": validation,
    }
    params_path = output_dir / "pooled_irt_params.json"
    with open(params_path, "w") as f:
        json.dump(irt_output, f, indent=2)
    print(f"  -> {params_path}")

    # 2. pooled_difficulties.parquet
    diffs = fit_result["difficulties"].copy()
    diffs = diffs.merge(
        item_stats[["pass_rate", "n_models_solved_by"]].reset_index(),
        left_on="item_id", right_on="item_id", how="left",
    )
    diffs["pass_rate_pooled"] = diffs["pass_rate"]
    diffs_path = output_dir / "pooled_difficulties.parquet"
    diffs.to_parquet(diffs_path, index=False)
    print(f"  -> {diffs_path}")

    # 3. pooled_abilities.parquet
    abilities_path = output_dir / "pooled_abilities.parquet"
    abilities_df.to_parquet(abilities_path, index=False)
    print(f"  -> {abilities_path}")

    # 4. pooled_response_matrix.parquet
    resp_path = output_dir / "pooled_response_matrix.parquet"
    pooled_resp.to_parquet(resp_path, index=False)
    print(f"  -> {resp_path}")

    # 5. Augmented problem files
    if args.domain == "cross_domain":
        domain_tags_save = ["codeforces", "math"]
        if any("sat/" in n for n in loaded):
            domain_tags_save.append("sat")
        for domain_tag in domain_tags_save:
            if domain_tag == "codeforces":
                prefix = "cf_"
            elif domain_tag == "sat":
                prefix = "sat_"
            else:
                prefix = "math_"
            domain_cfg = next(
                (cfg for n, cfg in configs.items()
                 if ("code/" in n if domain_tag == "codeforces"
                     else "sat/" in n if domain_tag == "sat"
                     else "math/" in n)),
                None,
            )
            if domain_cfg is None:
                continue
            problems = pd.read_parquet(domain_cfg["paths"]["problems"])
            domain_diffs = diffs[diffs["item_id"].str.startswith(prefix)].copy()
            domain_diffs["item_id_raw"] = (
                domain_diffs["item_id"].str[len(prefix):]
            )
            augmented = problems.merge(
                domain_diffs[["item_id_raw", "difficulty", "discrimination"]]
                .rename(columns={
                    "item_id_raw": "join_key",
                    "difficulty": "pooled_difficulty",
                    "discrimination": "pooled_discrimination",
                }),
                on="join_key", how="left",
            )
            if domain_tag == "codeforces":
                suffix = "code"
            elif domain_tag == "sat":
                suffix = "sat"
            else:
                suffix = "math"
            aug_path = output_dir / f"pooled_augmented_500_{suffix}.parquet"
            augmented.to_parquet(aug_path, index=False)
            print(f"  -> {aug_path}")
    else:
        first_cfg = next(iter(configs.values()))
        problems = pd.read_parquet(first_cfg["paths"]["problems"])
        augmented = problems.merge(
            diffs[["item_id", "difficulty", "discrimination"]].rename(
                columns={
                    "item_id": "join_key",
                    "difficulty": "pooled_difficulty",
                    "discrimination": "pooled_discrimination",
                }
            ),
            on="join_key", how="left",
        )
        aug_path = output_dir / "pooled_augmented_500.parquet"
        augmented.to_parquet(aug_path, index=False)
        print(f"  -> {aug_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pooled cross-model IRT calibration (binomial Rasch).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="List of model config names "
             "(e.g. 'code/deepseek-r1-14b code/qwen-7b'). "
             "Not required if --auto-discover is used.",
    )
    parser.add_argument(
        "--auto-discover", action="store_true",
        help="Auto-discover all available configs for the specified domain. "
             "Overrides --configs.",
    )
    parser.add_argument(
        "--include-eval-only", action="store_true",
        help="When using --auto-discover, also include configs from "
             "configs/local/ and configs/api/. Default is pipeline configs only.",
    )
    parser.add_argument(
        "--domain", type=str, required=True,
        choices=["codeforces", "math", "sat", "cross_domain"],
        help="Output mode: 'codeforces' or 'math' for single-domain, "
             "'cross_domain' for joint calibration.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if cached outputs exist.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5000,
        help="Max IRT optimization epochs (default: 5000).",
    )
    parser.add_argument(
        "--patience", type=int, default=300,
        help="Early stopping patience (default: 300).",
    )
    args = parser.parse_args()

    if not args.auto_discover and not args.configs:
        parser.error("Either --configs or --auto-discover must be specified.")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _auto_discover_configs(
    domain: str,
    include_eval_only: bool = False,
) -> list[str]:
    """Auto-discover all config names matching the given domain.

    Parameters
    ----------
    domain : str
        ``"codeforces"``, ``"math"``, or ``"cross_domain"``.
    include_eval_only : bool
        Also include configs from ``configs/local/`` and ``configs/api/``.

    Returns
    -------
    list of config name strings (e.g. ``["code/deepseek-r1-14b", ...]``).
    """
    from src.config import list_configs

    all_names = list_configs(include_eval_only=include_eval_only)
    results = []

    for name in all_names:
        if domain == "cross_domain":
            # Include code/, math/, and sat/ configs
            if name.startswith(("pipeline/code/", "pipeline/math/", "pipeline/sat/")) or (
                include_eval_only and name.startswith(
                    ("local/code/", "local/math/", "local/sat/",
                     "api/code/", "api/math/", "api/sat/")
                )
            ):
                results.append(name)
        elif domain == "codeforces":
            if name.startswith(("pipeline/code/", "local/code/", "api/code/")):
                if not include_eval_only and not name.startswith("pipeline/"):
                    continue
                results.append(name)
        elif domain == "math":
            if name.startswith(("pipeline/math/", "local/math/", "api/math/")):
                if not include_eval_only and not name.startswith("pipeline/"):
                    continue
                results.append(name)
        elif domain == "sat":
            if name.startswith(("pipeline/sat/", "local/sat/", "api/sat/")):
                if not include_eval_only and not name.startswith("pipeline/"):
                    continue
                results.append(name)

    return sorted(results)


def main():
    args = parse_args()

    from src.config import load_config

    # Determine output directory
    if args.domain == "cross_domain":
        output_dir = PROJECT_ROOT / "data" / "results" / "pooled_irt" / "cross_domain"
    elif args.domain == "math":
        output_dir = PROJECT_ROOT / "data" / "results" / "pooled_irt" / "math"
    elif args.domain == "sat":
        output_dir = PROJECT_ROOT / "data" / "results" / "pooled_irt" / "sat"
    else:
        output_dir = PROJECT_ROOT / "data" / "results" / "pooled_irt" / "code"

    params_path = output_dir / "pooled_irt_params.json"

    # Resolve config list
    if args.auto_discover:
        config_names = _auto_discover_configs(
            args.domain, include_eval_only=args.include_eval_only,
        )
        if not config_names:
            print("ERROR: No configs found for auto-discovery.")
            sys.exit(1)
    else:
        config_names = args.configs

    print("=" * 70)
    print("  Pooled Cross-Model IRT Calibration (Binomial Rasch)")
    print("=" * 70)
    print(f"  Domain:         {args.domain}")
    print(f"  Auto-discover:  {args.auto_discover}")
    if args.auto_discover:
        print(f"  Incl eval-only: {args.include_eval_only}")
    print(f"  Configs:        {', '.join(config_names)}")
    print(f"  N configs:      {len(config_names)}")
    print(f"  Output:         {output_dir}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Patience:       {args.patience}")
    print(f"  Force:          {args.force}")
    print()

    # Cache check
    if not args.force and params_path.exists():
        print(f"  SKIP (cached): {params_path}")
        print("  Use --force to recompute.")
        return

    t0 = time.time()

    # ---- Step 1: Load configs ----
    print("[1] Loading model configs...")
    configs = {}
    for name in config_names:
        try:
            cfg = load_config(name)
            configs[name] = cfg
        except FileNotFoundError:
            print(f"  WARNING: Config {name} not found, skipping")

    if not configs:
        print("ERROR: No valid configs found.")
        sys.exit(1)

    # ---- Step 2: Build pooled response matrix ----
    print(f"\n[2] Building pooled response matrix...")
    pooled_resp, loaded, skipped = build_pooled_response_matrix(
        configs, args.domain,
    )

    if pooled_resp is None or not loaded:
        print("ERROR: No correctness results found for any model.")
        print("  Run 'python scripts/run_cot_analysis.py' for each model first.")
        sys.exit(1)

    n_models = pooled_resp["model_id"].nunique()
    n_items = pooled_resp["item_id"].nunique()
    n_obs = len(pooled_resp)
    n_trials = int(pooled_resp["n_total"].sum())
    print(f"\n  Pooled: {n_models} models, {n_items} items, "
          f"{n_obs} observations ({n_trials} total trials)")
    print(f"  Models loaded: {len(loaded)}, skipped: {len(skipped)}")

    # ---- Step 3: Filter uninformative items ----
    print(f"\n[3] Filtering uninformative items...")
    filtered, item_stats = filter_uninformative_items(pooled_resp)

    all_pass = int((item_stats["pass_rate"] == 1.0).sum())
    all_fail = int((item_stats["pass_rate"] == 0.0).sum())
    n_informative = n_items - all_pass - all_fail
    print(f"  Informative items: {n_informative} "
          f"(always correct: {all_pass}, always wrong: {all_fail})")

    if n_informative < 10:
        print("ERROR: Not enough informative items for IRT calibration.")
        sys.exit(1)

    # ---- Step 4: Fit IRT ----
    print(f"\n[4] Fitting binomial Rasch 1PL IRT...")
    fit_result = fit_pooled_irt(
        filtered, item_stats,
        max_epochs=args.epochs, patience=args.patience,
    )

    # ---- Step 5: Validate ----
    print(f"\n[5] Validating against native ratings/levels...")
    validation = {}

    if args.domain == "cross_domain":
        # Validate each domain separately
        domain_tags = ["codeforces", "math"]
        # Include SAT if any sat configs were loaded
        if any("sat/" in n for n in loaded):
            domain_tags.append("sat")
        for domain_tag in domain_tags:
            if domain_tag == "codeforces":
                prefix = "cf_"
            elif domain_tag == "sat":
                prefix = "sat_"
            else:
                prefix = "math_"
            domain_diffs = fit_result["difficulties"][
                fit_result["difficulties"]["item_id"].str.startswith(prefix)
            ].copy()
            domain_diffs["item_id"] = domain_diffs["item_id"].str[len(prefix):]
            domain_cfg = next(
                (cfg for n, cfg in configs.items()
                 if ("code/" in n if domain_tag == "codeforces"
                     else "sat/" in n if domain_tag == "sat"
                     else "math/" in n)),
                None,
            )
            if domain_cfg:
                v = validate_against_native(
                    domain_diffs[["item_id", "difficulty", "discrimination"]],
                    domain_tag,
                    domain_cfg["paths"]["problems"],
                )
                validation[domain_tag] = v
                if domain_tag == "codeforces":
                    print(f"  CF: Pearson={v['pearson_r']:.4f}, "
                          f"Spearman={v['spearman_rho']:.4f} (n={v['n']})")
                elif domain_tag == "sat":
                    print(f"  SAT: Spearman={v['spearman_rho']:.4f} "
                          f"(n={v['n']})")
                else:
                    print(f"  MATH: Spearman={v['spearman_rho']:.4f} "
                          f"(n={v['n']})")
    elif args.domain == "sat":
        first_cfg = next(iter(configs.values()))
        from src.irt import compare_irt_to_clause_count
        sat_problems = pd.read_parquet(first_cfg["paths"]["problems"])
        meta = sat_problems[["join_key", "num_clauses"]].rename(
            columns={"join_key": "item_id"})
        validation = compare_irt_to_clause_count(
            fit_result["difficulties"], meta,
        )
        print(f"  Spearman={validation['spearman_rho']:.4f} "
              f"(n={validation['n']})")
    else:
        first_cfg = next(iter(configs.values()))
        validation = validate_against_native(
            fit_result["difficulties"],
            args.domain,
            first_cfg["paths"]["problems"],
        )
        if args.domain == "math":
            print(f"  Spearman={validation['spearman_rho']:.4f} "
                  f"(n={validation['n']})")
        else:
            # Also compute informative-only correlations
            informative_items = item_stats[
                (item_stats["pass_rate"] > 0.0) & (item_stats["pass_rate"] < 1.0)
            ].index.tolist()
            info_diffs = fit_result["difficulties"][
                fit_result["difficulties"]["item_id"].isin(informative_items)
            ]
            info_val = validate_against_native(
                info_diffs, args.domain, first_cfg["paths"]["problems"],
            )
            print(f"  All items:         Pearson={validation['pearson_r']:.4f}, "
                  f"Spearman={validation['spearman_rho']:.4f} (n={validation['n']})")
            print(f"  Informative only:  Pearson={info_val['pearson_r']:.4f}, "
                  f"Spearman={info_val['spearman_rho']:.4f} (n={info_val['n']})")
            validation["pearson_r_informative"] = info_val["pearson_r"]
            validation["spearman_rho_informative"] = info_val["spearman_rho"]
            validation["n_informative"] = info_val["n"]

    # ---- Step 7: Save outputs ----
    print(f"\n[7] Saving outputs...")
    save_outputs(
        output_dir, fit_result, pooled_resp, item_stats,
        validation, configs, loaded, skipped, args,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Pooled IRT calibration complete in {_fmt_elapsed(elapsed)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
