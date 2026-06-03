#!/usr/bin/env python
"""CLI for API-based CoT trace generation.

Generates Chain-of-Thought traces via cloud LLM APIs for eval-only IRT
models.  Outputs ``cot_traces.jsonl`` compatible with the existing
analysis pipeline (Stages 02-04, pooled IRT).

Usage
-----
    # Generate configs only (no API calls)
    python scripts/run_api_generation.py --domain both --generate-configs-only

    # Run specific models for code domain
    python scripts/run_api_generation.py --domain code --models gpt-4o-mini deepseek-r1-api

    # Run all models for both domains
    python scripts/run_api_generation.py --domain both

    # Resume interrupted run (automatic)
    python scripts/run_api_generation.py --domain code --models gpt-4o

    # Dry run
    python scripts/run_api_generation.py --domain code --dry-run

    # Batch mode (50% cost savings for OpenAI/Anthropic)
    python scripts/run_api_generation.py --domain code --batch

    # Check batch status
    python scripts/run_api_generation.py --batch-status

    # Collect completed batch results
    python scripts/run_api_generation.py --domain code --batch-collect
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

import pandas as pd

# Allow running from project root: PYTHONPATH=. python scripts/run_api_generation.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api_generation import (
    API_MODELS,
    generate_eval_only_config,
    run_model_domain,
    validate_api_keys,
)
from src.api_batch import (
    print_batch_status,
    run_batch_pipeline,
    supports_batch,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT traces via cloud LLM APIs for eval-only IRT models"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["code", "math", "sat", "both"],
        default="both",
        help="Problem domain to generate for (default: both)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Model keys to run (default: all). Use --list-models to see options.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of independent traces per problem (default: 5)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override max output tokens per trace (default: per-model limit)",
    )
    _default_api_outputs = Path(os.environ.get("IRT_OUTPUTS_ROOT", "outputs")) / "api"
    parser.add_argument(
        "--output-base",
        type=Path,
        default=_default_api_outputs,
        help=(
            "Base directory for output files. Defaults to "
            "$IRT_OUTPUTS_ROOT/api (or ./outputs/api when the env var is unset)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override max concurrent requests per model (default: per-model setting)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making API calls",
    )
    parser.add_argument(
        "--generate-configs-only",
        action="store_true",
        help="Only generate YAML configs, no API calls",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available API models and exit",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch API for OpenAI/Anthropic models (50%% cost savings). "
             "Non-batch providers fall back to real-time.",
    )
    parser.add_argument(
        "--batch-collect",
        action="store_true",
        help="Collect results from previously submitted batches without polling.",
    )
    parser.add_argument(
        "--batch-status",
        action="store_true",
        help="Show status of all active/completed batches and exit.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between batch status polls (default: 60)",
    )
    parser.add_argument(
        "--litellm-model",
        type=str,
        default=None,
        help="Override the litellm model string for the selected model(s). "
             "Useful when a provider renames or deprecates a model ID.",
    )
    args = parser.parse_args()

    # --- Argument validation ---
    if args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    if args.poll_interval < 5:
        parser.error("--poll-interval must be >= 5")
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be >= 1")

    # --list-models
    if args.list_models:
        print(f"{'Model Key':<22} {'Provider':<12} {'Concurrency':<13} {'Reasoning':<10} {'Batch'}")
        print("-" * 70)
        for key, cfg in sorted(API_MODELS.items()):
            print(
                f"{key:<22} {cfg['provider']:<12} {cfg['max_concurrency']:<13} "
                f"{'yes' if cfg['is_reasoning'] else 'no':<10} "
                f"{'yes' if supports_batch(cfg['provider'], cfg['litellm_model']) else 'no'}"
            )
        return

    # --batch-status
    if args.batch_status:
        print_batch_status(args.output_base)
        return

    # Resolve models
    model_keys = args.models or sorted(API_MODELS.keys())
    unknown = [k for k in model_keys if k not in API_MODELS]
    if unknown:
        parser.error(
            f"Unknown model(s): {unknown}\n"
            f"Available: {sorted(API_MODELS.keys())}"
        )

    if args.litellm_model and len(model_keys) != 1:
        parser.error("--litellm-model requires exactly one model (use --models)")

    # Resolve domains
    domains = ["code", "math", "sat"] if args.domain == "both" else [args.domain]
    project_root = Path(__file__).resolve().parent.parent

    # --- Generate configs ---
    print(f"Generating eval-only configs for {len(model_keys)} models x {len(domains)} domains...")
    for model_key in model_keys:
        for domain in domains:
            config_path = generate_eval_only_config(
                model_key, domain, args.output_base, args.num_runs
            )
            print(f"  {config_path.relative_to(project_root)}")

    if args.generate_configs_only:
        print(f"\nDone. Generated {len(model_keys) * len(domains)} config files.")
        return

    # --- Dry run ---
    if args.dry_run:
        mode = "BATCH" if (args.batch or args.batch_collect) else "REAL-TIME"
        print(f"\n[DRY RUN] Would generate traces ({mode} mode):")
        for model_key in model_keys:
            cfg = dict(API_MODELS[model_key])
            if args.litellm_model:
                cfg["litellm_model"] = args.litellm_model
            for domain in domains:
                if args.batch and supports_batch(cfg["provider"], cfg["litellm_model"]):
                    tag = f"[batch:{cfg['provider']}]"
                else:
                    tag = "[real-time]"
                override = f" (litellm: {cfg['litellm_model']})" if args.litellm_model else ""
                print(f"  {model_key}/{domain}: 500 problems x {args.num_runs} runs {tag}{override}")
        providers = sorted({API_MODELS[k]["provider"] for k in model_keys})
        print(f"\nRequired API keys: {providers}")
        return

    # --- Validate API keys ---
    validate_api_keys(model_keys)

    # --- Load problems ---
    problems_cache = {}
    for domain in domains:
        if domain == "math":
            path = project_root / "data" / "datasets" / "selected_500_math.parquet"
        elif domain == "sat":
            path = project_root / "data" / "datasets" / "selected_500_sat.parquet"
        else:
            path = project_root / "data" / "datasets" / "selected_500.parquet"
        if not path.exists():
            print(f"ERROR: problem file not found: {path}", file=sys.stderr)
            sys.exit(1)
        problems_cache[domain] = pd.read_parquet(path)
        print(f"Loaded {len(problems_cache[domain])} {domain} problems from {path}")

    # --- Run generation ---
    use_batch = args.batch or args.batch_collect
    summaries = {}

    async def run_all():
        # --batch-collect or non-batch: sequential as before
        if not use_batch or args.batch_collect:
            for model_key in model_keys:
                model_config = dict(API_MODELS[model_key])
                if args.litellm_model:
                    model_config["litellm_model"] = args.litellm_model
                for domain in domains:
                    output_dir = args.output_base / domain / model_key
                    label = f"{model_key}/{domain}"

                    print(f"\n{'='*60}")
                    print(f"  {label}")
                    print(f"{'='*60}")

                    if use_batch and supports_batch(model_config["provider"], model_config["litellm_model"]):
                        summary = await run_batch_pipeline(
                            model_key=model_key,
                            model_config=model_config,
                            domain=domain,
                            problems=problems_cache[domain],
                            num_runs=args.num_runs,
                            output_dir=output_dir,
                            max_output_tokens=args.max_output_tokens,
                            base_seed=args.seed,
                            poll_interval=args.poll_interval,
                            collect_only=args.batch_collect,
                        )
                    else:
                        if args.batch_collect:
                            print(f"  Skipping {model_key} (no batch support)")
                            continue
                        summary = await run_model_domain(
                            model_key=model_key,
                            model_config=model_config,
                            domain=domain,
                            problems=problems_cache[domain],
                            num_runs=args.num_runs,
                            output_dir=output_dir,
                            max_output_tokens=args.max_output_tokens,
                            concurrency_override=args.concurrency,
                            base_seed=args.seed,
                        )
                    summaries[label] = summary
            return

        # --batch: submit-all-then-poll workflow
        pending = []  # (label, model_key, model_config, domain, output_dir)

        # --- Pass 1: submit all batches (fast, sequential) ---
        print(f"\n{'='*60}")
        print("  Pass 1: Submitting all batches")
        print(f"{'='*60}")

        for model_key in model_keys:
            model_config = dict(API_MODELS[model_key])
            if args.litellm_model:
                model_config["litellm_model"] = args.litellm_model
            for domain in domains:
                output_dir = args.output_base / domain / model_key
                label = f"{model_key}/{domain}"

                if supports_batch(model_config["provider"], model_config["litellm_model"]):
                    summary = await run_batch_pipeline(
                        model_key=model_key,
                        model_config=model_config,
                        domain=domain,
                        problems=problems_cache[domain],
                        num_runs=args.num_runs,
                        output_dir=output_dir,
                        max_output_tokens=args.max_output_tokens,
                        base_seed=args.seed,
                        poll_interval=args.poll_interval,
                        submit_only=True,
                    )
                    # Already complete — record and skip poll
                    if summary.get("completed", 0) >= len(problems_cache[domain]) * args.num_runs:
                        summaries[label] = summary
                    else:
                        pending.append((label, model_key, model_config, domain, output_dir))
                else:
                    # Non-batch model: run real-time immediately
                    summary = await run_model_domain(
                        model_key=model_key,
                        model_config=model_config,
                        domain=domain,
                        problems=problems_cache[domain],
                        num_runs=args.num_runs,
                        output_dir=output_dir,
                        max_output_tokens=args.max_output_tokens,
                        concurrency_override=args.concurrency,
                        base_seed=args.seed,
                    )
                    summaries[label] = summary

        if not pending:
            print("\nAll batches already complete.")
            return

        # --- Pass 2: poll + collect all pending batches concurrently ---
        print(f"\n{'='*60}")
        print(f"  Pass 2: Polling {len(pending)} batches concurrently")
        print(f"{'='*60}")

        async def poll_one(label, model_key, model_config, domain, output_dir):
            return await run_batch_pipeline(
                model_key=model_key,
                model_config=model_config,
                domain=domain,
                problems=problems_cache[domain],
                num_runs=args.num_runs,
                output_dir=output_dir,
                max_output_tokens=args.max_output_tokens,
                base_seed=args.seed,
                poll_interval=args.poll_interval,
            )

        results = await asyncio.gather(
            *(poll_one(*entry) for entry in pending),
            return_exceptions=True,
        )

        for (label, model_key_p, model_config_p, domain_p, output_dir_p), result in zip(pending, results):
            if isinstance(result, Exception):
                # Report actual request count from batch state, not hardcoded 1
                from src.api_batch import load_batch_state
                batch_state = load_batch_state(output_dir_p)
                num_errors = batch_state["num_requests"] if batch_state else 0
                print(f"  {label}: ERROR — {result}")
                summaries[label] = {
                    "new": 0, "errors": num_errors,
                    "input_tokens": 0, "output_tokens": 0,
                }
            else:
                summaries[label] = result

    asyncio.run(run_all())

    # --- Print summary ---
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Model/Domain':<35} {'New':>6} {'Errors':>7} {'In Tok':>10} {'Out Tok':>10}")
    print("-" * 70)

    total_new = 0
    total_errors = 0
    total_in = 0
    total_out = 0

    for label, s in sorted(summaries.items()):
        print(
            f"{label:<35} {s['new']:>6} {s['errors']:>7} "
            f"{s['input_tokens']:>10,} {s['output_tokens']:>10,}"
        )
        total_new += s["new"]
        total_errors += s["errors"]
        total_in += s["input_tokens"]
        total_out += s["output_tokens"]

    print("-" * 70)
    print(
        f"{'TOTAL':<35} {total_new:>6} {total_errors:>7} "
        f"{total_in:>10,} {total_out:>10,}"
    )


if __name__ == "__main__":
    main()
