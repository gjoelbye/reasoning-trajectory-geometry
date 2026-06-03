"""API-based CoT generation for eval-only IRT models.

Calls cloud LLM APIs (OpenAI, Anthropic, Google, DeepSeek, Together AI) via
``litellm`` to generate Chain-of-Thought traces.  Outputs ``cot_traces.jsonl``
in the exact same schema as ``run_pipeline.py`` so the existing analysis
pipeline (Stages 02-04, pooled IRT) works unchanged.

Usage
-----
    from src.api_generation import run_model_domain, generate_eval_only_config

    # Generate configs only
    generate_eval_only_config("gpt-4o-mini", "code", output_base)

    # Run generation
    import asyncio
    asyncio.run(run_model_domain("gpt-4o-mini", API_MODELS["gpt-4o-mini"],
                                 "code", problems_df, num_runs=5, output_dir=...))
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

import litellm
from src.config import PROJECT_ROOT
from src.models import format_prompt, format_math_generation_prompt, format_sat_generation_prompt

litellm.suppress_debug_info = True
litellm.drop_params = True

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

API_MODELS: Dict[str, Dict[str, Any]] = {
    # OpenAI (env: OPENAI_API_KEY)
    "gpt-4o": {
        "litellm_model": "gpt-4o",
        "provider": "openai",
        "max_concurrency": 5,
        "is_reasoning": False,
        "max_output_tokens": 16384,
    },
    "gpt-4o-mini": {
        "litellm_model": "gpt-4o-mini",
        "provider": "openai",
        "max_concurrency": 10,
        "is_reasoning": False,
        "max_output_tokens": 16384,
    },
    "o4-mini": {
        "litellm_model": "o4-mini",
        "provider": "openai",
        "max_concurrency": 5,
        "is_reasoning": True,
        "max_output_tokens": 32768,
    },
    # Anthropic (env: ANTHROPIC_API_KEY)
    "claude-haiku-4.5": {
        "litellm_model": "anthropic/claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "max_concurrency": 5,
        "is_reasoning": False,
        "max_output_tokens": 32768,
    },
    "claude-sonnet-4": {
        "litellm_model": "anthropic/claude-sonnet-4-20250514",
        "provider": "anthropic",
        "max_concurrency": 3,
        "is_reasoning": False,
        "max_output_tokens": 32768,
    },
    # Google (env: GEMINI_API_KEY)
    "gemini-2.5-flash-lite": {
        "litellm_model": "gemini/gemini-2.5-flash-lite",
        "provider": "google",
        "max_concurrency": 5,
        "is_reasoning": False,
        "max_output_tokens": 32768,
    },
    "gemini-2.5-flash": {
        "litellm_model": "gemini/gemini-2.5-flash",
        "provider": "google",
        "max_concurrency": 5,
        "is_reasoning": True,
        "max_output_tokens": 32768,
    },
    "gemini-2.5-pro": {
        "litellm_model": "gemini/gemini-2.5-pro",
        "provider": "google",
        "max_concurrency": 3,
        "is_reasoning": True,
        "max_output_tokens": 32768,
    },
    # DeepSeek via Together AI (env: TOGETHER_API_KEY)
    "deepseek-v3": {
        "litellm_model": "together_ai/deepseek-ai/DeepSeek-V3.1",
        "provider": "together",
        "max_concurrency": 5,
        "is_reasoning": False,
        "max_output_tokens": 8192,
    },
    # Together AI (env: TOGETHER_API_KEY)
    "llama-3.3-70b": {
        "litellm_model": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "provider": "together",
        "max_concurrency": 5,
        "is_reasoning": False,
        "max_output_tokens": 16384,
    },
    "qwen-2.5-72b": {
        "litellm_model": "openrouter/qwen/qwen-2.5-72b-instruct",
        "provider": "openrouter",
        "max_concurrency": 5,
        "is_reasoning": False,
        "max_output_tokens": 16384,
    },
    "mistral-small-24b": {
        "litellm_model": "together_ai/mistralai/Mistral-Small-24B-Instruct-2501",
        "provider": "together",
        "max_concurrency": 5,
        "is_reasoning": False,
        "max_output_tokens": 16384,
    },
    "gemma-3-27b": {
        "litellm_model": "gemini/gemma-3-27b-it",
        "provider": "google",
        "max_concurrency": 5,
        "is_reasoning": False,
        "max_output_tokens": 32768,
    },
}

ENV_KEYS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_api_keys(model_keys: List[str]) -> None:
    """Raise ``EnvironmentError`` if required API keys are missing."""
    providers_needed = {API_MODELS[k]["provider"] for k in model_keys}
    missing = []
    for provider in sorted(providers_needed):
        env_var = ENV_KEYS[provider]
        if not os.environ.get(env_var):
            missing.append(f"  {env_var} (for {provider})")
    if missing:
        raise EnvironmentError(
            "Missing API keys:\n" + "\n".join(missing)
            + "\nSet them as environment variables before running."
        )


def build_user_message(row, domain: str) -> str:
    """Build the user-facing prompt from a problem row."""
    if domain == "math":
        return format_math_generation_prompt(row["formatted_prompt"])
    elif domain == "sat":
        return format_sat_generation_prompt(row["formatted_prompt"])
    return format_prompt(row["formatted_prompt"])


def derive_seed(base_seed: int, problem_id: str, run_idx: int) -> int:
    """Deterministic per-trace seed, matching ``run_pipeline.py``."""
    hash_input = f"{base_seed}:{problem_id}:{run_idx}"
    return int.from_bytes(hash_input.encode(), byteorder="big") % (2**32)


def _load_completed(jsonl_path: Path) -> set:
    """Load already-completed ``(problem_id, run_idx)`` pairs.

    Entries with empty traces are excluded so they get retried on resume.
    Malformed JSON lines are skipped with a warning.
    """
    done = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "%s:%d: skipping malformed JSON line", jsonl_path, lineno
                    )
                    continue
                if r.get("trace"):
                    done.add((r["problem_id"], r["run_idx"]))
    return done


def _purge_failed_traces(jsonl_path: Path) -> int:
    """Remove entries with empty traces from *jsonl_path*.

    Malformed JSON lines are also removed.  The rewrite is atomic
    (write to tempfile + ``os.replace``) to avoid data loss on crash.

    Returns the number of entries removed.
    """
    if not jsonl_path.exists():
        return 0
    kept = []
    removed = 0
    with open(jsonl_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "%s:%d: removing malformed JSON line", jsonl_path, lineno
                )
                removed += 1
                continue
            if r.get("trace"):
                kept.append(line)
            else:
                removed += 1
    if removed > 0:
        fd, tmp_path = tempfile.mkstemp(
            dir=jsonl_path.parent, prefix=".purge_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                for line in kept:
                    f.write(line + "\n")
            os.replace(tmp_path, jsonl_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    return removed


def _append_jsonl(record: dict, path: Path) -> None:
    """Append a single JSON record to a file."""
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Single trace generation
# ---------------------------------------------------------------------------

async def generate_single_trace(
    model_key: str,
    model_config: Dict[str, Any],
    user_message: str,
    problem_id: str,
    run_idx: int,
    rating: float,
    seed: int,
    semaphore: asyncio.Semaphore,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Call the LLM API for a single trace with retry + rate limiting.

    Returns a dict matching the ``cot_traces.jsonl`` schema plus
    ``_input_tokens`` and ``_output_tokens`` for cost tracking.
    """

    messages = [{"role": "user", "content": user_message}]

    # Use per-model limit, optionally capped by CLI override
    model_max = model_config["max_output_tokens"]
    effective_max = min(max_output_tokens, model_max) if max_output_tokens else model_max

    # Build kwargs
    kwargs: Dict[str, Any] = {
        "model": model_config["litellm_model"],
        "messages": messages,
        "max_tokens": effective_max,
        "seed": seed,
    }

    if model_config["is_reasoning"]:
        # Reasoning models: no temperature/top_p
        # For OpenAI o-series, use max_completion_tokens instead of max_tokens
        if model_config["provider"] == "openai":
            kwargs.pop("max_tokens")
            kwargs["max_completion_tokens"] = effective_max
    else:
        kwargs["temperature"] = 0.6

    max_retries = 5
    base_wait = 5.0

    async with semaphore:
        t0 = time.time()
        last_exc = None

        for attempt in range(max_retries + 1):
            try:
                response = await litellm.acompletion(**kwargs)
                elapsed = time.time() - t0

                choice = response.choices[0]
                content = choice.message.content or ""

                # Extract reasoning tokens if available
                reasoning = getattr(choice.message, "reasoning_content", None) or ""
                if reasoning:
                    trace = f"<think>{reasoning}</think>{content}"
                    has_think_tags = True
                else:
                    trace = content
                    has_think_tags = "<think>" in trace and "</think>" in trace

                # Treat empty responses as retryable failures
                if not trace.strip():
                    raise ValueError("API returned empty response")

                # Token usage
                usage = response.usage
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0

                return {
                    "problem_id": problem_id,
                    "run_idx": run_idx,
                    "rating": rating,
                    "seed": seed,
                    "prompt": user_message,
                    "trace": trace,
                    "has_think_tags": has_think_tags,
                    "trace_length_chars": len(trace),
                    "generation_time_seconds": round(elapsed, 2),
                    "_input_tokens": input_tokens,
                    "_output_tokens": output_tokens,
                }

            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = min(base_wait * (2 ** attempt), 120.0)
                    print(
                        f"  [{model_key}] {problem_id} run {run_idx}: "
                        f"attempt {attempt + 1} failed ({type(exc).__name__}: {exc}), "
                        f"retrying in {wait:.0f}s...",
                        flush=True,
                    )
                    await asyncio.sleep(wait)

        # All retries exhausted
        elapsed = time.time() - t0
        print(
            f"  [{model_key}] {problem_id} run {run_idx}: "
            f"FAILED after {max_retries + 1} attempts: {last_exc}",
            flush=True,
        )
        return {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "rating": rating,
            "seed": seed,
            "prompt": user_message,
            "trace": "",
            "has_think_tags": False,
            "trace_length_chars": 0,
            "generation_time_seconds": round(elapsed, 2),
            "_input_tokens": 0,
            "_output_tokens": 0,
            "_error": str(last_exc),
        }


# ---------------------------------------------------------------------------
# Run all traces for one model + domain
# ---------------------------------------------------------------------------

async def run_model_domain(
    model_key: str,
    model_config: Dict[str, Any],
    domain: str,
    problems,  # pd.DataFrame
    num_runs: int,
    output_dir: Path,
    max_output_tokens: Optional[int] = None,
    concurrency_override: Optional[int] = None,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Generate all traces for one model + domain combination.

    Supports incremental resume: reads existing JSONL and skips completed
    ``(problem_id, run_idx)`` pairs.

    Returns
    -------
    dict
        Summary with keys: completed, new, errors, input_tokens, output_tokens.
    """
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "cot_traces.jsonl"
    cost_path = output_dir / "cost_log.jsonl"

    purged = _purge_failed_traces(jsonl_path)
    if purged:
        print(f"  {model_key}/{domain}: purged {purged} failed trace(s), will retry")

    completed = _load_completed(jsonl_path)
    total_pairs = len(problems) * num_runs
    already_done = len(completed)

    if already_done >= total_pairs:
        print(f"  {model_key}/{domain}: all {total_pairs} traces complete, skipping")
        return {
            "completed": already_done,
            "new": 0,
            "errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    if already_done > 0:
        print(f"  {model_key}/{domain}: resuming ({already_done}/{total_pairs} done)")

    concurrency = concurrency_override or model_config["max_concurrency"]
    semaphore = asyncio.Semaphore(concurrency)

    # Build task list
    tasks = []
    for run_idx in range(num_runs):
        for _, row in problems.iterrows():
            problem_id = str(row["join_key"])
            if (problem_id, run_idx) in completed:
                continue

            if domain == "math":
                rating = float(row.get("level_int", -1))
            elif domain == "sat":
                rating = float(row.get("num_clauses", -1))
            else:
                rating = float(row.get("unnorm_rating", -1))

            user_message = build_user_message(row, domain)
            seed = derive_seed(base_seed, problem_id, run_idx)

            tasks.append((
                model_key, model_config, user_message,
                problem_id, run_idx, rating, seed,
                semaphore, max_output_tokens,
            ))

    # Stream results as they complete (updates progress bar in real time)
    new_count = 0
    persisted_errors = 0  # errors written to JSONL (skipped on resume)
    transient_errors = 0  # gather exceptions (not persisted, retried on resume)
    total_input_tokens = 0
    total_output_tokens = 0

    pbar = tqdm(
        total=len(tasks),
        desc=f"  {model_key}/{domain}",
        unit="trace",
    )

    async def _run_and_record(args):
        """Run a single trace and write results immediately."""
        nonlocal new_count, persisted_errors, transient_errors
        nonlocal total_input_tokens, total_output_tokens
        try:
            result = await generate_single_trace(*args)
        except Exception as exc:
            print(f"  [{model_key}] Unexpected exception: {exc}", flush=True)
            transient_errors += 1
            pbar.update(1)
            return

        # Write trace (omit internal cost fields)
        trace_record = {
            k: v for k, v in result.items()
            if not k.startswith("_")
        }

        if result.get("_error"):
            persisted_errors += 1
        else:
            new_count += 1

        _append_jsonl(trace_record, jsonl_path)

        # Write cost log
        cost_record = {
            "problem_id": result["problem_id"],
            "run_idx": result["run_idx"],
            "input_tokens": result["_input_tokens"],
            "output_tokens": result["_output_tokens"],
            "timestamp": datetime.now().isoformat(),
        }
        _append_jsonl(cost_record, cost_path)

        total_input_tokens += result["_input_tokens"]
        total_output_tokens += result["_output_tokens"]
        pbar.update(1)

    # Launch all tasks; the semaphore limits concurrency
    await asyncio.gather(*[_run_and_record(args) for args in tasks])

    pbar.close()

    return {
        "completed": already_done + new_count + persisted_errors,
        "new": new_count,
        "errors": persisted_errors + transient_errors,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def generate_eval_only_config(
    model_key: str,
    domain: str,
    output_base: Path,
    num_runs: int = 5,
) -> Path:
    """Write an eval-only YAML config for an API model.

    Parameters
    ----------
    model_key : str
        Key from ``API_MODELS``.
    domain : str
        ``"code"`` or ``"math"``.
    output_base : Path
        Base directory for trace outputs (e.g. ``$IRT_OUTPUTS_ROOT/api``).
    num_runs : int
        Number of independent traces per problem (default: 5).

    Returns
    -------
    Path
        The written config file path.
    """
    model_dir = output_base / domain / model_key
    if domain == "sat":
        problems_file = "data/datasets/selected_500_sat.parquet"
    elif domain == "math":
        problems_file = "data/datasets/selected_500_math.parquet"
    else:
        problems_file = "data/datasets/selected_500.parquet"

    model_info = API_MODELS[model_key]
    gen_config: Dict[str, Any] = {
        "num_runs": num_runs,
        "max_new_tokens": model_info["max_output_tokens"],
        "seed": 42,
    }
    if not model_info["is_reasoning"]:
        gen_config["temperature"] = 0.6

    config = {
        "model": {
            "name": model_key,
            "model_id": model_info["litellm_model"],
            "hidden_dim": 0,
            "num_layers": 0,
            "eval_only": True,
        },
        "generation": gen_config,
        "paths": {
            "problems": problems_file,
            "pipeline": {
                "cot_traces": str(model_dir / "cot_traces.jsonl"),
            },
            "analysis": {
                "cot_analysis": str(model_dir / "cot_analysis.parquet"),
            },
        },
    }

    config_dir = PROJECT_ROOT / "configs" / "api" / domain
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{model_key}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path
