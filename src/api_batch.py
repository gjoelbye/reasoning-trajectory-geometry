"""Batch API support for OpenAI, Anthropic, Google, and Together AI.

Submits requests via provider batch APIs (50% cost reduction, 24h window),
polls for completion, and collects results into the same ``cot_traces.jsonl``
schema as real-time generation.

Usage
-----
    from src.api_batch import supports_batch, run_batch_pipeline

    if supports_batch(model_config["provider"]):
        summary = await run_batch_pipeline(model_key, model_config, domain, ...)
"""

from __future__ import annotations

import json
import os
import tempfile

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api_generation import (
    _append_jsonl,
    _load_completed,
    _purge_failed_traces,
    build_user_message,
    derive_seed,
)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

BATCH_PROVIDERS = {"openai", "anthropic", "google", "together"}

# Together's batch API only supports a subset of models.
# Models not listed here will fall back to real-time generation.
TOGETHER_BATCH_MODELS = {
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-R1",
}


def supports_batch(provider: str, litellm_model: str) -> bool:
    """Return True if the provider (and model) supports batch API."""
    if provider not in BATCH_PROVIDERS:
        return False
    if provider == "together":
        # Strip together_ai/ prefix
        model_id = litellm_model
        if model_id.startswith("together_ai/"):
            model_id = model_id[len("together_ai/"):]
        return model_id in TOGETHER_BATCH_MODELS
    if provider == "google":
        # Gemma models don't support batchGenerateContent on the Gemini API
        model_id = litellm_model
        if model_id.startswith("gemini/"):
            model_id = model_id[len("gemini/"):]
        if model_id.startswith("gemma"):
            return False
    return True


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_batch_state(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load ``batch_state.json`` from *output_dir*.

    Returns None if not found or if status is ``"completed"``.
    """
    state_path = output_dir / "batch_state.json"
    if not state_path.exists():
        return None
    with open(state_path) as f:
        state = json.load(f)
    if state.get("status") in ("completed", "failed"):
        return None
    return state


def save_batch_state(state: Dict[str, Any], output_dir: Path) -> None:
    """Atomic write of ``batch_state.json`` (write tmp + rename)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "batch_state.json"
    fd, tmp_path = tempfile.mkstemp(
        dir=output_dir, prefix=".batch_state_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, state_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Prompt recovery (backward compat for old 5-element custom_id_map entries)
# ---------------------------------------------------------------------------

def _recover_prompt(
    meta: list,
    problems,  # pd.DataFrame
    domain: str,
) -> str:
    """Get the user prompt for a batch result entry.

    New-format *meta* has 4 elements ``[problem_id, run_idx, seed, rating]``;
    old format has 5 with ``user_message`` appended.  For new format, rebuild
    the prompt from *problems* via ``build_user_message()``.
    """
    if len(meta) > 4:
        # Old format: prompt stored directly
        return meta[4]
    # New format: reconstruct from problems DataFrame
    if problems is None:
        return ""
    problem_id = meta[0]
    row = problems.loc[problems["join_key"].astype(str) == str(problem_id)]
    if row.empty:
        return ""
    return build_user_message(row.iloc[0], domain)


# ---------------------------------------------------------------------------
# Build pending tasks (shared between providers)
# ---------------------------------------------------------------------------

def build_pending_tasks(
    model_key: str,
    model_config: Dict[str, Any],
    domain: str,
    problems,  # pd.DataFrame
    num_runs: int,
    output_dir: Path,
    base_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Build list of task dicts for traces not yet completed.

    Each dict has keys: problem_id, run_idx, rating, seed, user_message.
    """
    jsonl_path = output_dir / "cot_traces.jsonl"
    completed = _load_completed(jsonl_path)

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

            tasks.append({
                "problem_id": problem_id,
                "run_idx": run_idx,
                "rating": rating,
                "seed": seed,
                "user_message": user_message,
            })

    return tasks


# ---------------------------------------------------------------------------
# OpenAI Batch API
# ---------------------------------------------------------------------------

def _openai_client():
    """Lazy-init OpenAI client."""
    from openai import OpenAI
    return OpenAI()


def submit_openai_batch(
    tasks: List[Dict[str, Any]],
    model_key: str,
    model_config: Dict[str, Any],
    domain: str,
    output_dir: Path,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Upload JSONL and create an OpenAI batch.

    Returns the batch state dict.
    """
    client = _openai_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use per-model limit, optionally capped by CLI override
    model_max = model_config["max_output_tokens"]
    effective_max = min(max_output_tokens, model_max) if max_output_tokens else model_max

    # Build custom_id map and JSONL lines
    custom_id_map = {}
    lines = []

    for i, task in enumerate(tasks):
        custom_id = f"req_{i}"
        custom_id_map[custom_id] = [
            task["problem_id"],
            task["run_idx"],
            task["seed"],
            task["rating"],
        ]

        body: Dict[str, Any] = {
            "model": model_config["litellm_model"],
            "messages": [{"role": "user", "content": task["user_message"]}],
            "seed": task["seed"],
        }

        if model_config["is_reasoning"]:
            body["max_completion_tokens"] = effective_max
        else:
            body["max_tokens"] = effective_max
            body["temperature"] = 0.6

        lines.append(json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }, ensure_ascii=False))

    # Write to temp file and upload
    jsonl_content = "\n".join(lines)
    input_file = client.files.create(
        file=("batch_input.jsonl", jsonl_content.encode("utf-8")),
        purpose="batch",
    )

    # Create batch
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"model_key": model_key, "domain": domain},
    )

    state = {
        "model_key": model_key,
        "domain": domain,
        "provider": "openai",
        "batch_id": batch.id,
        "input_file_id": input_file.id,
        "status": "submitted",
        "num_requests": len(tasks),
        "custom_id_map": custom_id_map,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "poll_count": 0,
        "error": None,
    }

    return state


def poll_openai_batch(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check status of an OpenAI batch. Returns updated state."""
    client = _openai_client()
    batch = client.batches.retrieve(state["batch_id"])

    state["poll_count"] += 1

    if batch.status == "completed":
        state["status"] = "collecting"
        state["output_file_id"] = batch.output_file_id
        state["error_file_id"] = batch.error_file_id
        state["completed_at"] = datetime.now().isoformat()
    elif batch.status in ("failed", "expired", "cancelled"):
        state["status"] = "failed"
        errors = []
        if batch.errors and batch.errors.data:
            errors = [e.message for e in batch.errors.data[:5]]
        state["error"] = f"Batch {batch.status}: {errors}"
    else:
        state["status"] = "polling"

    # Report progress
    counts = batch.request_counts
    if counts:
        completed = counts.completed or 0
        failed = counts.failed or 0
        total = counts.total or state["num_requests"]
        print(
            f"  [openai] Batch {state['batch_id'][:20]}... "
            f"status={batch.status} "
            f"progress={completed + failed}/{total} "
            f"(poll #{state['poll_count']})",
            flush=True,
        )

    return state


def collect_openai_results(
    state: Dict[str, Any],
    output_dir: Path,
    problems=None,  # pd.DataFrame, needed for new-format custom_id_map
    domain: str = "",
) -> Dict[str, int]:
    """Download and parse OpenAI batch results.

    Returns dict with new, errors, input_tokens, output_tokens counts.
    """
    client = _openai_client()
    jsonl_path = output_dir / "cot_traces.jsonl"
    cost_path = output_dir / "cost_log.jsonl"

    # Skip already-collected traces
    already_done = _load_completed(jsonl_path)

    # Download output file
    output_content = client.files.content(state["output_file_id"])

    new_count = 0
    error_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for line in output_content.text.strip().split("\n"):
        if not line.strip():
            continue
        result = json.loads(line)
        custom_id = result["custom_id"]
        meta = state["custom_id_map"].get(custom_id)
        if meta is None:
            continue

        problem_id, run_idx, seed, rating = meta[:4]
        prompt = _recover_prompt(meta, problems, domain)

        if (problem_id, run_idx) in already_done:
            continue

        response_body = result.get("response", {}).get("body", {})
        error_obj = result.get("error")

        if error_obj or not response_body.get("choices"):
            error_count += 1
            _append_jsonl({
                "problem_id": problem_id, "run_idx": run_idx,
                "rating": rating, "seed": seed, "prompt": prompt,
                "trace": "", "has_think_tags": False,
                "trace_length_chars": 0, "generation_time_seconds": 0.0,
            }, jsonl_path)
            _append_jsonl({
                "problem_id": problem_id, "run_idx": run_idx,
                "input_tokens": 0, "output_tokens": 0,
                "timestamp": datetime.now().isoformat(),
            }, cost_path)
            continue

        choice = response_body["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        # Handle reasoning_content (o4-mini)
        reasoning = message.get("reasoning_content", "") or ""
        if reasoning:
            trace = f"<think>{reasoning}</think>{content}"
            has_think_tags = True
        else:
            trace = content
            has_think_tags = "<think>" in trace and "</think>" in trace

        usage = response_body.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("completion_tokens", 0) or 0

        trace_record = {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "rating": rating,
            "seed": seed,
            "prompt": prompt,
            "trace": trace,
            "has_think_tags": has_think_tags,
            "trace_length_chars": len(trace),
            "generation_time_seconds": 0.0,
        }
        _append_jsonl(trace_record, jsonl_path)

        cost_record = {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "timestamp": datetime.now().isoformat(),
        }
        _append_jsonl(cost_record, cost_path)

        new_count += 1
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    # Also check error file (requests that failed at the batch level)
    if state.get("error_file_id"):
        try:
            error_content = client.files.content(state["error_file_id"])
            for line in error_content.text.strip().split("\n"):
                if not line.strip():
                    continue
                err_entry = json.loads(line)
                cid = err_entry.get("custom_id")
                meta = state["custom_id_map"].get(cid) if cid else None
                if meta is None:
                    continue
                pid, ridx, seed, rating = meta[:4]
                prompt = _recover_prompt(meta, problems, domain)
                if (pid, ridx) in already_done:
                    continue
                error_count += 1
                _append_jsonl({
                    "problem_id": pid, "run_idx": ridx, "rating": rating,
                    "seed": seed, "prompt": prompt, "trace": "",
                    "has_think_tags": False, "trace_length_chars": 0,
                    "generation_time_seconds": 0.0,
                }, jsonl_path)
                _append_jsonl({
                    "problem_id": pid, "run_idx": ridx,
                    "input_tokens": 0, "output_tokens": 0,
                    "timestamp": datetime.now().isoformat(),
                }, cost_path)
        except Exception as exc:
            print(f"  [openai] Warning: failed to read error file: {exc}", flush=True)

    return {
        "new": new_count,
        "errors": error_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }


# ---------------------------------------------------------------------------
# Anthropic Batch API
# ---------------------------------------------------------------------------

def _anthropic_client():
    """Lazy-init Anthropic client."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for Anthropic batch API.\n"
            "Install it with: pip install anthropic"
        )
    return anthropic.Anthropic()


def submit_anthropic_batch(
    tasks: List[Dict[str, Any]],
    model_key: str,
    model_config: Dict[str, Any],
    domain: str,
    output_dir: Path,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Submit an Anthropic message batch.

    Returns the batch state dict.
    """
    if len(tasks) > 10000:
        raise ValueError(
            f"Anthropic batch API supports max 10,000 requests per batch, "
            f"got {len(tasks)}. Reduce --num-runs or split manually."
        )

    client = _anthropic_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use per-model limit, optionally capped by CLI override
    model_max = model_config["max_output_tokens"]
    effective_max = min(max_output_tokens, model_max) if max_output_tokens else model_max

    # Strip "anthropic/" prefix from litellm model ID
    model_id = model_config["litellm_model"]
    if model_id.startswith("anthropic/"):
        model_id = model_id[len("anthropic/"):]

    custom_id_map = {}
    requests = []

    for i, task in enumerate(tasks):
        custom_id = f"req_{i}"
        custom_id_map[custom_id] = [
            task["problem_id"],
            task["run_idx"],
            task["seed"],
            task["rating"],
        ]

        params: Dict[str, Any] = {
            "model": model_id,
            "max_tokens": effective_max,
            "messages": [{"role": "user", "content": task["user_message"]}],
        }

        if not model_config["is_reasoning"]:
            params["temperature"] = 0.6

        requests.append({
            "custom_id": custom_id,
            "params": params,
        })

    batch = client.messages.batches.create(requests=requests)

    state = {
        "model_key": model_key,
        "domain": domain,
        "provider": "anthropic",
        "batch_id": batch.id,
        "status": "submitted",
        "num_requests": len(tasks),
        "custom_id_map": custom_id_map,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "poll_count": 0,
        "error": None,
    }

    return state


def poll_anthropic_batch(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check status of an Anthropic batch. Returns updated state."""
    client = _anthropic_client()
    batch = client.messages.batches.retrieve(state["batch_id"])

    state["poll_count"] += 1

    if batch.processing_status == "ended":
        state["status"] = "collecting"
        state["completed_at"] = datetime.now().isoformat()

        counts = batch.request_counts
        print(
            f"  [anthropic] Batch {state['batch_id'][:20]}... "
            f"status=ended "
            f"succeeded={counts.succeeded} errored={counts.errored} "
            f"expired={counts.expired} canceled={counts.canceled} "
            f"(poll #{state['poll_count']})",
            flush=True,
        )

        if counts.succeeded == 0 and (counts.errored + counts.expired + counts.canceled) > 0:
            state["status"] = "failed"
            state["error"] = (
                f"All requests failed: errored={counts.errored}, "
                f"expired={counts.expired}, canceled={counts.canceled}"
            )
    else:
        state["status"] = "polling"
        counts = batch.request_counts
        processing = counts.processing
        succeeded = counts.succeeded
        total = state["num_requests"]
        print(
            f"  [anthropic] Batch {state['batch_id'][:20]}... "
            f"status={batch.processing_status} "
            f"progress={succeeded}/{total} (processing={processing}) "
            f"(poll #{state['poll_count']})",
            flush=True,
        )

    return state


def collect_anthropic_results(
    state: Dict[str, Any],
    output_dir: Path,
    problems=None,  # pd.DataFrame, needed for new-format custom_id_map
    domain: str = "",
) -> Dict[str, int]:
    """Stream and parse Anthropic batch results.

    Returns dict with new, errors, input_tokens, output_tokens counts.
    """
    client = _anthropic_client()
    jsonl_path = output_dir / "cot_traces.jsonl"
    cost_path = output_dir / "cost_log.jsonl"

    already_done = _load_completed(jsonl_path)

    new_count = 0
    error_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for entry in client.messages.batches.results(state["batch_id"]):
        custom_id = entry.custom_id
        meta = state["custom_id_map"].get(custom_id)
        if meta is None:
            continue

        problem_id, run_idx, seed, rating = meta[:4]
        prompt = _recover_prompt(meta, problems, domain)

        if (problem_id, run_idx) in already_done:
            continue

        if entry.result.type != "succeeded":
            error_count += 1
            _append_jsonl({
                "problem_id": problem_id, "run_idx": run_idx,
                "rating": rating, "seed": seed, "prompt": prompt,
                "trace": "", "has_think_tags": False,
                "trace_length_chars": 0, "generation_time_seconds": 0.0,
            }, jsonl_path)
            _append_jsonl({
                "problem_id": problem_id, "run_idx": run_idx,
                "input_tokens": 0, "output_tokens": 0,
                "timestamp": datetime.now().isoformat(),
            }, cost_path)
            continue

        message = entry.result.message

        # Extract text and thinking blocks
        text_parts = []
        thinking_parts = []
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)

        content = "".join(text_parts)
        reasoning = "".join(thinking_parts)

        if reasoning:
            trace = f"<think>{reasoning}</think>{content}"
            has_think_tags = True
        else:
            trace = content
            has_think_tags = "<think>" in trace and "</think>" in trace

        input_tokens = message.usage.input_tokens if message.usage else 0
        output_tokens = message.usage.output_tokens if message.usage else 0

        trace_record = {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "rating": rating,
            "seed": seed,
            "prompt": prompt,
            "trace": trace,
            "has_think_tags": has_think_tags,
            "trace_length_chars": len(trace),
            "generation_time_seconds": 0.0,
        }
        _append_jsonl(trace_record, jsonl_path)

        cost_record = {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "timestamp": datetime.now().isoformat(),
        }
        _append_jsonl(cost_record, cost_path)

        new_count += 1
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    return {
        "new": new_count,
        "errors": error_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }


# ---------------------------------------------------------------------------
# Google Batch API
# ---------------------------------------------------------------------------

def _google_client():
    """Lazy-init Google GenAI client."""
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "The 'google-genai' package is required for Google batch API.\n"
            "Install it with: pip install google-genai"
        )
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def submit_google_batch(
    tasks: List[Dict[str, Any]],
    model_key: str,
    model_config: Dict[str, Any],
    domain: str,
    output_dir: Path,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Upload JSONL and create a Google batch job.

    Returns the batch state dict.
    """
    from google.genai import types as genai_types

    client = _google_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use per-model limit, optionally capped by CLI override
    model_max = model_config["max_output_tokens"]
    effective_max = min(max_output_tokens, model_max) if max_output_tokens else model_max

    # Strip "gemini/" prefix from litellm model ID → raw model ID
    model_id = model_config["litellm_model"]
    if model_id.startswith("gemini/"):
        model_id = model_id[len("gemini/"):]

    custom_id_map = {}
    lines = []

    for i, task in enumerate(tasks):
        custom_id = f"req_{i}"
        custom_id_map[custom_id] = [
            task["problem_id"],
            task["run_idx"],
            task["seed"],
            task["rating"],
        ]

        generation_config: Dict[str, Any] = {
            "max_output_tokens": effective_max,
            "response_mime_type": "text/plain",
        }

        if not model_config["is_reasoning"]:
            generation_config["temperature"] = 0.6

        request = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": task["user_message"]}],
                }
            ],
            "generation_config": generation_config,
        }

        lines.append(json.dumps({
            "key": custom_id,
            "request": request,
        }, ensure_ascii=False))

    # Write JSONL to temp file, upload via File API
    jsonl_content = "\n".join(lines)
    fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", prefix="batch_input_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(jsonl_content)

        uploaded_file = client.files.upload(
            file=tmp_path,
            config=genai_types.UploadFileConfig(mime_type="application/jsonl"),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Create batch job
    batch_job = client.batches.create(
        model=model_id,
        src=uploaded_file.name,
        config=genai_types.CreateBatchJobConfig(
            display_name=f"{model_key}/{domain}",
        ),
    )

    state = {
        "model_key": model_key,
        "domain": domain,
        "provider": "google",
        "batch_id": batch_job.name,
        "input_file_name": uploaded_file.name,
        "status": "submitted",
        "num_requests": len(tasks),
        "custom_id_map": custom_id_map,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "poll_count": 0,
        "error": None,
    }

    return state


def poll_google_batch(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check status of a Google batch job. Returns updated state."""
    client = _google_client()
    batch_job = client.batches.get(name=state["batch_id"])

    state["poll_count"] += 1

    job_state = batch_job.state.name if hasattr(batch_job.state, "name") else str(batch_job.state)

    if job_state == "JOB_STATE_SUCCEEDED":
        state["status"] = "collecting"
        state["completed_at"] = datetime.now().isoformat()
        # Store output file name for collection
        if batch_job.dest and batch_job.dest.file_name:
            state["output_file_name"] = batch_job.dest.file_name
    elif job_state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
        state["status"] = "failed"
        state["error"] = f"Batch job {job_state}"
    else:
        state["status"] = "polling"

    print(
        f"  [google] Batch {state['batch_id'][:30]}... "
        f"state={job_state} "
        f"(poll #{state['poll_count']})",
        flush=True,
    )

    return state


def collect_google_results(
    state: Dict[str, Any],
    output_dir: Path,
    problems=None,  # pd.DataFrame, needed for new-format custom_id_map
    domain: str = "",
) -> Dict[str, int]:
    """Download and parse Google batch results.

    Returns dict with new, errors, input_tokens, output_tokens counts.
    """
    client = _google_client()
    jsonl_path = output_dir / "cot_traces.jsonl"
    cost_path = output_dir / "cost_log.jsonl"

    already_done = _load_completed(jsonl_path)

    # Get output file name — either from state (saved during polling)
    # or re-fetch from the batch job
    output_file_name = state.get("output_file_name")
    if not output_file_name:
        batch_job = client.batches.get(name=state["batch_id"])
        output_file_name = batch_job.dest.file_name

    # Download output file
    dl_result = client.files.download(file=output_file_name)
    # dl_result is bytes
    if isinstance(dl_result, bytes):
        output_text = dl_result.decode("utf-8")
    else:
        output_text = str(dl_result)

    new_count = 0
    error_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for line in output_text.strip().split("\n"):
        if not line.strip():
            continue
        result = json.loads(line)
        custom_id = result.get("key")
        meta = state["custom_id_map"].get(custom_id)
        if meta is None:
            continue

        problem_id, run_idx, seed, rating = meta[:4]
        prompt = _recover_prompt(meta, problems, domain)

        if (problem_id, run_idx) in already_done:
            continue

        response = result.get("response")

        # Check for errors
        if not response or not response.get("candidates"):
            error_count += 1
            _append_jsonl({
                "problem_id": problem_id, "run_idx": run_idx,
                "rating": rating, "seed": seed, "prompt": prompt,
                "trace": "", "has_think_tags": False,
                "trace_length_chars": 0, "generation_time_seconds": 0.0,
            }, jsonl_path)
            _append_jsonl({
                "problem_id": problem_id, "run_idx": run_idx,
                "input_tokens": 0, "output_tokens": 0,
                "timestamp": datetime.now().isoformat(),
            }, cost_path)
            continue

        # Extract text and thinking parts from candidates
        candidate = response["candidates"][0]
        content_parts = candidate.get("content", {}).get("parts", [])

        text_parts = []
        thinking_parts = []
        for part in content_parts:
            if part.get("thought"):
                thinking_parts.append(part.get("text", ""))
            else:
                text_parts.append(part.get("text", ""))

        content = "".join(text_parts)
        reasoning = "".join(thinking_parts)

        if reasoning:
            trace = f"<think>{reasoning}</think>{content}"
            has_think_tags = True
        else:
            trace = content
            has_think_tags = "<think>" in trace and "</think>" in trace

        # Extract token usage
        usage = response.get("usageMetadata") or response.get("usage_metadata", {})
        input_tokens = usage.get("promptTokenCount") or usage.get("prompt_token_count", 0) or 0
        output_tokens = usage.get("candidatesTokenCount") or usage.get("candidates_token_count", 0) or 0

        trace_record = {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "rating": rating,
            "seed": seed,
            "prompt": prompt,
            "trace": trace,
            "has_think_tags": has_think_tags,
            "trace_length_chars": len(trace),
            "generation_time_seconds": 0.0,
        }
        _append_jsonl(trace_record, jsonl_path)

        cost_record = {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "timestamp": datetime.now().isoformat(),
        }
        _append_jsonl(cost_record, cost_path)

        new_count += 1
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    return {
        "new": new_count,
        "errors": error_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }


# ---------------------------------------------------------------------------
# Together AI Batch API
# ---------------------------------------------------------------------------

def _together_client():
    """Lazy-init Together AI client."""
    try:
        from together import Together
    except ImportError:
        raise ImportError(
            "The 'together' package is required for Together AI batch API.\n"
            "Install it with: pip install together"
        )
    return Together()


def submit_together_batch(
    tasks: List[Dict[str, Any]],
    model_key: str,
    model_config: Dict[str, Any],
    domain: str,
    output_dir: Path,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Upload JSONL and create a Together AI batch.

    Returns the batch state dict.
    """
    client = _together_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use per-model limit, optionally capped by CLI override
    model_max = model_config["max_output_tokens"]
    effective_max = min(max_output_tokens, model_max) if max_output_tokens else model_max

    # Strip "together_ai/" prefix from litellm model ID → raw model ID
    model_id = model_config["litellm_model"]
    if model_id.startswith("together_ai/"):
        model_id = model_id[len("together_ai/"):]

    custom_id_map = {}
    lines = []

    for i, task in enumerate(tasks):
        custom_id = f"req_{i}"
        custom_id_map[custom_id] = [
            task["problem_id"],
            task["run_idx"],
            task["seed"],
            task["rating"],
        ]

        # Together enforces max_tokens + input_tokens <= context_window.
        # Estimate input tokens (~1 tok per 3 chars + margin) and reduce.
        est_input = len(task["user_message"]) // 3 + 256
        safe_max = max(1024, effective_max - est_input)

        body: Dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": task["user_message"]}],
            "max_tokens": safe_max,
            "temperature": 0.6,
            "seed": task["seed"],
        }

        lines.append(json.dumps({
            "custom_id": custom_id,
            "body": body,
        }, ensure_ascii=False))

    # Write to temp file and upload
    jsonl_content = "\n".join(lines)
    fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", prefix="batch_input_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(jsonl_content)

        uploaded_file = client.files.upload(
            file=tmp_path, purpose="batch-api", check=False,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Create batch
    batch_response = client.batches.create(
        input_file_id=uploaded_file.id, endpoint="/v1/chat/completions",
    )

    state = {
        "model_key": model_key,
        "domain": domain,
        "provider": "together",
        "batch_id": batch_response.job.id,
        "input_file_id": uploaded_file.id,
        "status": "submitted",
        "num_requests": len(tasks),
        "custom_id_map": custom_id_map,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "poll_count": 0,
        "error": None,
    }

    return state


def poll_together_batch(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check status of a Together AI batch. Returns updated state."""
    client = _together_client()
    batch = client.batches.retrieve(state["batch_id"])

    state["poll_count"] += 1

    status = batch.status
    if status == "COMPLETED":
        state["status"] = "collecting"
        state["output_file_id"] = batch.output_file_id
        state["error_file_id"] = getattr(batch, "error_file_id", None)
        state["completed_at"] = datetime.now().isoformat()
    elif status in ("FAILED", "CANCELLED"):
        state["status"] = "failed"
        state["error"] = f"Batch {status}"
    else:
        state["status"] = "polling"

    print(
        f"  [together] Batch {state['batch_id'][:20]}... "
        f"status={status} "
        f"(poll #{state['poll_count']})",
        flush=True,
    )

    return state


def collect_together_results(
    state: Dict[str, Any],
    output_dir: Path,
    problems=None,  # pd.DataFrame, needed for new-format custom_id_map
    domain: str = "",
) -> Dict[str, int]:
    """Download and parse Together AI batch results.

    Returns dict with new, errors, input_tokens, output_tokens counts.
    """
    client = _together_client()
    jsonl_path = output_dir / "cot_traces.jsonl"
    cost_path = output_dir / "cost_log.jsonl"

    already_done = _load_completed(jsonl_path)

    # Download output file
    response = client.files.content(state["output_file_id"])
    output_text = response.text()

    new_count = 0
    error_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for line in output_text.strip().split("\n"):
        if not line.strip():
            continue
        result = json.loads(line)
        custom_id = result["custom_id"]
        meta = state["custom_id_map"].get(custom_id)
        if meta is None:
            continue

        problem_id, run_idx, seed, rating = meta[:4]
        prompt = _recover_prompt(meta, problems, domain)

        if (problem_id, run_idx) in already_done:
            continue

        response_body = result.get("response", {}).get("body", {})
        error_obj = result.get("error")

        if error_obj or not response_body.get("choices"):
            error_count += 1
            _append_jsonl({
                "problem_id": problem_id, "run_idx": run_idx,
                "rating": rating, "seed": seed, "prompt": prompt,
                "trace": "", "has_think_tags": False,
                "trace_length_chars": 0, "generation_time_seconds": 0.0,
            }, jsonl_path)
            _append_jsonl({
                "problem_id": problem_id, "run_idx": run_idx,
                "input_tokens": 0, "output_tokens": 0,
                "timestamp": datetime.now().isoformat(),
            }, cost_path)
            continue

        choice = response_body["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        # All Together AI models in our registry are non-reasoning
        trace = content
        has_think_tags = "<think>" in trace and "</think>" in trace

        usage = response_body.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("completion_tokens", 0) or 0

        trace_record = {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "rating": rating,
            "seed": seed,
            "prompt": prompt,
            "trace": trace,
            "has_think_tags": has_think_tags,
            "trace_length_chars": len(trace),
            "generation_time_seconds": 0.0,
        }
        _append_jsonl(trace_record, jsonl_path)

        cost_record = {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "timestamp": datetime.now().isoformat(),
        }
        _append_jsonl(cost_record, cost_path)

        new_count += 1
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    # Check error file if present
    if state.get("error_file_id"):
        try:
            err_response = client.files.content(state["error_file_id"])
            error_text = err_response.text()

            for line in error_text.strip().split("\n"):
                if not line.strip():
                    continue
                err_entry = json.loads(line)
                cid = err_entry.get("custom_id")
                meta = state["custom_id_map"].get(cid) if cid else None
                if meta is None:
                    continue
                pid, ridx, seed, rating = meta[:4]
                prompt = _recover_prompt(meta, problems, domain)
                if (pid, ridx) in already_done:
                    continue
                error_count += 1
                _append_jsonl({
                    "problem_id": pid, "run_idx": ridx, "rating": rating,
                    "seed": seed, "prompt": prompt, "trace": "",
                    "has_think_tags": False, "trace_length_chars": 0,
                    "generation_time_seconds": 0.0,
                }, jsonl_path)
                _append_jsonl({
                    "problem_id": pid, "run_idx": ridx,
                    "input_tokens": 0, "output_tokens": 0,
                    "timestamp": datetime.now().isoformat(),
                }, cost_path)
        except Exception as exc:
            print(f"  [together] Warning: failed to read error file: {exc}", flush=True)

    return {
        "new": new_count,
        "errors": error_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_batch_pipeline(
    model_key: str,
    model_config: Dict[str, Any],
    domain: str,
    problems,  # pd.DataFrame
    num_runs: int,
    output_dir: Path,
    max_output_tokens: Optional[int] = None,
    base_seed: int = 42,
    poll_interval: int = 60,
    collect_only: bool = False,
    submit_only: bool = False,
) -> Dict[str, Any]:
    """End-to-end batch pipeline: submit → poll → collect.

    Handles resume at any phase via ``batch_state.json``.

    Parameters
    ----------
    collect_only : bool
        If True, only collect results from an already-submitted batch.
    submit_only : bool
        If True, submit the batch and return immediately without polling.

    Returns
    -------
    dict
        Summary with keys: completed, new, errors, input_tokens, output_tokens.
        When *submit_only* is True, also includes ``status: "submitted"``.
    """
    import asyncio

    provider = model_config["provider"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Purge failed traces so they get retried
    jsonl_path = output_dir / "cot_traces.jsonl"
    purged = _purge_failed_traces(jsonl_path)
    if purged:
        print(f"  {model_key}/{domain}: purged {purged} failed trace(s), will retry")

    # Check total expected
    total_pairs = len(problems) * num_runs
    already_done = len(_load_completed(jsonl_path))

    if already_done >= total_pairs:
        print(f"  {model_key}/{domain}: all {total_pairs} traces complete, skipping")
        return {
            "completed": already_done,
            "new": 0,
            "errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    # Check for existing batch state (resume)
    state = load_batch_state(output_dir)

    if collect_only and state is None:
        print(f"  {model_key}/{domain}: no active batch to collect")
        return {
            "completed": already_done,
            "new": 0,
            "errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    # --- Submit phase ---
    if state is None:
        tasks = build_pending_tasks(
            model_key, model_config, domain, problems,
            num_runs, output_dir, base_seed,
        )

        if not tasks:
            print(f"  {model_key}/{domain}: no pending tasks")
            return {
                "completed": already_done,
                "new": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        print(
            f"  {model_key}/{domain}: submitting {len(tasks)} requests "
            f"via {provider} batch API..."
        )

        if provider == "openai":
            state = submit_openai_batch(
                tasks, model_key, model_config, domain, output_dir,
                max_output_tokens,
            )
        elif provider == "anthropic":
            state = submit_anthropic_batch(
                tasks, model_key, model_config, domain, output_dir,
                max_output_tokens,
            )
        elif provider == "google":
            state = submit_google_batch(
                tasks, model_key, model_config, domain, output_dir,
                max_output_tokens,
            )
        elif provider == "together":
            state = submit_together_batch(
                tasks, model_key, model_config, domain, output_dir,
                max_output_tokens,
            )
        else:
            raise ValueError(f"No batch support for provider: {provider}")

        save_batch_state(state, output_dir)
        print(
            f"  {model_key}/{domain}: batch submitted — "
            f"id={state['batch_id'][:30]}..."
        )

        if submit_only:
            return {
                "status": "submitted",
                "completed": already_done,
                "new": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
    else:
        print(
            f"  {model_key}/{domain}: resuming batch {state['batch_id'][:30]}... "
            f"(status={state['status']})"
        )

        if submit_only:
            return {
                "status": "already_submitted",
                "completed": already_done,
                "new": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

    # --- Poll phase ---
    if state["status"] in ("submitted", "polling"):
        if collect_only:
            print(
                f"  {model_key}/{domain}: batch not yet complete "
                f"(status={state['status']}). Run with --batch to poll."
            )
            return {
                "completed": already_done,
                "new": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        print(
            f"  {model_key}/{domain}: polling every {poll_interval}s "
            f"(Ctrl+C to stop, re-run to resume)..."
        )

        max_polls = (25 * 3600) // poll_interval  # 25h safety margin

        while state["status"] in ("submitted", "polling"):
            if state["poll_count"] >= max_polls:
                state["status"] = "failed"
                state["error"] = (
                    f"Poll timeout after {state['poll_count']} polls "
                    f"({state['poll_count'] * poll_interval / 3600:.1f}h)"
                )
                save_batch_state(state, output_dir)
                break

            await asyncio.sleep(poll_interval)

            if provider == "openai":
                state = poll_openai_batch(state)
            elif provider == "anthropic":
                state = poll_anthropic_batch(state)
            elif provider == "google":
                state = poll_google_batch(state)
            elif provider == "together":
                state = poll_together_batch(state)

            save_batch_state(state, output_dir)

    # --- Handle failure ---
    if state["status"] == "failed":
        print(f"  {model_key}/{domain}: BATCH FAILED — {state.get('error', 'unknown')}")
        print(
            f"  {model_key}/{domain}: reporting {state['num_requests']} errors "
            f"(worst-case upper bound; re-running may recover results)"
        )
        return {
            "completed": already_done,
            "new": 0,
            "errors": state["num_requests"],
            "input_tokens": 0,
            "output_tokens": 0,
        }

    # --- Collect phase ---
    if state["status"] == "collecting":
        print(f"  {model_key}/{domain}: collecting results...")

        if provider == "openai":
            counts = collect_openai_results(state, output_dir, problems, domain)
        elif provider == "anthropic":
            counts = collect_anthropic_results(state, output_dir, problems, domain)
        elif provider == "google":
            counts = collect_google_results(state, output_dir, problems, domain)
        elif provider == "together":
            counts = collect_together_results(state, output_dir, problems, domain)
        else:
            raise ValueError(f"No batch support for provider: {provider}")

        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        save_batch_state(state, output_dir)

        final_done = len(_load_completed(jsonl_path))
        print(
            f"  {model_key}/{domain}: collected {counts['new']} traces "
            f"({counts['errors']} errors). Total: {final_done}/{total_pairs}"
        )

        return {
            "completed": final_done,
            "new": counts["new"],
            "errors": counts["errors"],
            "input_tokens": counts["input_tokens"],
            "output_tokens": counts["output_tokens"],
        }

    # Shouldn't reach here
    return {
        "completed": already_done,
        "new": 0,
        "errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------

def print_batch_status(output_base: Path) -> None:
    """Scan output dirs for ``batch_state.json`` files and print status."""
    found = []
    for state_path in sorted(output_base.rglob("batch_state.json")):
        with open(state_path) as f:
            state = json.load(f)
        found.append((state_path.parent, state))

    if not found:
        print("No active or completed batches found.")
        return

    print(f"{'Model/Domain':<35} {'Provider':<12} {'Status':<12} {'Requests':>9} {'Polls':>6}")
    print("-" * 76)

    for output_dir, state in found:
        label = f"{state.get('model_key', '?')}/{state.get('domain', '?')}"

        print(
            f"{label:<35} {state['provider']:<12} {state['status']:<12} "
            f"{state['num_requests']:>9} {state['poll_count']:>6}"
        )

        if state.get("error"):
            print(f"  ERROR: {state['error']}")

    print(f"\nTotal: {len(found)} batch(es)")
