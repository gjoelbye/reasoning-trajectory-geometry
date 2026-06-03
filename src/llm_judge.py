"""LLM-as-a-judge per-segment classification of CoT reasoning patterns.

Supports three providers (Gemini, OpenAI, local vLLM) with sentence-level
segmentation.  Produces per-trace aggregated results and per-segment raw
labels with character offsets for full traceability.

The judge configuration (prompt template, category definitions, segmentation
parameters) is loaded from ``configs/llm_judge.yaml`` so that prompt
engineering can be iterated without code changes.

Usage
-----
    from src.llm_judge import (
        load_judge_config,
        build_system_prompt,
        extract_reasoning_text,
        segment_trace_sentences,
        build_sentence_windows,
        build_judge_batch_jsonl,
        parse_judge_batch_results,
        aggregate_segments,
    )
"""

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.models import parse_think_response


# ---------------------------------------------------------------------------
# Project root (same convention as src/config.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JUDGE_CONFIG_PATH = PROJECT_ROOT / "configs" / "llm_judge.yaml"


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "self_correction": {"type": "boolean"},
        "verification": {"type": "boolean"},
        "strategy_shifting": {"type": "boolean"},
        "uncertainty_monitoring": {"type": "boolean"},
        "problem_restatement": {"type": "boolean"},
        "subgoal_decomposition": {"type": "boolean"},
    },
    "required": [
        "self_correction",
        "verification",
        "strategy_shifting",
        "uncertainty_monitoring",
        "problem_restatement",
        "subgoal_decomposition",
    ],
}

CATEGORY_NAMES: List[str] = list(JUDGE_SCHEMA["properties"].keys())


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_judge_config(path: Optional[Path] = None) -> dict:
    """Load the LLM judge YAML configuration.

    Parameters
    ----------
    path : Path or None
        Override config path.  Defaults to ``configs/llm_judge.yaml``.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    p = Path(path) if path else JUDGE_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Judge config not found: {p}")
    with open(p) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Judge output directory resolution
# ---------------------------------------------------------------------------

def judge_model_slug(model_name: str, *, api: bool = False) -> str:
    """Derive a directory-safe slug from a judge model name.

    Parameters
    ----------
    model_name : str
        Full model identifier (e.g. ``"Qwen/Qwen2.5-7B-Instruct"``).
    api : bool
        If True, append ``"_api"`` suffix to distinguish API judges.

    Returns
    -------
    str
        Lowercased last path component, e.g. ``"qwen2.5-7b-instruct"``
        or ``"gpt-4.1-nano_api"``.
    """
    slug = model_name.rsplit("/", 1)[-1].lower()
    if api:
        slug = f"{slug}_api"
    return slug


def judge_output_dir(
    model_results_dir,
    judge_model: str,
    *,
    api: bool = False,
) -> Path:
    """Return the judge output directory for a given judge model.

    Parameters
    ----------
    model_results_dir : str or Path
        Per-model results directory
        (e.g. ``data/results/code/deepseek-r1-7b``).
    judge_model : str
        Full judge model identifier.
    api : bool
        If True, append ``"_api"`` suffix to the slug.

    Returns
    -------
    Path
        ``model_results_dir / "llm_judge" / <slug>``.
    """
    slug = judge_model_slug(judge_model, api=api)
    return Path(model_results_dir) / "llm_judge" / slug


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_base_system_prompt(judge_config: dict, domain: str) -> str:
    """Render the base system prompt template from the judge config.

    Substitutes ``{domain_context}``, ``{category_definitions}``, and
    ``{disambiguation_rules}`` into the template string.  Called internally
    by :func:`build_system_prompt` which appends sentence-level instructions.

    Parameters
    ----------
    judge_config : dict
        Loaded judge YAML config.
    domain : str
        ``"code"`` or ``"math"``.

    Returns
    -------
    str
        Base system prompt without sentence-level instructions.
    """
    if "code" in domain:
        domain_key = "code"
    elif "sat" in domain:
        domain_key = "sat"
    else:
        domain_key = "math"
    domain_context = judge_config["domain_context"][domain_key]

    cat_blocks = []
    for name, cat in judge_config["categories"].items():
        lines = [f"## {name}"]
        lines.append(cat["definition"])
        pos = "; ".join(cat["positive_signals"])
        lines.append(f"LOOK FOR: {pos}")
        neg = "; ".join(cat["negative_signals"])
        lines.append(f"NOT: {neg}")
        examples = "\n".join(f"- {ex}" for ex in cat["examples"])
        lines.append(f"EXAMPLES:\n{examples}")
        cat_blocks.append("\n".join(lines))

    category_definitions = "\n\n".join(cat_blocks)

    disambig_lines = ["## DISAMBIGUATION",
                      "Some segments may exhibit multiple overlapping "
                      "patterns. Apply each category independently:"]
    for rule in judge_config["disambiguation"]:
        disambig_lines.append(f"- {rule}")
    disambiguation_rules = "\n".join(disambig_lines)

    template = judge_config["system_prompt"]
    prompt = template.format(
        domain_context=domain_context,
        category_definitions=category_definitions,
        disambiguation_rules=disambiguation_rules,
    )
    return prompt.strip()


def prompt_version_hash(prompt: str) -> str:
    """Return a short SHA-256 hex digest of the rendered prompt."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Reasoning text extraction
# ---------------------------------------------------------------------------

def extract_reasoning_text(trace: str, domain: str) -> str:
    """Extract only the reasoning portion from a full model trace.

    For reasoning models (with ``<think>`` tags), returns the think text.
    For non-reasoning models, strips the answer portion (code fence for
    code domain, ``\\boxed{}`` for math domain) and returns the reasoning
    text before it.

    Parameters
    ----------
    trace : str
        Full model output from ``cot_traces.jsonl``.
    domain : str
        ``"codeforces"`` or ``"math"`` (or containing those substrings).

    Returns
    -------
    str
        Reasoning text only.
    """
    think, answer, _ = parse_think_response(trace)
    think = think.strip()
    answer = answer.strip()

    if think:
        return think

    # Think tags present but think content empty → no reasoning produced
    if "</think>" in trace:
        return ""

    # Non-reasoning model: use full trace but strip answer portion
    text = trace

    if "code" in domain or "codeforces" in domain:
        m = (re.search(r"```[Pp]ython", text)
             or re.search(r"```[Cc]pp", text)
             or re.search(r"```", text))
        if m:
            text = text[:m.start()]
        else:
            for tag in (r"<answer>", r"<solution>",
                        r"<[Pp]ython[^>]*>", r"<code>"):
                m = re.search(tag, text)
                if m:
                    text = text[:m.start()]
                    break
    elif "sat" in domain:
        # Look for SATISFIABLE/UNSATISFIABLE as answer boundary
        m = re.search(r"(?:UN)?SATISFIABLE", text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
    elif "math" in domain:
        pos = text.find("\\boxed{")
        if pos >= 0:
            text = text[:pos]

    return text.strip()


# ---------------------------------------------------------------------------
# Sentence-level segmentation
# ---------------------------------------------------------------------------

_SPACY_NLP = None


def _get_spacy_nlp():
    """Lazy-load a lightweight spaCy pipeline with only the sentencizer."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        _SPACY_NLP = spacy.blank("en")
        _SPACY_NLP.add_pipe("sentencizer")
    return _SPACY_NLP


def _doc_to_sentences(doc) -> List[Dict[str, Any]]:
    """Extract sentence dicts from a spaCy Doc."""
    sentences: List[Dict[str, Any]] = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        # Adjust start for leading whitespace removed by strip()
        leading = len(sent.text) - len(sent.text.lstrip())
        start = sent.start_char + leading
        end = start + len(sent_text)
        sentences.append({
            "text": sent_text,
            "char_start": start,
            "char_end": end,
        })
    return sentences


def segment_trace_sentences(text: str) -> List[Dict[str, Any]]:
    """Split reasoning text into individual sentences with char offsets.

    Uses spaCy sentence segmentation on the full text in a single call.

    Parameters
    ----------
    text : str
        Reasoning text to segment.

    Returns
    -------
    list of dict
        Each dict has ``"text"`` (str), ``"char_start"`` (int),
        ``"char_end"`` (int).
    """
    if not text or not text.strip():
        return []

    nlp = _get_spacy_nlp()
    nlp.max_length = max(nlp.max_length, len(text) + 1000)
    doc = nlp(text)
    return _doc_to_sentences(doc)


def segment_traces_sentences_batch(
    texts: List[str],
    batch_size: int = 256,
) -> List[List[Dict[str, Any]]]:
    """Batch-segment multiple texts using ``nlp.pipe()``.

    Parameters
    ----------
    texts : list of str
        Reasoning texts to segment.
    batch_size : int
        spaCy pipe batch size.

    Returns
    -------
    list of list of dict
        One sentence list per input text.
    """
    nlp = _get_spacy_nlp()
    max_len = max((len(t) for t in texts if t), default=0) + 1000
    nlp.max_length = max(nlp.max_length, max_len)

    # Replace empty/blank texts with a placeholder so pipe() indices stay
    # aligned; we'll return [] for those.
    placeholder = "."
    cleaned = [t if (t and t.strip()) else placeholder for t in texts]
    is_empty = [not (t and t.strip()) for t in texts]

    results: List[List[Dict[str, Any]]] = []
    for doc, empty in zip(nlp.pipe(cleaned, batch_size=batch_size),
                          is_empty):
        if empty:
            results.append([])
        else:
            results.append(_doc_to_sentences(doc))
    return results


def build_sentence_windows(
    sentences: List[Dict[str, Any]],
    context_size: int = 1,
) -> List[Dict[str, Any]]:
    """Build sliding context windows around each sentence.

    For each sentence at index *i*, builds a window containing up to
    *context_size* preceding and following sentences.  The target sentence
    is marked with ``>>>`` and ``<<<`` delimiters.

    Parameters
    ----------
    sentences : list of dict
        Output from ``segment_trace_sentences()``.
    context_size : int
        Number of context sentences on each side (default 1 = 3-sentence
        window).

    Returns
    -------
    list of dict
        Each dict has ``"target_text"``, ``"context_text"``,
        ``"target_char_start"``, ``"target_char_end"``,
        ``"target_sentence_idx"``.
    """
    windows: List[Dict[str, Any]] = []
    n = len(sentences)

    for i in range(n):
        parts = []
        # Preceding context
        for j in range(max(0, i - context_size), i):
            parts.append(sentences[j]["text"])
        # Target sentence with markers
        parts.append(">>> " + sentences[i]["text"] + " <<<")
        # Following context
        for j in range(i + 1, min(n, i + context_size + 1)):
            parts.append(sentences[j]["text"])

        windows.append({
            "target_text": sentences[i]["text"],
            "context_text": "\n".join(parts),
            "target_char_start": sentences[i]["char_start"],
            "target_char_end": sentences[i]["char_end"],
            "target_sentence_idx": i,
        })

    return windows


def build_system_prompt(judge_config: dict, domain: str) -> str:
    """Build the full system prompt with sentence-level instructions.

    Renders the base prompt from the judge config template and appends
    sentence-level classification instructions.

    Parameters
    ----------
    judge_config : dict
        Loaded judge YAML config.
    domain : str
        ``"code"`` or ``"math"``.

    Returns
    -------
    str
        Complete system prompt with sentence-level instructions.
    """
    base = _build_base_system_prompt(judge_config, domain)
    sentence_block = (
        "\n\n"
        "SENTENCE-LEVEL MODE: You are classifying a SINGLE sentence within a\n"
        "short context window. The target sentence is marked with >>> and <<<.\n"
        "Only classify the behavior present in the marked sentence -- the\n"
        "surrounding sentences are provided solely for context."
    )
    return base + sentence_block


# ---------------------------------------------------------------------------
# Explicit cache management
# ---------------------------------------------------------------------------

def _google_client():
    """Lazy-init Google GenAI client."""
    from google import genai
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def create_prompt_cache(
    client,
    system_prompt: str,
    model: str = "models/gemini-2.5-flash-lite",
) -> str:
    """Create an explicit CachedContent for the system prompt.

    Parameters
    ----------
    client
        Google GenAI client.
    system_prompt : str
        Rendered system prompt text (~1,100 tokens).
    model : str
        Gemini model identifier.

    Returns
    -------
    str
        Cache name (e.g. ``"cachedContents/abc123"``).
    """
    from google.genai import types as genai_types

    cache = client.caches.create(
        model=model,
        config=genai_types.CreateCachedContentConfig(
            system_instruction=system_prompt,
            ttl="86400s",
        ),
    )
    return cache.name


def delete_prompt_cache(client, cache_name: str) -> None:
    """Delete an explicit CachedContent after use."""
    try:
        client.caches.delete(name=cache_name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Batch JSONL construction
# ---------------------------------------------------------------------------

def build_judge_batch_jsonl(
    traces_data: List[Dict[str, Any]],
    judge_config: dict,
    output_dir: Path,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """Build JSONL file(s) for the Gemini Batch API.

    Each line is one segment classification request with the system
    prompt in a ``systemInstruction`` field.  If the total
    number of segments exceeds ``batch.max_segments_per_batch``,
    multiple JSONL files are written.

    Parameters
    ----------
    traces_data : list of dict
        Each dict has ``"problem_id"``, ``"run_idx"``,
        ``"segments"`` (list of segment dicts from ``segment_trace_sentences()``).
    judge_config : dict
        Loaded judge YAML config.
    output_dir : Path
        Directory to write the JSONL file(s).
    system_prompt : str
        Rendered system prompt text.

    Returns
    -------
    list of dict
        One entry per batch file, each with keys ``"jsonl_path"`` (Path),
        ``"num_requests"`` (int), ``"custom_id_map"`` (dict).
    """
    cfg = judge_config["judge"]

    # Batch JSONL requires response_json_schema (not response_schema).
    # See: github.com/googleapis/python-genai/issues/1150
    judge_schema_with_title = {
        "type": "object",
        "title": "SegmentClassification",
        "properties": JUDGE_SCHEMA["properties"],
        "required": JUDGE_SCHEMA["required"],
    }
    generation_config = {
        "max_output_tokens": cfg["max_output_tokens"],
        "response_mime_type": "application/json",
        "response_json_schema": judge_schema_with_title,
        "temperature": cfg["temperature"],
    }

    max_per_batch = judge_config["batch"].get(
        "max_segments_per_batch", 15000)

    all_items: List[Dict[str, Any]] = []
    for trace_idx, td in enumerate(traces_data):
        for seg in td["segments"]:
            seg_idx = seg["segment_idx"]
            key = f"t{trace_idx}_s{seg_idx}"
            all_items.append({
                "key": key,
                "meta": [td["problem_id"], td["run_idx"], seg_idx],
                "text": seg["text"],
            })

    output_dir.mkdir(parents=True, exist_ok=True)

    batches = []
    for batch_idx in range(0, max(len(all_items), 1), max_per_batch):
        chunk = all_items[batch_idx:batch_idx + max_per_batch]
        if not chunk:
            break

        custom_id_map: Dict[str, List] = {}
        lines: List[str] = []
        for item in chunk:
            custom_id_map[item["key"]] = item["meta"]

            request = {
                "systemInstruction": {
                    "parts": [{"text": system_prompt}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": item["text"]}],
                    }
                ],
                "generation_config": generation_config,
            }
            lines.append(json.dumps(
                {"key": item["key"], "request": request},
                ensure_ascii=False,
            ))

        file_idx = batch_idx // max_per_batch
        jsonl_path = output_dir / f"batch_input_{file_idx}.jsonl"
        with open(jsonl_path, "w") as f:
            f.write("\n".join(lines))

        batches.append({
            "jsonl_path": jsonl_path,
            "num_requests": len(lines),
            "custom_id_map": custom_id_map,
        })

    return batches


def submit_judge_batch(
    client,
    jsonl_path: Path,
    model: str,
    display_name: str,
) -> Dict[str, Any]:
    """Upload JSONL and create a Gemini batch job.

    Parameters
    ----------
    client
        Google GenAI client.
    jsonl_path : Path
        Path to the JSONL input file.
    model : str
        Gemini model identifier (without ``models/`` prefix for batch).
    display_name : str
        Human-readable label for the batch job.

    Returns
    -------
    dict
        Batch state dict for persistence.
    """
    import time as _time
    from google.genai import types as genai_types
    from google.genai.errors import ClientError

    max_retries = 5
    base_delay = 30  # seconds

    # Upload with retry
    for attempt in range(max_retries):
        try:
            uploaded_file = client.files.upload(
                file=str(jsonl_path),
                config=genai_types.UploadFileConfig(
                    mime_type="application/jsonl"),
            )
            break
        except ClientError as exc:
            if exc.code == 429 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  Rate limited on upload (attempt {attempt + 1}/"
                      f"{max_retries}), retrying in {delay}s...")
                _time.sleep(delay)
            else:
                raise

    # Create batch with retry
    for attempt in range(max_retries):
        try:
            batch_job = client.batches.create(
                model=model,
                src=uploaded_file.name,
                config=genai_types.CreateBatchJobConfig(
                    display_name=display_name,
                ),
            )
            break
        except ClientError as exc:
            if exc.code == 429 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  Rate limited on batch create (attempt "
                      f"{attempt + 1}/{max_retries}), "
                      f"retrying in {delay}s...")
                _time.sleep(delay)
            else:
                raise

    return {
        "batch_id": batch_job.name,
        "input_file_name": uploaded_file.name,
        "status": "submitted",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "poll_count": 0,
        "error": None,
    }


def poll_judge_batch(client, state: Dict[str, Any]) -> Dict[str, Any]:
    """Check status of a Gemini batch job.  Returns updated state."""
    batch_job = client.batches.get(name=state["batch_id"])
    state["poll_count"] += 1

    job_state = (batch_job.state.name
                 if hasattr(batch_job.state, "name")
                 else str(batch_job.state))

    if job_state == "JOB_STATE_SUCCEEDED":
        state["status"] = "collecting"
        state["completed_at"] = datetime.now().isoformat()
        if batch_job.dest and batch_job.dest.file_name:
            state["output_file_name"] = batch_job.dest.file_name
    elif job_state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                       "JOB_STATE_EXPIRED"):
        state["status"] = "failed"
        state["error"] = f"Batch job {job_state}"
    else:
        state["status"] = "polling"

    print(
        f"  [judge] Batch {state['batch_id'][:30]}... "
        f"state={job_state} "
        f"(poll #{state['poll_count']})",
        flush=True,
    )
    return state


def collect_judge_results(
    client,
    state: Dict[str, Any],
) -> str:
    """Download output JSONL text from a completed batch job."""
    output_file_name = state.get("output_file_name")
    if not output_file_name:
        batch_job = client.batches.get(name=state["batch_id"])
        output_file_name = batch_job.dest.file_name

    dl_result = client.files.download(file=output_file_name)
    if isinstance(dl_result, bytes):
        return dl_result.decode("utf-8")
    return str(dl_result)


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_judge_batch_results(
    output_text: str,
    custom_id_map: Dict[str, List],
) -> pd.DataFrame:
    """Parse batch output JSONL into a per-segment DataFrame.

    Parameters
    ----------
    output_text : str
        Raw JSONL text from the batch output file.
    custom_id_map : dict
        Mapping from key to [problem_id, run_idx, segment_idx].

    Returns
    -------
    pd.DataFrame
        Columns: problem_id, run_idx, segment_idx, + 6 category booleans.
    """
    records = []
    errors = 0

    for line in output_text.strip().split("\n"):
        if not line.strip():
            continue
        result = json.loads(line)
        key = result.get("key")
        meta = custom_id_map.get(key)
        if meta is None:
            continue

        problem_id, run_idx, segment_idx = meta

        response = result.get("response")
        if not response or not response.get("candidates"):
            errors += 1
            record = {
                "problem_id": problem_id,
                "run_idx": run_idx,
                "segment_idx": segment_idx,
            }
            for cat in CATEGORY_NAMES:
                record[cat] = False
            records.append(record)
            continue

        candidate = response["candidates"][0]
        content_parts = candidate.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in content_parts)

        try:
            labels = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            labels = {}

        record = {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "segment_idx": segment_idx,
        }
        for cat in CATEGORY_NAMES:
            record[cat] = bool(labels.get(cat, False))
        records.append(record)

    if errors:
        print(f"  [judge] {errors} segment(s) returned errors "
              f"(defaulted to all-false)")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_segments(
    segment_df: pd.DataFrame,
    segments_info: Dict[Tuple[str, int], int],
) -> pd.DataFrame:
    """Aggregate per-segment labels into per-trace summary statistics.

    Parameters
    ----------
    segment_df : pd.DataFrame
        Per-segment results from ``parse_judge_batch_results()``.
    segments_info : dict
        Mapping ``(problem_id, run_idx)`` -> total number of segments
        for that trace.

    Returns
    -------
    pd.DataFrame
        Per-trace DataFrame with columns: problem_id, run_idx,
        num_segments, and for each category:
        judge_{cat}_present, judge_{cat}_count, judge_{cat}_rate,
        judge_{cat}_first, judge_{cat}_last, judge_{cat}_pos_mean.
    """
    records = []
    for (pid, ridx), group in segment_df.groupby(
        ["problem_id", "run_idx"], sort=False
    ):
        n_segs = segments_info.get((pid, ridx), len(group))
        group_sorted = group.sort_values("segment_idx")

        row: Dict[str, Any] = {
            "problem_id": pid,
            "run_idx": ridx,
            "num_segments": n_segs,
        }

        for cat in CATEGORY_NAMES:
            mask = group_sorted[cat].values.astype(bool)
            count = int(mask.sum())
            row[f"judge_{cat}_present"] = count > 0
            row[f"judge_{cat}_count"] = count
            row[f"judge_{cat}_rate"] = count / n_segs if n_segs > 0 else 0.0

            if count > 0:
                indices = group_sorted["segment_idx"].values[mask]
                denom = max(n_segs - 1, 1)
                row[f"judge_{cat}_first"] = float(indices[0]) / denom
                row[f"judge_{cat}_last"] = float(indices[-1]) / denom
                row[f"judge_{cat}_pos_mean"] = (
                    float(np.mean(indices)) / denom
                )
            else:
                row[f"judge_{cat}_first"] = float("nan")
                row[f"judge_{cat}_last"] = float("nan")
                row[f"judge_{cat}_pos_mean"] = float("nan")

        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Column rename mapping (judge -> regex convention)
# ---------------------------------------------------------------------------

def build_judge_rename_map(judge_config: dict) -> dict:
    """Map judge column names to regex-convention names.

    Parameters
    ----------
    judge_config : dict
        Loaded judge YAML config (from ``load_judge_config()``).

    Returns
    -------
    dict
        E.g. ``{'judge_self_correction_rate': 'backtrack_rate', ...}``.
    """
    rename = {}
    for cat_name, cat_data in judge_config["categories"].items():
        regex_name = cat_data["regex_mapping"]
        for suffix in ("present", "count", "rate", "first", "last",
                        "pos_mean"):
            rename[f"judge_{cat_name}_{suffix}"] = f"{regex_name}_{suffix}"
    return rename


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_judge_state(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load ``judge_batch_state.json`` from *output_dir*.

    Returns None if not found or if status is ``"completed"`` or ``"failed"``.
    """
    state_path = output_dir / "judge_batch_state.json"
    if not state_path.exists():
        return None
    with open(state_path) as f:
        state = json.load(f)
    if state.get("status") in ("completed", "failed"):
        return None
    return state


def save_judge_state(state: Dict[str, Any], output_dir: Path) -> None:
    """Atomic write of ``judge_batch_state.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "judge_batch_state.json"
    fd, tmp_path = tempfile.mkstemp(
        dir=output_dir, prefix=".judge_state_", suffix=".tmp"
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
# OpenAI Batch API
# ---------------------------------------------------------------------------

OPENAI_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "SegmentClassification",
        "schema": {
            "type": "object",
            "properties": JUDGE_SCHEMA["properties"],
            "required": JUDGE_SCHEMA["required"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def _openai_client():
    """Lazy-init OpenAI client."""
    from openai import OpenAI
    return OpenAI()


def build_openai_judge_jsonl(
    traces_data: List[Dict[str, Any]],
    judge_config: dict,
    output_dir: Path,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """Build JSONL file(s) for the OpenAI Batch API.

    Each line is one segment classification request.  The system prompt
    is sent as a standard ``system`` role message.

    Parameters
    ----------
    traces_data : list of dict
        Each dict has ``"problem_id"``, ``"run_idx"``,
        ``"segments"`` (list of segment dicts from ``segment_trace_sentences()``).
    judge_config : dict
        Loaded judge YAML config.
    output_dir : Path
        Directory to write the JSONL file(s).
    system_prompt : str
        Rendered system prompt text.

    Returns
    -------
    list of dict
        One entry per batch file, each with keys ``"jsonl_path"`` (Path),
        ``"num_requests"`` (int), ``"custom_id_map"`` (dict).
    """
    cfg = judge_config["judge_openai"]
    max_per_batch = judge_config["batch_openai"].get(
        "max_requests_per_batch", 50000)
    # OpenAI Batch API limit is 100 MB per file; use 95 MB as safe cap
    max_bytes_per_batch = judge_config["batch_openai"].get(
        "max_bytes_per_batch", 95 * 1024 * 1024)

    output_dir.mkdir(parents=True, exist_ok=True)

    body_template: Dict[str, Any] = {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "response_format": OPENAI_RESPONSE_FORMAT,
    }
    if "temperature" in cfg and cfg["temperature"] is not None:
        body_template["temperature"] = cfg["temperature"]

    batches = []
    file_idx = 0
    custom_id_map: Dict[str, List] = {}
    lines: List[str] = []
    current_bytes = 0

    for trace_idx, td in enumerate(traces_data):
        for seg in td["segments"]:
            seg_idx = seg["segment_idx"]
            key = f"t{trace_idx}_s{seg_idx}"

            body = dict(body_template)
            body["messages"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": seg["text"]},
            ]

            line = json.dumps({
                "custom_id": key,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }, ensure_ascii=False)
            line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline

            # Split if adding this line would exceed size or count limits
            if lines and (current_bytes + line_bytes > max_bytes_per_batch
                          or len(lines) >= max_per_batch):
                jsonl_path = (output_dir
                              / f"batch_input_openai_{file_idx}.jsonl")
                with open(jsonl_path, "w") as f:
                    f.write("\n".join(lines))
                batches.append({
                    "jsonl_path": jsonl_path,
                    "num_requests": len(lines),
                    "custom_id_map": custom_id_map,
                })
                file_idx += 1
                custom_id_map = {}
                lines = []
                current_bytes = 0

            custom_id_map[key] = [td["problem_id"], td["run_idx"], seg_idx]
            lines.append(line)
            current_bytes += line_bytes

    # Flush remaining
    if lines:
        jsonl_path = output_dir / f"batch_input_openai_{file_idx}.jsonl"
        with open(jsonl_path, "w") as f:
            f.write("\n".join(lines))
        batches.append({
            "jsonl_path": jsonl_path,
            "num_requests": len(lines),
            "custom_id_map": custom_id_map,
        })

    return batches


def submit_openai_judge_batch(
    client,
    jsonl_path: Path,
    display_name: str,
) -> Dict[str, Any]:
    """Upload JSONL and create an OpenAI batch job.

    Parameters
    ----------
    client
        OpenAI client.
    jsonl_path : Path
        Path to the JSONL input file.
    display_name : str
        Human-readable label stored in batch metadata.

    Returns
    -------
    dict
        Batch state dict for persistence.
    """
    with open(jsonl_path, "rb") as fh:
        input_file = client.files.create(file=fh, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"display_name": display_name},
    )

    return {
        "batch_id": batch.id,
        "input_file_id": input_file.id,
        "status": "submitted",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "poll_count": 0,
        "error": None,
    }


def poll_openai_judge_batch(
    client,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Check status of an OpenAI batch job.  Returns updated state."""
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

    counts = batch.request_counts
    if counts:
        completed = counts.completed or 0
        failed = counts.failed or 0
        total = counts.total or 0
        print(
            f"  [openai-judge] Batch {state['batch_id'][:20]}... "
            f"status={batch.status} "
            f"progress={completed + failed}/{total} "
            f"(poll #{state['poll_count']})",
            flush=True,
        )

    return state


def collect_openai_judge_results(
    client,
    state: Dict[str, Any],
) -> str:
    """Download output JSONL text from a completed OpenAI batch job.

    Also downloads the error file (if present) and prints a summary of
    the first few failed requests for diagnostics.
    """
    output_file_id = state.get("output_file_id")
    if not output_file_id:
        batch = client.batches.retrieve(state["batch_id"])
        output_file_id = batch.output_file_id

    # Download and summarise error file if present
    error_file_id = state.get("error_file_id")
    if not error_file_id:
        batch = client.batches.retrieve(state["batch_id"])
        error_file_id = batch.error_file_id
    if error_file_id:
        try:
            error_content = client.files.content(error_file_id)
            error_lines = [
                l for l in error_content.text.strip().split("\n")
                if l.strip()
            ]
            if error_lines:
                print(f"  [openai-judge] Error file: {len(error_lines)} "
                      f"failed request(s)")
                for line in error_lines[:5]:
                    err = json.loads(line)
                    cid = err.get("custom_id", "?")
                    resp = err.get("response", {})
                    body = resp.get("body", {})
                    err_msg = body.get("error", {})
                    msg = err_msg.get("message", "?")
                    print(f"    {cid}: {msg[:120]}")
                if len(error_lines) > 5:
                    print(f"    ... and {len(error_lines) - 5} more")
        except Exception:
            pass

    if not output_file_id:
        raise RuntimeError(
            f"Batch {state['batch_id']} has no output file "
            "(all requests may have failed -- check errors above)."
        )

    content = client.files.content(output_file_id)
    return content.text


def parse_openai_judge_results(
    output_text: str,
    custom_id_map: Dict[str, List],
) -> pd.DataFrame:
    """Parse OpenAI batch output JSONL into a per-segment DataFrame.

    Parameters
    ----------
    output_text : str
        Raw JSONL text from the batch output file.
    custom_id_map : dict
        Mapping from custom_id to [problem_id, run_idx, segment_idx].

    Returns
    -------
    pd.DataFrame
        Columns: problem_id, run_idx, segment_idx, + 6 category booleans.
    """
    records = []
    errors = 0

    for line in output_text.strip().split("\n"):
        if not line.strip():
            continue
        result = json.loads(line)
        custom_id = result.get("custom_id")
        meta = custom_id_map.get(custom_id)
        if meta is None:
            continue

        problem_id, run_idx, segment_idx = meta

        response_body = result.get("response", {}).get("body", {})
        error_obj = result.get("error")

        if error_obj or not response_body.get("choices"):
            errors += 1
            record = {
                "problem_id": problem_id,
                "run_idx": run_idx,
                "segment_idx": segment_idx,
            }
            for cat in CATEGORY_NAMES:
                record[cat] = False
            records.append(record)
            continue

        choice = response_body["choices"][0]
        text = choice.get("message", {}).get("content", "") or ""

        try:
            labels = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            labels = {}

        record = {
            "problem_id": problem_id,
            "run_idx": run_idx,
            "segment_idx": segment_idx,
        }
        for cat in CATEGORY_NAMES:
            record[cat] = bool(labels.get(cat, False))
        records.append(record)

    if errors:
        print(f"  [openai-judge] {errors} segment(s) returned errors "
              f"(defaulted to all-false)")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Local vLLM inference
# ---------------------------------------------------------------------------

def load_judge_model(
    model_name: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
    enable_prefix_caching: bool = True,
    dtype: str = "auto",
):
    """Load a vLLM model for judge inference.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (e.g. ``"Qwen/Qwen2.5-7B-Instruct"``).
    tensor_parallel_size : int
        Number of GPUs for tensor parallelism.
    gpu_memory_utilization : float
        Fraction of GPU memory to use (0.0-1.0).
    max_model_len : int
        Maximum model context length.
    enable_prefix_caching : bool
        Enable automatic prefix caching.
    dtype : str
        Model dtype (``"auto"``, ``"bfloat16"``, ``"float16"``).

    Returns
    -------
    vllm.LLM
        Loaded vLLM engine.
    """
    import os
    import warnings
    # Suppress noisy third-party warnings:
    #   - cuda.cudart/cuda.nvrtc FutureWarning (deprecated modules)
    #   - xgrammar UserWarning (torch.tensor copy-construct)
    #   - torch inductor "Not enough SMs" warning
    warnings.filterwarnings("ignore", category=FutureWarning, module="cuda")
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="xgrammar")
    os.environ.setdefault(
        "VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault(
        "PYTHONWARNINGS",
        "ignore::FutureWarning:cuda,ignore::UserWarning:xgrammar")
    os.environ.setdefault("TORCH_LOGS", "-inductor")

    from vllm import LLM

    return LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        dtype=dtype,
    )


def parse_judge_output(
    output_text: str,
    problem_id: str,
    run_idx: int,
    segment_idx: int,
) -> Dict[str, Any]:
    """Parse a single model output into a result dict.

    With guided JSON decoding the output is guaranteed to be valid JSON,
    so this is a simple ``json.loads()`` with a safety-net fallback to
    all-false labels.

    Parameters
    ----------
    output_text : str
        Raw text from model output.
    problem_id : str
        Problem identifier.
    run_idx : int
        Run index.
    segment_idx : int
        Segment index within the trace.

    Returns
    -------
    dict
        Result with problem_id, run_idx, segment_idx, and 6 booleans.
    """
    record = {
        "problem_id": problem_id,
        "run_idx": run_idx,
        "segment_idx": segment_idx,
    }

    try:
        labels = json.loads(output_text.strip())
    except (json.JSONDecodeError, ValueError):
        labels = {}

    for cat in CATEGORY_NAMES:
        record[cat] = bool(labels.get(cat, False))

    return record


def classify_segments_batch(
    llm,
    segments: List[Dict[str, Any]],
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 128,
    use_tqdm: bool = True,
) -> List[Dict[str, Any]]:
    """Classify a batch of segments using vLLM with guided JSON decoding.

    Parameters
    ----------
    llm : vllm.LLM
        Loaded vLLM engine.
    segments : list of dict
        Each dict must have ``"problem_id"``, ``"run_idx"``,
        ``"segment_idx"``, ``"text"``.
    system_prompt : str
        Rendered system prompt text.
    temperature : float
        Sampling temperature (0 = greedy).
    max_tokens : int
        Maximum output tokens per segment.
    use_tqdm : bool
        Show vLLM's internal progress bars (default True).

    Returns
    -------
    list of dict
        One result dict per segment with problem_id, run_idx,
        segment_idx, and 6 category booleans.
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    structured_params = StructuredOutputsParams(json=JUDGE_SCHEMA)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        structured_outputs=structured_params,
    )

    # Build chat messages, skipping segments that exceed context length
    max_model_len = llm.llm_engine.model_config.max_model_len
    max_input_tokens = max_model_len - max_tokens
    tokenizer = llm.get_tokenizer()

    # Check whether the model's chat template supports a system role.
    # Gemma models do not — prepend the system prompt to the user message.
    _test_msgs = [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}]
    try:
        tokenizer.apply_chat_template(_test_msgs, add_generation_prompt=True)
        _supports_system = True
    except Exception:
        _supports_system = False
        print("  [local-judge] Model does not support system role — "
              "prepending system prompt to user message")

    messages_list = []
    valid_indices = []
    skipped = 0
    for idx, seg in enumerate(segments):
        if _supports_system:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": seg["text"]},
            ]
        else:
            msgs = [
                {"role": "user", "content": system_prompt + "\n\n" + seg["text"]},
            ]
        token_ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True)
        if len(token_ids) > max_input_tokens:
            skipped += 1
            continue
        messages_list.append(msgs)
        valid_indices.append(idx)

    if skipped:
        print(f"  [local-judge] Skipped {skipped} segment(s) exceeding "
              f"context length ({max_input_tokens} input tokens)")

    # Single vLLM call -- handles batching internally
    outputs = llm.chat(
        messages_list, sampling_params=sampling_params,
        use_tqdm=use_tqdm) if messages_list else []

    # Build output-index lookup
    output_map = {}
    for out_idx, orig_idx in enumerate(valid_indices):
        output_map[orig_idx] = out_idx

    # Parse outputs
    parse_errors = 0
    results = []
    for idx, seg in enumerate(segments):
        if idx not in output_map:
            # Skipped segment — return all-false default
            record = parse_judge_output(
                "{}", seg["problem_id"], seg["run_idx"],
                seg["segment_idx"],
            )
            results.append(record)
            continue

        output = outputs[output_map[idx]]
        output_text = output.outputs[0].text

        record = parse_judge_output(
            output_text, seg["problem_id"], seg["run_idx"],
            seg["segment_idx"],
        )

        try:
            json.loads(output_text.strip())
        except (json.JSONDecodeError, ValueError):
            parse_errors += 1

        results.append(record)

    if parse_errors:
        print(f"  [local-judge] {parse_errors} segment(s) failed JSON parse "
              f"(defaulted to all-false)")

    return results
