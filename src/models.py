"""Model loading, configuration, and prompt formatting.

Centralizes all model-specific configuration so that notebooks and scripts
import a single source of truth for per-model specs, generation parameters,
and special tokens.
"""

import torch
from transformers import AutoTokenizer
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Multi-model configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    # --- 7B-class ---
    "deepseek-r1-7b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "num_layers": 28,
        "hidden_dim": 3584,
    },
    "qwen-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "num_layers": 28,
        "hidden_dim": 3584,
        "system_prompt": "",  # suppress default "You are Qwen..." prompt
    },
    "qwen-math-7b": {
        "model_id": "Qwen/Qwen2.5-Math-7B-Instruct",
        "num_layers": 28,
        "hidden_dim": 3584,
        "system_prompt": "",
    },
    "llama-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "num_layers": 32,
        "hidden_dim": 4096,
    },
    "r1-distill-llama-8b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "num_layers": 32,
        "hidden_dim": 4096,
    },
    "gemma-9b": {
        "model_id": "google/gemma-2-9b-it",
        "num_layers": 42,
        "hidden_dim": 3584,
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "num_layers": 32,
        "hidden_dim": 4096,
    },
    # --- 14B-class ---
    "qwen-14b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "num_layers": 48,
        "hidden_dim": 5120,
        "system_prompt": "",
    },
    "deepseek-r1-14b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "num_layers": 48,
        "hidden_dim": 5120,
    },
    "phi-3.5": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "num_layers": 32,
        "hidden_dim": 3072,
    },
    "phi-4": {
        "model_id": "microsoft/phi-4",
        "num_layers": 40,
        "hidden_dim": 5120,
    },
    "phi-4-reasoning": {
        "model_id": "microsoft/Phi-4-reasoning",
        "num_layers": 40,
        "hidden_dim": 5120,
    },
    # --- 32B-class ---
    "qwq-32b": {
        "model_id": "Qwen/QwQ-32B",
        "num_layers": 64,
        "hidden_dim": 5120,
    },
    "deepseek-r1-32b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "num_layers": 64,
        "hidden_dim": 5120,
    },
    "qwen-32b": {
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "num_layers": 64,
        "hidden_dim": 5120,
        "system_prompt": "",
    },
    # --- Eval-only models (6-7B, IRT calibration, no activation extraction) ---
    "deepseek-7b-chat": {
        "model_id": "deepseek-ai/deepseek-llm-7b-chat",
        "num_layers": 30,
        "hidden_dim": 4096,
    },
    "olmo-7b": {
        "model_id": "allenai/OLMo-2-1124-7B-Instruct",
        "num_layers": 32,
        "hidden_dim": 4096,
    },
    "qwen2-7b": {
        "model_id": "Qwen/Qwen2-7B-Instruct",
        "num_layers": 28,
        "hidden_dim": 3584,
        "system_prompt": "",
    },
    "zephyr-7b": {
        "model_id": "HuggingFaceH4/zephyr-7b-beta",
        "num_layers": 32,
        "hidden_dim": 4096,
    },
    "mistral-small-24b": {
        "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "num_layers": 40,
        "hidden_dim": 5120,
    },
}

def get_model_config(version: str = "deepseek-r1-7b") -> dict:
    """Return model config dict for the given version.

    Accepts any key from ``MODEL_CONFIGS``.
    """
    if version not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model version: {version!r}. "
            f"Choose from {sorted(MODEL_CONFIGS)}"
        )
    return MODEL_CONFIGS[version]

# ---------------------------------------------------------------------------
# Special tokens for the think block
# ---------------------------------------------------------------------------

THINK_START = "<think>"
THINK_END = "</think>"

def parse_think_response(response: str) -> Tuple[str, str, str]:
    """Parse a DeepSeek-R1 response into think content and answer content.

    The chat template includes ``<think>`` as the generation prefix, so the
    decoded output of generated-only tokens typically starts mid-thought
    without the opening tag.  This function detects that case and prepends
    ``<think>`` before parsing.

    Parameters
    ----------
    response : the decoded text of the generated tokens (prompt stripped)

    Returns
    -------
    (think_content, answer_content, full_response)
        think_content : text inside the think block (empty string if absent)
        answer_content : text after ``</think>`` (or entire response if no tags)
        full_response : the normalized response with both tags present
    """
    # Normalize: if </think> present but <think> missing, the opening tag
    # was consumed as part of the chat template's generation prefix.
    if THINK_END in response and THINK_START not in response:
        response = THINK_START + response
    elif THINK_END not in response and THINK_START not in response:
        return ("", response, response)

    start = response.find(THINK_START)
    end = response.find(THINK_END)

    if start != -1 and end != -1:
        think_content = response[start + len(THINK_START):end]
        answer_content = response[end + len(THINK_END):]
    elif start != -1:
        # Opening tag but no closing -- treat rest as think content
        think_content = response[start + len(THINK_START):]
        answer_content = ""
    else:
        think_content = ""
        answer_content = response

    return (think_content, answer_content, response)


# ---------------------------------------------------------------------------
# Default generation configuration
# ---------------------------------------------------------------------------

DEFAULT_GEN_KWARGS = {
    "max_new_tokens": 32768,
    "temperature": 0.6,
    "top_p": 0.95,
    "do_sample": True,
}

def default_extraction_layers(num_layers: int) -> list:
    """Return 5 evenly spaced layer indices from 0 to the final layer."""
    if num_layers < 5:
        return list(range(num_layers))
    last = num_layers - 1
    return [i * last // 4 for i in range(5)]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _ensure_model_downloaded(model_id: str) -> None:
    """Pre-download model weights with visible progress.

    ``nnsight.LanguageModel`` calls ``from_pretrained`` internally but may
    suppress the HuggingFace download progress bars, making it look like
    the cell is stuck.  Downloading via ``huggingface_hub`` first ensures
    progress bars are displayed.  Once the files are cached, the subsequent
    ``from_pretrained`` call loads from disk instantly.
    """
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    # Quick check: if the config file is cached, the snapshot is probably
    # complete (avoids a redundant API call on every load).
    cached = try_to_load_from_cache(model_id, "config.json")
    if cached is not None and not isinstance(cached, str):
        # _CACHED_NO_EXIST sentinel -- not cached
        cached = None
    if cached is not None:
        # Weights likely cached already; snapshot_download will verify and
        # is a no-op when everything is present, but it still makes an API
        # call.  Skip it for speed.
        return
    print(f"  Downloading weights for {model_id} (this only happens once)...")
    snapshot_download(model_id)
    print("  Download complete.")


def load_model(
    model_id: str,
    device_map: str = "auto",
    dtype: Optional[torch.dtype] = None,
    load_in_8bit: bool = False,
    dispatch: bool = True,
):
    """Load a model wrapped in nnsight's LanguageModel for tracing.

    Parameters
    ----------
    model_id : HuggingFace model identifier
    device_map : device placement strategy
    dtype : weight dtype (defaults to bfloat16 if available)
    load_in_8bit : use bitsandbytes INT8 quantization (~50% memory reduction)
    dispatch : whether to dispatch the model immediately

    Returns
    -------
    nnsight.LanguageModel instance
    """
    import os
    import time

    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    dtype_name = {torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(
        dtype, str(dtype)
    )
    if load_in_8bit:
        dtype_name = f"8bit (compute: {dtype_name})"
    print(f"Loading {model_id} ({dtype_name}, device_map={device_map!r})...",
          flush=True)
    t0 = time.time()

    _ensure_model_downloaded(model_id)

    # Build quantization config if requested
    quantization_config = None
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )

    # nnsight's top-level __init__ imports its NDIF remote-execution client,
    # which pulls in ``requests`` and ``huggingface_hub.HfApi``.  On servers
    # with restricted outbound network access this can stall for minutes
    # while those libraries attempt DNS lookups or API health-checks.
    # Setting HF_HUB_OFFLINE prevents huggingface_hub from phoning home.
    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        print("  Importing nnsight...", flush=True)
        from nnsight import LanguageModel
    finally:
        # Restore the previous value so later downloads still work
        if prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline

    # Only InternLM still requires vendored code; all other models
    # (including Phi-3.5) have native transformers support in 5.x.
    # InternLM is not in MODEL_CONFIGS but load_model() accepts arbitrary
    # model IDs, so this guard is kept for forward-compatibility.
    needs_remote_code = "internlm" in model_id.lower()

    print("  Initializing model (loading weights into memory)...", flush=True)
    lm_kwargs = dict(
        device_map=device_map,
        dtype=dtype,
        dispatch=dispatch,
        trust_remote_code=needs_remote_code,
        tokenizer_kwargs={"trust_remote_code": needs_remote_code},
    )
    if quantization_config is not None:
        lm_kwargs["quantization_config"] = quantization_config
    model = LanguageModel(model_id, **lm_kwargs)

    # Some models (e.g. InternLM2) have a tokenizer vocab larger than the
    # embedding table, causing CUDA index-out-of-bounds errors during the
    # embedding lookup.  Resize the embedding layer to match the tokenizer.
    tok_vocab = len(model.tokenizer)
    emb_size = model._model.get_input_embeddings().weight.shape[0]
    if tok_vocab > emb_size:
        print(f"  Resizing embeddings: {emb_size} -> {tok_vocab} "
              f"(tokenizer has {tok_vocab - emb_size} extra tokens)")
        model._model.resize_token_embeddings(tok_vocab)

    elapsed = time.time() - t0
    print(f"  Model ready in {elapsed:.1f}s.", flush=True)
    return model


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load the tokenizer independently of the model.

    Useful for offline analysis of token sequences without loading
    the full model into memory.
    """
    needs_remote_code = "internlm" in model_id.lower()
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=needs_remote_code)


def get_layer_module(model, layer_idx: int):
    """Return the nnsight Envoy for a specific transformer layer.

    For Qwen2-based models the path is model.model.layers[i].
    """
    return model.model.layers[layer_idx]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(problem_text: str) -> str:
    """Format a Codeforces problem as a generation prompt.

    Follows the DeepSeek-R1 recommendation: no system prompt, place the
    full instruction in the user message. The model will produce a
    <think>...</think> block followed by the answer.
    """
    prompt = (
        "Solve the following competitive programming problem. "
        "Think step by step, then provide your solution as Python code.\n\n"
        f"{problem_text}"
    )
    return prompt


def format_math_generation_prompt(problem_text: str) -> str:
    r"""Format a MATH problem as a generation prompt.

    Instructs the model to think step by step and provide the final
    answer in ``\boxed{}`` format, matching the MATH benchmark convention.
    """
    prefix = (
        "Solve the following math problem. "
        "Think step by step, then provide your final answer "
        "in \\boxed{} format.\n\n"
    )
    if problem_text.startswith(prefix):
        return problem_text
    return prefix + problem_text


def format_sat_generation_prompt(problem_text: str) -> str:
    """Format a SATBench problem as a generation prompt.

    Instructs the model to reason step by step about the logical
    puzzle and clearly state SATISFIABLE or UNSATISFIABLE.
    """
    prefix = (
        "Solve the following logical reasoning puzzle. "
        "Think step by step, then clearly state whether all "
        "conditions can be satisfied simultaneously. "
        "Answer with SATISFIABLE or UNSATISFIABLE.\n\n"
    )
    if problem_text.startswith(prefix):
        return problem_text
    return prefix + problem_text


def format_chat_prompt(
    problem_text: str,
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str] = None,
    domain: str = "codeforces",
) -> str:
    """Format using the chat template if the tokenizer provides one.

    Parameters
    ----------
    problem_text : raw problem statement
    tokenizer : model tokenizer (provides the chat template)
    system_prompt : optional system message. ``None`` means don't include
        one (lets the template decide). An explicit value — even ``""`` —
        overrides the template's default system prompt.
    domain : ``"codeforces"`` (default) or ``"math"``. Selects the
        appropriate prompt formatter.

    Falls back to the raw prompt if no chat template is available.
    """
    if domain == "math":
        user_content = format_math_generation_prompt(problem_text)
    elif domain == "sat":
        user_content = format_sat_generation_prompt(problem_text)
    else:
        user_content = format_prompt(problem_text)

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": user_content,
    })
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        # Common cases: AttributeError (no chat_template),
        # jinja2.TemplateError (malformed template)
        import warnings
        warnings.warn(f"Chat template failed ({type(e).__name__}: {e}), using raw prompt")
        return user_content
