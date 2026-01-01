"""Hidden-state extraction from transformer models during generation.

Provides two modes of extraction:
  1. Single forward pass: extracts activations for a fixed input sequence
     using nnsight's tracing API.
  2. Generation-time: extracts the hidden state of the newly generated token
     at each autoregressive step using PyTorch forward hooks on the
     underlying HuggingFace model.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.models import (
    get_layer_module,
    THINK_START,
    THINK_END,
)


# ---------------------------------------------------------------------------
# Single forward pass extraction
# ---------------------------------------------------------------------------

def extract_hidden_states_single_forward(
    model,
    input_text: str,
    layer_indices: List[int],
) -> Dict[int, torch.Tensor]:
    """Extract hidden states from a single forward pass (no generation).

    Parameters
    ----------
    model : nnsight.LanguageModel
    input_text : the prompt string
    layer_indices : which layers to extract from

    Returns
    -------
    dict mapping layer_index -> Tensor of shape (seq_len, hidden_dim)
    """
    saved = {}
    with model.trace(input_text):
        for idx in layer_indices:
            layer = get_layer_module(model, idx)
            # layer.output is a tuple; the first element is the hidden state
            saved[idx] = layer.output[0].save()

    return {idx: saved[idx].squeeze(0).detach().cpu() for idx in layer_indices}


# ---------------------------------------------------------------------------
# Generation-time extraction
# ---------------------------------------------------------------------------

def extract_hidden_states_during_generation(
    model,
    input_text: str,
    layer_indices: List[int],
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> Tuple[Dict[int, List[torch.Tensor]], List[int]]:
    """Extract hidden states at each generation step.

    At each autoregressive step, the hidden state of the newly generated
    token (the last position) is saved from each specified layer.

    Uses PyTorch forward hooks on the underlying HuggingFace model rather
    than nnsight's generation tracing, which has scoping limitations that
    prevent reliable variable access after the tracing context exits.

    Parameters
    ----------
    model : nnsight.LanguageModel
    input_text : prompt string
    layer_indices : which layers to extract from
    max_new_tokens : generation length limit
    temperature : sampling temperature
    top_p : nucleus sampling threshold

    Returns
    -------
    hidden_states : dict mapping layer_index -> list of Tensors,
        each of shape (hidden_dim,), one per generated token
    generated_token_ids : list of integer token ids
    """
    collected: Dict[int, List[torch.Tensor]] = {idx: [] for idx in layer_indices}

    # Register a forward hook on each target transformer layer.
    hooks = []
    for idx in layer_indices:
        layer_module = model._model.model.layers[idx]

        def _hook(module, input, output, _idx=idx):
            # Qwen2DecoderLayer returns a plain tensor; other architectures
            # may return a tuple whose first element is the hidden state.
            h = output if isinstance(output, torch.Tensor) else output[0]
            h = h[:, -1, :].detach().cpu()       # last position, move to CPU
            collected[_idx].append(h.squeeze(0))  # (hidden_dim,)

        hooks.append(layer_module.register_forward_hook(_hook))

    # Tokenize and generate using the underlying HF model directly.
    tokenizer = model.tokenizer
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model._model.device)
    attention_mask = inputs["attention_mask"].to(model._model.device)

    with torch.no_grad():
        outputs = model._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    # Clean up hooks.
    for hook in hooks:
        hook.remove()

    # The first hook call per layer is the prefill pass (processes all input
    # tokens at once).  Drop it so that the remaining entries align 1:1 with
    # the generated tokens.
    hidden_states = {idx: collected[idx][1:] for idx in layer_indices}

    generated_ids = outputs[0][input_ids.shape[-1]:].tolist()

    return hidden_states, generated_ids


# ---------------------------------------------------------------------------
# Think-block boundary detection
# ---------------------------------------------------------------------------

def find_think_boundaries(
    token_ids: List[int],
    tokenizer,
) -> Tuple[Optional[int], Optional[int]]:
    """Find the token-level start and end of the <think>...</think> block.

    When using the DeepSeek-R1 chat template with ``add_generation_prompt=True``,
    the ``<think>`` tag is part of the prompt and therefore absent from the
    generated ``token_ids``.  In that case the entire generated sequence starts
    inside the think block, so ``start_idx = 0``.

    Optimized version: Uses batch decoding and binary search instead of
    token-by-token decoding for 2-10x speedup.

    Parameters
    ----------
    token_ids : list of token ids from generation
    tokenizer : the model's tokenizer

    Returns
    -------
    (start_idx, end_idx) : token indices where the think block starts
        and ends, or (None, None) if neither tag is found.
    """
    text = tokenizer.decode(token_ids)
    think_start_char = text.find(THINK_START)
    think_end_char = text.find(THINK_END)

    # If <think> is absent but </think> is present, the opening tag was
    # consumed by the chat template.  Treat token 0 as the start.
    implicit_start = False
    if think_start_char == -1 and think_end_char != -1:
        implicit_start = True
    elif think_start_char == -1:
        return None, None

    # Build cumulative character lengths using batch decoding
    # This is much faster than decoding one token at a time
    token_texts = [tokenizer.decode([tid]) for tid in token_ids]
    token_lengths = [len(t) for t in token_texts]
    cumulative_lengths = np.cumsum(token_lengths)

    # Use binary search to find token boundaries
    start_tok = 0 if implicit_start else None
    end_tok = None

    if not implicit_start and think_start_char != -1:
        # Find first token where cumulative length > think_start_char
        start_tok = int(np.searchsorted(cumulative_lengths, think_start_char, side='right'))

    if think_end_char != -1:
        # Find first token where cumulative length > think_end_char
        end_tok = int(np.searchsorted(cumulative_lengths, think_end_char, side='right'))

    return start_tok, end_tok


def extract_think_block_states(
    hidden_states: Dict[int, List[torch.Tensor]],
    token_ids: List[int],
    tokenizer,
) -> Dict[int, List[torch.Tensor]]:
    """Extract only the hidden states within the <think>...</think> block.

    Parameters
    ----------
    hidden_states : dict from ``extract_hidden_states_during_generation``
    token_ids : generated token ids
    tokenizer : the model's tokenizer

    Returns
    -------
    dict mapping layer_index -> list of Tensors for tokens inside the think block
    """
    start, end = find_think_boundaries(token_ids, tokenizer)
    if start is None:
        return hidden_states  # no think block found, return all

    if end is None:
        end = len(token_ids) - 1

    result = {}
    for layer_idx, states in hidden_states.items():
        result[layer_idx] = states[start:end + 1]
    return result


# ---------------------------------------------------------------------------
# Selective extraction at specific positions
# ---------------------------------------------------------------------------

def extract_at_positions(
    hidden_states: Dict[int, List[torch.Tensor]],
    positions: List[int],
) -> Dict[int, torch.Tensor]:
    """Select hidden states at specific token positions.

    Parameters
    ----------
    hidden_states : dict mapping layer_index -> list of Tensors
    positions : list of token-position indices to select

    Returns
    -------
    dict mapping layer_index -> Tensor of shape (len(positions), hidden_dim)
    """
    result = {}
    for layer_idx, states in hidden_states.items():
        selected = [states[p] for p in positions if p < len(states)]
        if selected:
            result[layer_idx] = torch.stack(selected)
    return result


def subsample_positions(
    n_tokens: int,
    n_samples: int = 100,
    include_first_last: bool = True,
) -> List[int]:
    """Generate a set of uniformly spaced token positions for extraction.

    Useful when storing all token positions is too expensive.

    Parameters
    ----------
    n_tokens : total number of tokens in the sequence
    n_samples : desired number of positions
    include_first_last : whether to always include position 0 and n_tokens-1
    """
    if n_tokens <= n_samples:
        return list(range(n_tokens))

    positions = np.linspace(0, n_tokens - 1, n_samples, dtype=int).tolist()

    if include_first_last:
        if 0 not in positions:
            positions = [0] + positions
        if n_tokens - 1 not in positions:
            positions.append(n_tokens - 1)

    return sorted(set(positions))
