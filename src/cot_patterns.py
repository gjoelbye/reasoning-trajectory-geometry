"""Chain-of-thought structural pattern detection.

Provides regex-based pattern counting for reasoning behaviours observed in
LLM chain-of-thought traces: self-correction (backtracking), verification,
strategy shifting, uncertainty monitoring (hesitation), problem restatement,
and subgoal decomposition.

All patterns are applied to lowercased text.  Functions return both raw counts
and length-normalised rates (per 1000 characters) so that downstream analysis
is not confounded by trace length.

The module also provides a length-corrected repetition metric (moving-average
type-token ratio, MATTR) to replace the naive unique-word ratio used
previously.
"""

import re
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

BACKTRACK_PATTERNS: List[str] = [
    # Strong signals -- explicit self-correction
    r"that's not right",
    r"that doesn't work",
    r"that's wrong",
    r"that's incorrect",
    r"that can't be right",
    r"i made (a |an )?(error|mistake)",
    r"let me re-?think",
    r"let me reconsider",
    r"let me start over",
    r"going back to",
    r"scratch that",
    # Medium signals -- trigger word + correction context
    r"wait,?\s+(that|this|no|i|the|but|actually|let)",
    r"actually,?\s+(that|this|no|i|the|let|we|it)",
    r"no,\s+(that|this|i|we|the|it|let)",
    r"hmm,?\s+(that|this|but|wait|no|actually|let|maybe|i)",
    # Weaker signals
    r"on second thought",
    r"i need to reconsider",
    r"this approach (doesn't|won't|isn't|can't)",
    r"this (doesn't|won't|isn't|can't) work",
]

VERIFY_PATTERNS: List[str] = [
    # Explicit verification
    r"let me (verify|check|confirm|test|validate|double[\s-]?check)",
    r"let's (verify|check|confirm|test|validate)",
    r"(checking|verifying|confirming|testing|validating)\s+(this|that|the|my|our|if|whether)",
    r"sanity check",
    r"to (confirm|verify|check|make sure)",
    # Testing with examples -- implicit verification
    r"(let me|let's|i'll)\s+(try|test|plug in|substitute)\s+(this|that|it|the|with|an? example)",
    r"(for|as|take)\s+(example|instance)",
    r"if\s+(we|i)\s+(plug|substitute|put|set|use)\s",
    r"(let's|let me|i'll)\s+check\s+(with|using|for|if)",
    # Re-derivation
    r"(let me|let's|i'll)\s+(re-?derive|re-?compute|re-?calculate|recalculate)",
    r"(does|should)\s+this\s+(equal|give|produce|yield|result in|match)",
    r"which\s+(equals|gives|yields|is)\s",
]

STRATEGY_PATTERNS: List[str] = [
    # Explicit strategy switches
    r"different approach",
    r"another (way|approach|method|strategy|idea|angle)",
    r"alternatively",
    r"instead,?\s+(let me|i'll|we can|let's|of)",
    r"try\s+(a |another )?(different|new|alternative)",
    r"let me try",
    r"let's try",
    # Reframing
    r"perhaps (i |we )?(should|could|can|need to)",
    r"what if (i|we)\s",
    r"let's think about this (differently|another way|from)",
    r"(maybe|perhaps) (a |the )?(better|different|easier|simpler) (way|approach|method)",
    r"how about",
    # Abandoning current approach
    r"this (approach|method|idea|strategy|solution) (is|seems|isn't|doesn't|won't)",
    r"(scrap|abandon|drop) (this|that|the current)",
    r"(back to|return to) (the )?(drawing board|basics|square one)",
]

HESITATION_PATTERNS: List[str] = [
    r"i('m| am) not sure",
    r"i('m| am) not certain",
    r"i think (maybe|perhaps|probably)",
    r"this (might|may|could) (not |be )",
    r"(unclear|confusing|tricky|complicated|hard to tell)",
    r"i('m| am) (confused|unsure|uncertain)",
    r"not sure (if|whether|how|what|why)",
]

PROBLEM_REREAD_PATTERNS: List[str] = [
    r"(re-?read|look at|read) the (problem|question|statement|description|input|constraint)",
    r"the problem (says|states|asks|requires|mentions|specifies)",
    r"(going|go) back to the (problem|question|statement)",
    r"(wait|let me).{0,30}(problem|question|statement)\s+(says|states|asks|mentions)",
    r"(according|per) the (problem|statement|question|constraints)",
]

SUBGOAL_PATTERNS: List[str] = [
    r"first,?\s+(let's|let me|we need to|i'll|i need to|we should)",
    r"step\s+\d",
    r"break\s+(this|it|the problem)\s+(down|into)",
    r"(the|a) key (idea|insight|observation|step|thing) (is|here)",
    r"(let's|let me|we can|we need to)\s+(start|begin) (by|with)",
    r"(to solve|to find|to compute|to determine|to get)\s+this,?\s+(we|i|let)",
    r"(now|next|then),?\s+(we need|let's|let me|i'll|i need to)",
    r"our (goal|plan|strategy|approach) is",
    r"(divid\w+|split|decompos\w+|break\w*) (this|it|the problem)\s+(into|down)",
    r"(sub-?problem|sub-?task|sub-?goal|sub-?case)",
]

# All categories bundled for convenience
ALL_PATTERN_CATEGORIES: Dict[str, List[str]] = {
    "backtrack": BACKTRACK_PATTERNS,
    "verify": VERIFY_PATTERNS,
    "strategy": STRATEGY_PATTERNS,
    "hesitation": HESITATION_PATTERNS,
    "problem_reread": PROBLEM_REREAD_PATTERNS,
    "subgoal": SUBGOAL_PATTERNS,
}


# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------

def count_patterns(text: str, patterns: List[str]) -> int:
    """Count total regex matches across *patterns* in *text* (case-insensitive)."""
    text_lower = text.lower()
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text_lower))
    return total


def find_pattern_positions(text: str, patterns: List[str]) -> List[int]:
    """Return sorted list of character start positions for all matches."""
    text_lower = text.lower()
    positions = []
    for pat in patterns:
        for m in re.finditer(pat, text_lower):
            positions.append(m.start())
    positions.sort()
    return positions


def detect_all_patterns(think_text: str) -> Dict[str, int]:
    """Return raw match counts for every pattern category.

    Keys are ``"backtrack_count"``, ``"verify_count"``, etc.
    """
    return {
        f"{name}_count": count_patterns(think_text, pats)
        for name, pats in ALL_PATTERN_CATEGORIES.items()
    }


def detect_all_patterns_normalized(
    think_text: str,
    per_n_chars: int = 1000,
) -> Dict[str, float]:
    """Return counts, length-normalised rates, and position statistics.

    Rates are expressed as matches per *per_n_chars* characters of think text.
    A trace with 0 characters receives a rate of 0.

    Position statistics (normalised to 0--1 within the text):

    - ``*_first`` : position fraction of the first match (NaN if no matches)
    - ``*_last``  : position fraction of the last match (NaN if no matches)
    - ``*_pos_mean`` : mean position fraction across all matches (NaN if none)
    """
    length = max(len(think_text), 1)
    result = {}

    for name, pats in ALL_PATTERN_CATEGORIES.items():
        positions = find_pattern_positions(think_text, pats)
        count = len(positions)
        result[f"{name}_count"] = count
        result[f"{name}_rate"] = count / length * per_n_chars

        if count > 0:
            fracs = [p / length for p in positions]
            result[f"{name}_first"] = fracs[0]
            result[f"{name}_last"] = fracs[-1]
            result[f"{name}_pos_mean"] = float(np.mean(fracs))
        else:
            result[f"{name}_first"] = float("nan")
            result[f"{name}_last"] = float("nan")
            result[f"{name}_pos_mean"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Repetition / lexical diversity
# ---------------------------------------------------------------------------

def compute_repetition_score(text: str, window_size: int = 100) -> Dict:
    """Compute length-corrected repetition metrics.

    Returns
    -------
    dict with keys:
        ttr : float  -- raw type-token ratio (unique_words / total_words)
        mattr : float  -- moving-average TTR (robust to length)
        repeat_ratio : float  -- fraction of consecutive duplicate sentences
        is_repetitive : bool  -- flagged if mattr < 0.4 or repeat_ratio > 0.1
    """
    if not text or len(text) < 100:
        return {"ttr": 1.0, "mattr": 1.0, "repeat_ratio": 0.0, "is_repetitive": False}

    # Proper word tokenisation: lowercase alphabetic tokens only
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if len(words) == 0:
        return {"ttr": 1.0, "mattr": 1.0, "repeat_ratio": 0.0, "is_repetitive": False}

    ttr = len(set(words)) / len(words)

    # Moving-average type-token ratio (MATTR)
    win = min(window_size, len(words))
    ttrs = []
    for i in range(len(words) - win + 1):
        window = words[i : i + win]
        ttrs.append(len(set(window)) / win)
    mattr = float(np.mean(ttrs)) if ttrs else 1.0

    # Consecutive duplicate sentences
    sentences = re.split(r"[.!?\n]+", text)
    sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
    if len(sentences) > 1:
        consecutive_repeats = sum(
            1 for a, b in zip(sentences, sentences[1:]) if a == b
        )
        repeat_ratio = consecutive_repeats / len(sentences)
    else:
        repeat_ratio = 0.0

    is_repetitive = mattr < 0.4 or repeat_ratio > 0.1

    return {
        "ttr": round(ttr, 4),
        "mattr": round(mattr, 4),
        "repeat_ratio": round(repeat_ratio, 4),
        "is_repetitive": is_repetitive,
    }
