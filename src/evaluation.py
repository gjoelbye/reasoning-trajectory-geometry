"""Code extraction, answer extraction, and correctness evaluation for CoT traces.

Supports three evaluation modes:

**Codeforces** (code execution):
  Extracts Python code from model-generated text, validates syntax, executes
  against test cases in subprocess sandboxes, and classifies results.

**MATH** (symbolic/numerical answer comparison):
  Extracts the final ``\\boxed{}`` answer from model output (with a
  heuristic fallback for models that omit ``\\boxed{}``), normalizes
  mathematical expressions, and compares via exact string match, SymPy
  symbolic equivalence, or numerical tolerance.

**SAT** (satisfiability label extraction):
  Extracts SAT/UNSAT labels from model output by matching common
  satisfiability-related phrases and compares against ground truth.

Used by ``scripts/run_analysis.py`` and experiment notebooks 02+.
"""

import ast
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

from src.data import _is_nonempty_sequence

# ---------------------------------------------------------------------------
# XML-tag code extraction (fallback for Claude and similar models)
# ---------------------------------------------------------------------------

_XML_CODE_PATTERNS = [
    r"<answer>(.*?)</answer>",
    r"<solution>(.*?)</solution>",
    r"<[Pp]ython[^>]*>(.*?)</[Pp]ython[^>]*>",
    r"<code>(.*?)</code>",
    r"<code_block[^>]*>(.*?)</code_block>",
    r"<pre>(.*?)</pre>",
    r"<parameter[^>]*>(.*?)</parameter>",
]


def _extract_from_xml_tags(text):
    """Try XML-style tags, syntax-validated. Returns last valid match or None."""
    for pattern in _XML_CODE_PATTERNS:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content = matches[-1].strip()
            if content:
                try:
                    ast.parse(content)
                    return content
                except SyntaxError:
                    pass
    return None


def extract_python_code(text: str) -> Optional[str]:
    """Extract the LAST Python code block from the text.

    When the model iterates on code (writes, debugs, rewrites), the final
    version is most likely to be correct.  Searches for fenced code blocks
    with triple-backtick ``python`` fences first, then bare triple-backtick fences.

    Parameters
    ----------
    text : str
        The text to search (typically the answer portion of a CoT trace,
        or the full trace as fallback).

    Returns
    -------
    str or None
        The extracted code string, or None if no code block found.
    """
    # Case-insensitive: ```python, ```Python, ```PYTHON
    pattern = r"```[Pp]ython\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()  # LAST match
    # Bare fenced code blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Dashed separator blocks (e.g. phi-4-reasoning)
    pattern = r"-{10,}\s*\n(.*?)\n\s*-{10,}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Fallback: XML-style tags (Claude models wrap code in <answer>, etc.)
    return _extract_from_xml_tags(text)


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Check if code parses as valid Python.

    Parameters
    ----------
    code : str
        Python source code to validate.

    Returns
    -------
    (valid, error_message) : tuple of (bool, str)
        *valid* is True if the code parses; *error_message* is empty on
        success.
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def outputs_match(actual: str, expected: str) -> bool:
    """Flexible output comparison for competitive programming.

    Handles per-line whitespace stripping, trailing blank lines,
    case-insensitive YES/NO, and floating-point tolerance.

    Parameters
    ----------
    actual : str
        The actual stdout output from running the code.
    expected : str
        The expected output from the test case.

    Returns
    -------
    bool
        True if the outputs match within tolerance.
    """
    actual_lines = [line.strip() for line in actual.strip().splitlines()]
    expected_lines = [line.strip() for line in expected.strip().splitlines()]
    # Remove trailing empty lines
    while actual_lines and not actual_lines[-1]:
        actual_lines.pop()
    while expected_lines and not expected_lines[-1]:
        expected_lines.pop()
    if len(actual_lines) != len(expected_lines):
        return False
    for a, e in zip(actual_lines, expected_lines):
        if a == e:
            continue
        # Case-insensitive for YES/NO type answers
        if a.lower() == e.lower() and e.lower() in ("yes", "no", "true", "false"):
            continue
        # Float tolerance
        try:
            if abs(float(a) - float(e)) < 1e-6 * max(1.0, abs(float(e))):
                continue
        except (ValueError, OverflowError):
            pass
        return False
    return True


def run_code_classified(
    code: str,
    test_input: str,
    timeout: int = 5,
) -> Tuple[str, str, str]:
    """Execute code in a subprocess and classify the result.

    Parameters
    ----------
    code : str
        Python source code to execute.
    test_input : str
        String to feed to stdin.
    timeout : int
        Maximum seconds to allow execution (default 5).

    Returns
    -------
    (stdout, status, error_detail) : tuple of (str, str, str)
        *status* is one of: ``'success'``, ``'syntax_error'``,
        ``'runtime_error'``, ``'timeout'``, ``'execution_error'``.
    """
    is_valid, syntax_err = validate_python_syntax(code)
    if not is_valid:
        return "", "syntax_error", syntax_err

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python3", "-W", "ignore::SyntaxWarning", f.name],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                return result.stderr, "runtime_error", result.stderr[:200]
            return result.stdout.strip(), "success", ""
        except subprocess.TimeoutExpired:
            return "", "timeout", f"Exceeded {timeout}s"
        except Exception as e:
            return "", "execution_error", str(e)
        finally:
            Path(f.name).unlink(missing_ok=True)


def evaluate_on_tests(
    code,
    test_cases,
    timeout: int = 5,
    max_consecutive_failures: int = 3,
    max_wall_time: int = 60,
) -> Tuple[int, int, str]:
    """Run code against test cases with early-exit safeguards.

    Parameters
    ----------
    code : str or None
        Python source code to execute.
    test_cases : list of dicts
        Each dict should have ``'input'`` and ``'output'`` keys.
    timeout : int
        Per-test-case timeout in seconds (default 5).
    max_consecutive_failures : int
        Stop after this many consecutive non-success results (default 3).
    max_wall_time : int
        Stop if total wall-clock time exceeds this (default 60s).

    Returns
    -------
    (passed, total, error_type) : tuple of (int, int, str)
        *passed*: number of test cases passed.
        *total*: total test cases attempted + skipped.
        *error_type*: worst error encountered, or ``'success'``.
    """
    if not isinstance(code, str) or not _is_nonempty_sequence(test_cases):
        return 0, 0, "no_code" if not isinstance(code, str) else "no_tests"

    passed = 0
    total = 0
    skipped = 0
    worst_error = "success"
    consecutive_failures = 0
    start_time = time.time()

    for tc in test_cases:
        if not isinstance(tc, dict):
            continue
        test_input = tc.get("input", "")
        expected = tc.get("output", "").strip()
        if not expected:
            continue

        # Wall-clock guard: stop if we've spent too long on this trace
        if time.time() - start_time > max_wall_time:
            skipped += 1
            if worst_error == "success":
                worst_error = "wall_timeout"
            continue  # count remaining tests as skipped

        # Early exit: stop after N consecutive failures (timeout/error)
        if consecutive_failures >= max_consecutive_failures:
            skipped += 1
            continue  # count remaining tests as skipped

        total += 1
        actual, status, _ = run_code_classified(code, test_input, timeout)

        if status == "success" and outputs_match(actual, expected):
            passed += 1
            consecutive_failures = 0  # reset on success
        else:
            if status != "success":
                consecutive_failures += 1
                if worst_error == "success":
                    worst_error = status
            else:
                # Wrong answer (code ran fine but output didn't match)
                consecutive_failures = 0  # wrong_answer != systematic failure
                if worst_error == "success":
                    worst_error = "wrong_answer"

    if total == 0:
        return 0, 0, "no_tests"
    if passed < total and worst_error == "success":
        worst_error = "wrong_answer"

    return passed, total + skipped, worst_error


# ---------------------------------------------------------------------------
# MATH answer extraction and evaluation
# ---------------------------------------------------------------------------

def extract_math_answer(text: str) -> Optional[str]:
    r"""Extract the final answer from model output.

    Tries ``\boxed{...}`` first (handles nested braces).  When multiple
    ``\boxed{}`` groups are present, the last one is returned (the model
    may revise its answer during reasoning).

    If no ``\boxed{}`` is found, falls back to heuristic extraction from
    the last few lines of the response, looking for common answer-statement
    patterns such as "the answer is $...$" or trailing "= <value>".

    Parameters
    ----------
    text : str
        The model's full response or answer portion.

    Returns
    -------
    str or None
        The extracted answer string, or None if not found.
    """
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    return _extract_math_answer_fallback(text)


def _extract_boxed(text: str) -> Optional[str]:
    r"""Extract the last ``\boxed{...}`` from *text*."""
    pattern = r"\\boxed\{"
    starts = [m.end() for m in re.finditer(pattern, text)]
    if not starts:
        return None

    # Take the last \boxed{ and find its matching closing brace
    pos = starts[-1]
    depth = 1
    i = pos
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return text[pos:i - 1].strip()


def _extract_math_answer_fallback(text: str) -> Optional[str]:
    """Heuristic answer extraction when no ``\\boxed{}`` is present.

    Scans the last 10 non-empty lines for common answer-statement
    patterns produced by models that do not use ``\\boxed{}``.
    """
    lines = text.strip().split("\n")
    tail_lines = [l.strip() for l in lines if l.strip()][-10:]
    if not tail_lines:
        return None

    # --- 1. "is/equals $answer$" or "is/equals <number>" ---
    for line in reversed(tail_lines):
        m = re.search(
            r"(?:is|equals)\s+\$([^\$]+)\$\s*[.\s]*$",
            line, re.IGNORECASE,
        )
        if m:
            return _clean_extracted_answer(m.group(1))

        m = re.search(
            r"(?:is|equals)\s+\*?\*?(-?[0-9]+(?:[./][0-9]+)?)\*?\*?\s*[.\s]*$",
            line, re.IGNORECASE,
        )
        if m:
            return _clean_extracted_answer(m.group(1))

    # --- 2. Standalone math on the very last line ---
    last = tail_lines[-1]
    m = re.match(r"^\$\$(.+)\$\$$", last)
    if m:
        return _clean_extracted_answer(m.group(1))
    m = re.match(r"^\$([^\$]+)\$$", last)
    if m:
        return _clean_extracted_answer(m.group(1))

    # --- 3. "= <answer>" conclusion ---
    for line in reversed(tail_lines):
        m = re.search(r"=\s+\$([^\$]+)\$\s*[.\s]*$", line)
        if m:
            return _clean_extracted_answer(m.group(1))
        m = re.search(
            r"=\s+\$?(-?[0-9]+(?:[./][0-9]+)?)\$?\s*[.\s]*$",
            line,
        )
        if m:
            return _clean_extracted_answer(m.group(1))

    return None


def _clean_extracted_answer(ans: str) -> str:
    """Normalize a raw heuristically-extracted answer string."""
    ans = ans.strip()
    # Strip "LHS = RHS" keeping only RHS
    eq_parts = ans.rsplit("=", 1)
    if len(eq_parts) == 2:
        rhs = eq_parts[1].strip()
        if rhs and not rhs.startswith(("=", "<", ">")):
            ans = rhs
    # Strip leading variable assignment: "k = 2" -> "2"
    m = re.match(r"^[a-zA-Z_]\w*\s*=\s*(.+)$", ans)
    if m:
        ans = m.group(1).strip()
    # Normalize \dfrac -> \frac
    ans = ans.replace(r"\dfrac", r"\frac")
    # Strip trailing period
    ans = ans.rstrip(".")
    return ans


def normalize_math_expression(expr: str) -> str:
    r"""Normalize a mathematical expression for comparison.

    Strips whitespace, normalizes common LaTeX commands, and attempts
    SymPy simplification when possible.

    Parameters
    ----------
    expr : str
        A mathematical expression (possibly in LaTeX notation).

    Returns
    -------
    str
        The normalized expression string.
    """
    # Basic string normalization
    s = expr.strip()
    # Remove \left and \right delimiters
    s = s.replace(r"\left", "").replace(r"\right", "")
    # Normalize common formatting
    s = s.replace(r"\,", "").replace(r"\;", "").replace(r"\!", "")
    s = s.replace(r"\quad", " ").replace(r"\qquad", " ")
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def evaluate_math_answer(
    predicted: str,
    ground_truth: str,
    tolerance: float = 1e-6,
) -> bool:
    r"""Compare predicted answer to ground truth using multiple strategies.

    Checks in order:
    1. Exact string match after normalization.
    2. Symbolic equivalence via SymPy (if both parse as valid expressions).
    3. Numerical equivalence within tolerance.

    Parameters
    ----------
    predicted : str
        The model's predicted answer (extracted from ``\boxed{}``).
    ground_truth : str
        The reference answer from the MATH dataset.
    tolerance : float
        Relative tolerance for numerical comparison (default 1e-6).

    Returns
    -------
    bool
        True if any comparison strategy succeeds.
    """
    pred_norm = normalize_math_expression(predicted)
    gt_norm = normalize_math_expression(ground_truth)

    # 1. Exact string match
    if pred_norm == gt_norm:
        return True

    # 2. SymPy symbolic equivalence
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        pred_sym = parse_latex(pred_norm)
        gt_sym = parse_latex(gt_norm)

        if sympy.simplify(pred_sym - gt_sym) == 0:
            return True
    except Exception:
        pass

    # 3. Numerical equivalence
    try:
        pred_val = float(pred_norm)
        gt_val = float(gt_norm)
        if abs(pred_val - gt_val) < tolerance * max(1.0, abs(gt_val)):
            return True
    except (ValueError, OverflowError):
        pass

    return False


# ---------------------------------------------------------------------------
# SAT answer extraction and evaluation
# ---------------------------------------------------------------------------

def extract_sat_answer(text: str) -> Optional[str]:
    """Extract a SAT/UNSAT label from model output.

    Searches for explicit satisfiability labels in the text.
    Checks for "UNSATISFIABLE" before "SATISFIABLE" to avoid
    false matches (since "SATISFIABLE" is a substring of
    "UNSATISFIABLE").

    Parameters
    ----------
    text : str
        The model's response text.

    Returns
    -------
    str or None
        ``"sat"`` or ``"unsat"``, or None if no label found.
    """
    if not text:
        return None

    # Normalize for searching (case-insensitive)
    lower = text.lower()

    # Check UNSATISFIABLE first (longer match takes priority)
    unsat_patterns = [
        "unsatisfiable",
        "unsat",
        "is not satisfiable",
        "cannot be satisfied",
        "cannot all be satisfied",
        "not all .* can be satisfied",
        "no valid assignment",
        "no solution exists",
        "impossible to satisfy",
    ]
    sat_patterns = [
        "satisfiable",
        "is satisfiable",
        "can be satisfied",
        "can all be satisfied",
        "the answer is sat",
        "all conditions can be satisfied",
    ]

    # Search for UNSAT patterns first
    for pattern in unsat_patterns:
        if re.search(pattern, lower):
            return "unsat"

    # Then SAT patterns
    for pattern in sat_patterns:
        if re.search(pattern, lower):
            return "sat"

    return None


def evaluate_sat_answer(
    predicted: Optional[str],
    ground_truth_satisfiable: bool,
) -> bool:
    """Compare extracted SAT/UNSAT label to ground truth.

    Parameters
    ----------
    predicted : str or None
        ``"sat"`` or ``"unsat"`` (from ``extract_sat_answer``).
    ground_truth_satisfiable : bool
        True if the problem is satisfiable.

    Returns
    -------
    bool
        True if the prediction matches ground truth.
    """
    if predicted is None:
        return False
    if ground_truth_satisfiable:
        return predicted == "sat"
    return predicted == "unsat"
