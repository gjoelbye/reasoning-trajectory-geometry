"""Model configuration loading and path resolution.

Loads per-model YAML config files from the ``configs/`` directory and resolves
all relative paths against ``PROJECT_ROOT``.  Provides a single source of
truth for model metadata and file locations across analysis scripts,
``run_pipeline.py``, and the experiment notebooks.

Supports three config categories:
  - **Pipeline configs** (``configs/pipeline/code/``, ``configs/pipeline/math/``):
    full pipeline models with activations and probing.
  - **Local configs** (``configs/local/``): locally-run eval-only models.
    Set ``model.eval_only: true``.
  - **API configs** (``configs/api/``): API-based eval-only models.
    Set ``model.eval_only: true``.

Usage
-----
    from src.config import load_config, load_all_configs, list_configs

    cfg = load_config("pipeline/code/deepseek-r1-7b")  # Codeforces domain
    cfg = load_config("pipeline/math/deepseek-r1-7b")  # MATH domain
    cfg = load_config("api/code/claude-sonnet-4")       # API eval-only model
    cfg = load_config("configs/custom.yaml")             # by explicit path

    # Paths (all absolute Path objects after loading)
    cfg["paths"]["problems"]                   # shared input
    cfg["paths"]["pipeline"]["cot_traces"]      # CoT traces from run_pipeline.py
    cfg["paths"]["analysis"]["cot_analysis"]   # analysis output in data/{domain}/{model}/

    # Model metadata
    cfg["model"]["hidden_dim"]                 # 3584
    cfg["model"]["num_layers"]                 # 28

    # Bulk loading
    all_cfgs = load_all_configs(domain="code", include_eval_only=True)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Externally configurable roots for large outputs and per-model analysis
# artefacts. YAML configs reference these as ``${IRT_OUTPUTS_ROOT}/...``;
# ``_to_absolute`` expands the variables via ``os.path.expandvars``. The
# defaults keep everything inside the repo so a fresh clone works without
# any environment setup.
os.environ.setdefault("IRT_OUTPUTS_ROOT", str(PROJECT_ROOT / "outputs"))
os.environ.setdefault("IRT_RESULTS_ROOT", str(PROJECT_ROOT / "data" / "results"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(model_or_path: Union[str, Path]) -> dict:
    """Load a model config YAML and resolve all paths.

    Parameters
    ----------
    model_or_path : str or Path
        Either a config name (e.g. ``"code/deepseek-r1-7b"``), which
        resolves to the corresponding YAML file under ``configs/``, or
        an explicit path to a YAML file (absolute or relative to the
        project root).

    Returns
    -------
    dict
        Parsed config with every path value replaced by an absolute
        :class:`~pathlib.Path`.

    Raises
    ------
    FileNotFoundError
        If the resolved config file does not exist.
    ValueError
        If required keys are missing from the YAML.
    """
    config_path = _resolve_config_path(model_or_path)

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    _validate(cfg, config_path)
    _resolve_paths(cfg)
    return cfg


def list_configs(include_eval_only: bool = False) -> list[str]:
    """Return sorted names of all available config files.

    Configs in subdirectories include their prefix,
    e.g. ``"pipeline/code/deepseek-r1-7b"``, ``"api/math/gpt-4o"``.

    Parameters
    ----------
    include_eval_only : bool
        If True, also include configs from ``configs/local/`` and
        ``configs/api/``.  Default is False (pipeline configs only).
    """
    results = []
    for p in CONFIGS_DIR.rglob("*.yaml"):
        rel = p.relative_to(CONFIGS_DIR)
        name = str(rel.with_suffix(""))
        # Skip eval_only configs unless explicitly requested
        if name.startswith(("local/", "api/")) and not include_eval_only:
            continue
        results.append(name)
    return sorted(results)


def load_all_configs(
    domain: Optional[str] = None,
    include_eval_only: bool = False,
) -> Dict[str, dict]:
    """Load all configs, optionally filtered by domain.

    Parameters
    ----------
    domain : str or None
        If provided, only return configs whose name starts with this
        domain prefix (e.g. ``"code"``, ``"math"``).  Matches both
        ``"pipeline/code/..."`` and ``"local/code/..."`` / ``"api/code/..."``.
    include_eval_only : bool
        If True, also include configs from ``configs/local/`` and
        ``configs/api/``.

    Returns
    -------
    dict
        Mapping of config name → parsed config dict.
    """
    configs = {}
    for name in list_configs(include_eval_only=include_eval_only):
        if domain is not None:
            # Match "pipeline/code/...", "local/code/...", "api/code/...", etc.
            parts = name.split("/")
            config_domain = parts[1] if parts[0] in ("pipeline", "local", "api") else parts[0]
            if config_domain != domain:
                continue
        try:
            configs[name] = load_config(name)
        except (FileNotFoundError, ValueError) as exc:
            import warnings
            warnings.warn(f"Skipping config {name!r}: {exc}")
    return configs


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _resolve_config_path(model_or_path: Union[str, Path]) -> Path:
    """Turn a model name or partial path into an absolute config file path."""
    p = Path(model_or_path)

    # Bare name without a YAML extension → configs/{name}.yaml
    # Note: we check for .yaml/.yml explicitly rather than using ``p.suffix``
    # because model names like "phi-3.5" have a spurious suffix (".5").
    if p.suffix not in (".yaml", ".yml"):
        p = CONFIGS_DIR / f"{model_or_path}.yaml"

    # Relative path → resolve against project root
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    if not p.exists():
        available = list_configs()
        raise FileNotFoundError(
            f"Config not found: {p}\n"
            f"Available configs: {available}"
        )
    return p


def _validate(cfg: dict, path: Path) -> None:
    """Ensure all required keys are present.

    When ``model.eval_only`` is ``True``, activation-dependent paths
    (``paths.pipeline.activations``, ``paths.analysis.trajectories``,
    ``paths.analysis.probe_results``) are NOT required.
    """
    # model.*
    for key in ("name", "model_id", "hidden_dim", "num_layers"):
        if key not in cfg.get("model", {}):
            raise ValueError(f"Missing model.{key} in {path}")

    eval_only = cfg.get("model", {}).get("eval_only", False)

    # paths.*
    paths = cfg.get("paths", {})
    if "problems" not in paths:
        raise ValueError(f"Missing paths.problems in {path}")
    for section in ("pipeline", "analysis"):
        if section not in paths:
            raise ValueError(f"Missing paths.{section} in {path}")

    # paths.pipeline required keys
    pipeline_required = ["cot_traces"]
    if not eval_only:
        pipeline_required.append("activations")
    for key in pipeline_required:
        if key not in paths.get("pipeline", {}):
            raise ValueError(f"Missing paths.pipeline.{key} in {path}")

    # paths.analysis required keys
    analysis_always = ("cot_analysis",)
    analysis_full_only = ("trajectories", "probe_results", "trajectory_traces")
    for key in analysis_always:
        if key not in paths.get("analysis", {}):
            raise ValueError(f"Missing paths.analysis.{key} in {path}")
    if not eval_only:
        for key in analysis_full_only:
            if key not in paths.get("analysis", {}):
                raise ValueError(f"Missing paths.analysis.{key} in {path}")

    # generation.* (optional section, validated when present)
    gen = cfg.get("generation", {})
    if "num_runs" in gen:
        if not isinstance(gen["num_runs"], int) or gen["num_runs"] < 1:
            raise ValueError(
                f"generation.num_runs must be a positive integer, "
                f"got {gen['num_runs']!r} in {path}"
            )
    if "max_new_tokens" in gen:
        if not isinstance(gen["max_new_tokens"], int) or gen["max_new_tokens"] < 1:
            raise ValueError(
                f"generation.max_new_tokens must be a positive integer, "
                f"got {gen['max_new_tokens']!r} in {path}"
            )
        if gen["max_new_tokens"] > 65536:
            import warnings
            warnings.warn(
                f"generation.max_new_tokens={gen['max_new_tokens']} in {path} "
                f"is very large (>65536). This may cause OOM errors."
            )


def _resolve_paths(cfg: dict) -> None:
    """Convert all path strings to absolute ``Path`` objects.

    - ``paths.problems``: relative to PROJECT_ROOT
    - ``paths.pipeline.*``: absolute or relative to PROJECT_ROOT
    - ``paths.analysis.*``: relative to PROJECT_ROOT

    Keys that are absent (e.g. ``activations`` in eval-only configs) are
    silently skipped.
    """
    paths = cfg["paths"]

    # Shared input
    paths["problems"] = _to_absolute(paths["problems"])

    # Pipeline output paths (may be absolute or relative)
    for key in list(paths["pipeline"]):
        paths["pipeline"][key] = _to_absolute(paths["pipeline"][key])

    # Analysis output paths (relative to project root)
    for key in list(paths["analysis"]):
        paths["analysis"][key] = _to_absolute(paths["analysis"][key])


def _to_absolute(path_str: str) -> Path:
    """Resolve a path string to an absolute ``Path``.

    Expands ``${VAR}`` references (e.g. ``${IRT_OUTPUTS_ROOT}``) via
    ``os.path.expandvars`` first, then resolves the result relative to
    ``PROJECT_ROOT`` if it is not already absolute.
    """
    p = Path(os.path.expandvars(path_str))
    return p if p.is_absolute() else PROJECT_ROOT / p
