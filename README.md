# Reasoning Models Don't Just Think Longer, They Move Differently

Code for the paper.

Item Response Theory (IRT), calibrated from correctness patterns across
32 models per domain (Codeforces, Hendrycks MATH, SATBench), provides a
model-free difficulty scale. The codebase analyses how hidden-state
trajectory geometry (directness, curvature, intrinsic dimensionality) tracks
that scale before and after correcting for generation length.

Raw trajectory directness is dominated by generation length: harder problems
produce longer outputs, which appear mechanically less direct. After
correcting for length, the relationship reverses and harder problems elicit
more direct corrected trajectories in reasoning models. The effect is
clearest on code, smaller on math, and attenuated on SAT. Linear probing
alone does not separate reasoning models from their instruction-tuned
baselines; the gap is geometric.

## Installation

```bash
git clone https://github.com/gjoelbye/reasoning-trajectory-geometry
cd reasoning-trajectory-geometry
pip install -e .
```

## Repository layout

```
src/                Importable modules (IRT, trajectories, probing, judge).
scripts/            Pipeline scripts: generation, analysis, IRT, probing, judge.
configs/            YAML configs for 32 paper-aligned models x 3 domains.
notebooks/          Seven reproducibility notebooks.
data/datasets/      Input problem parquets (500 problems per domain).
data/calibration/   IRT response matrix (model_id, item_id, n_correct, n_total).
data/processed/     Pre-aggregated parquets backing notebooks 01-05.
data/results/       Causal / steering outputs backing notebooks 06-07.
figures/            Paper figures (PDF + PNG).
```

## Data

| Path | Contents |
|---|---|
| `data/datasets/selected_500.parquet` | 500 Codeforces problems (metadata and prompts) |
| `data/datasets/selected_500_math.parquet` | 500 Hendrycks MATH problems |
| `data/datasets/selected_500_sat.parquet` | 500 SATBench instances |
| `data/calibration/responses_{code,math,sat}.parquet` | (model_id, item_id, n_correct, n_total); 32 models per domain |
| `data/processed/` | Pre-aggregated parquets backing the notebooks |

Raw chain-of-thought traces and hidden-state activations for the 11
hidden-state models live on HuggingFace at
[`gjoelbye/cot-hidden-state-trajectories`](https://huggingface.co/datasets/gjoelbye/cot-hidden-state-trajectories).
The Codeforces `official_tests` column (used by
`scripts/run_cot_analysis.py` for code evaluation) is in the same dataset.

## Models

Eleven matched reasoning / baseline pairs are the subject of the trajectory
analysis. Their hidden states are released on HuggingFace.

| Paper name | YAML id |
|---|---|
| R1-Distill-Qwen-7B | pipeline/{code,math,sat}/deepseek-r1-7b |
| R1-Distill-Qwen-14B | pipeline/{code,math,sat}/deepseek-r1-14b |
| R1-Distill-Qwen-32B | pipeline/{code,math,sat}/deepseek-r1-32b |
| R1-Distill-Llama-8B | pipeline/{code,math,sat}/r1-distill-llama-8b |
| QwQ-32B | pipeline/{code,math,sat}/qwq-32b |
| Phi-4-Reasoning | pipeline/{code,math,sat}/phi-4-reasoning |
| Qwen2.5-7B-Instruct | pipeline/{code,math,sat}/qwen-7b |
| Qwen2.5-14B-Instruct | pipeline/{code,math,sat}/qwen-14b |
| Qwen2.5-32B-Instruct | pipeline/{code,math,sat}/qwen-32b |
| Llama-3.1-8B-Instruct | pipeline/{code,math,sat}/llama-8b |
| Phi-4 | pipeline/{code,math,sat}/phi-4 |

Twenty-one further models contribute correctness only, used to anchor the
IRT scale: `gemma-9b`, `phi-3.5`, `mistral-7b`, `qwen-math-7b`,
`deepseek-7b-chat`, `olmo-7b`, `qwen2-7b`, `zephyr-7b`, `claude-haiku-4.5`,
`claude-sonnet-4`, `deepseek-v3`, `gemini-2.5-{flash,flash-lite,pro}`,
`gemma-3-27b`, `gpt-4o`, `gpt-4o-mini`, `llama-3.3-70b`, `mistral-small-24b`,
`o4-mini`, `qwen-2.5-72b`.

## Reproducing the paper

Every notebook computes its tables from the shipped data. No paper numbers
are hard-coded.

| Notebook | Paper figures | Paper tables |
|---|---|---|
| `01_irt_calibration.ipynb` | `boundary_items.pdf`, `external_validation.pdf` | external validation r/ρ, 1PL vs 2PL ρ, LOO ρ stability |
| `02_main_sign_reversal.ipynb` | `sign_reversal_combined.pdf` (hero), `dimensionality_signal.pdf` | per-pair ρ_raw and ρ_perp for directness/curvature; orthogonal length decomposition |
| `03_length_correction_robustness.ipynb` | `length_correction.pdf`, `cubic_log_residuals.pdf`, `directness_prefix_sensitivity.pdf` | ρ_perp under 6 length-correction families; cross-method correlation matrix |
| `04_prompt_generation_dissociation.ipynb` | `prompt_generation_dissociation.pdf` | prompt vs generation R² per pair × domain; ΔR² vs Δρ_perp scatter |
| `05_behavioral_mediation.ipynb` | `behavioral_majority_stripe.pdf`, `temporal_directness_dynamics.pdf` | ρ(difficulty, behaviour); indirect-effect proportions per reasoning model × behaviour |
| `06_causal_validation.ipynb` | `causal_alpha_sweep.pdf` | nullspace projection (ρ_perp drop), INLP erasure, variance decomposition / mediation, think-only steering α-grid |
| `07_cross_model_steering.ipynb` | — | cross-model behaviour-direction transfer (diagnostic ρ); null active-steering effects |

Run all seven from the repo root:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_irt_calibration.ipynb --output 01_irt_calibration.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_main_sign_reversal.ipynb --output 02_main_sign_reversal.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_length_correction_robustness.ipynb --output 03_length_correction_robustness.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_prompt_generation_dissociation.ipynb --output 04_prompt_generation_dissociation.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_behavioral_mediation.ipynb --output 05_behavioral_mediation.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_causal_validation.ipynb --output 06_causal_validation.ipynb
jupyter nbconvert --to notebook --execute notebooks/07_cross_model_steering.ipynb --output 07_cross_model_steering.ipynb
```

## Regenerating from raw data

Notebooks 01-05 each contain a `## Generate analysis inputs` section that
recomputes its `data/processed/0X_*.parquet` inputs from `data/results/`
when the environment variable `IRT_RESULTS_ROOT` resolves to a populated
per-model analysis tree. With the variable unset (or pointing at a directory
without that tree) the notebooks fall back to the shipped cache. Notebooks
06-07 read their causal / steering inputs directly from the shipped
`data/results/`.

```bash
export IRT_RESULTS_ROOT=/path/to/data/results
jupyter nbconvert --to notebook --execute notebooks/01_irt_calibration.ipynb \
    --output 01_irt_calibration.ipynb
```

NB05's prefix-directness recomputation reads multi-GB
`trajectory_traces.parquet` per model and is gated behind a second flag:

```bash
export IRT_REGENERATE_HEAVY=1
```

To regenerate just the IRT response matrix from the command line:

```bash
python scripts/build_calibration_responses.py
```

Producing `data/results/` from the raw HuggingFace dataset uses the
existing per-model pipeline scripts:

```bash
# Generate CoTs + hidden states
python scripts/run_pipeline.py --config pipeline/code/deepseek-r1-7b

# Generate CoTs for an API calibration model
python scripts/run_api_generation.py --domain code --models claude-sonnet-4

# Analyse CoT structure and extract correctness
python scripts/run_cot_analysis.py --config pipeline/code/deepseek-r1-7b

# Trajectory geometry analysis (stages 03, 04)
python scripts/run_trajectory_analysis.py --config pipeline/code/deepseek-r1-7b --stages 03 04

# Pooled IRT calibration across all 32 models in the domain
python scripts/run_pooled_irt.py --domain codeforces --auto-discover --include-eval-only

# Prompt-stage probing
python scripts/run_hidden_state_probing.py --config pipeline/code/deepseek-r1-7b --mode all

# LLM judge segmentation for the behaviour analysis
python scripts/run_llm_judge.py --provider local --all
```

## Citation

```
@article{gjolbye2026reasoning,
    title         = {Reasoning Models Don't Just Think Longer, They Move Differently},
    author        = {Gj{\o}lbye, Anders and Hansen, Lars Kai and Koyejo, Sanmi},
    year          = {2026},
    journal       = {arXiv preprint arXiv:2605.15454},
    eprint        = {2605.15454},
    archivePrefix = {arXiv}
}
```

See `CITATION.cff` for the full machine-readable citation, including the three
upstream datasets (Easy2Hard-Bench, MATH, SATBench).

## License

CC-BY-SA-4.0 (see `LICENSE`).
