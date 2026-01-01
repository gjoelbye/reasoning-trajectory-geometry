# Datasheet: CoT Hidden-State Trajectories

Chain-of-thought traces and generation-time hidden-state activations from
11 open-weight language models, on Codeforces (competitive programming),
Hendrycks MATH, and SATBench (Boolean satisfiability).

This is the datasheet for the trajectory archive accompanying the paper
*Reasoning Models Don't Just Think Longer, They Move Differently*. The
archive is prepared for public release; its host identifier is withheld
during anonymous review. The paper asks whether reasoning-trained models
follow different hidden-state paths than matched instruction-tuned
baselines, after correcting for the fact that harder problems also elicit
longer generations. The archive provides the raw material: paired text
traces and sampled activations from six matched reasoning/baseline pairs
across three reasoning domains.

The analysis code in this repository reproduces every table and figure
from pre-aggregated files under `data/`, without needing the archive.

## Quick start

Once released, the tabular parts load through the `datasets` library and
the hidden states download as files (`<archive-id>` stands in for the
identifier withheld during review):

```python
from datasets import load_dataset

# Tabular: chain-of-thought traces
cot = load_dataset("<archive-id>", "codeforces_cot")
print(cot["train"][0]["trace"][:200])

# Tabular: problems (one row per problem_id)
probs = load_dataset("<archive-id>", "codeforces_problems")

# Hidden states: HDF5 files, downloaded as files (not a split)
from huggingface_hub import hf_hub_download
import h5py

path = hf_hub_download(
    repo_id="<archive-id>",
    repo_type="dataset",
    filename="data/hendrycks_math/qwen-7b/activations.h5",
)
with h5py.File(path) as f:
    print(dict(f.attrs))
    arr = f["test__algebra__1025.json/run_0/layer_13"][:]
    print(arr.shape, arr.dtype)
```

The release ships a helper that opens the sharded hero file
(`codeforces/deepseek-r1-7b`) transparently.

## Findings (from the paper)

- Without correcting for generation length, harder problems look less
  organized in hidden-state trajectories. That is mostly a length
  artifact: longer trajectories are mechanically less direct, and
  harder problems get longer trajectories. It is a confound, not a
  finding.
- Once trajectory statistics are residualized on length, the
  relationship reverses across all three domains. Harder problems
  produce more direct corrected trajectories.
- Length-corrected geometry separates reasoning models from matched
  instruction-tuned baselines most clearly on Codeforces (reasoning
  median directness-difficulty correlation +0.41 vs baseline -0.06).
  The separation is weaker on Hendrycks MATH and weakest on SATBench,
  where baselines also show positive corrected coupling.
- Prompt-stage linear probes do not mirror the Codeforces separation.
  Whatever distinguishes reasoning models is in the unfolding
  generation, not in their representation before they start writing.
- Independent sentence-level annotations of the traces show that the
  models with stronger geometric coupling also do more
  strategy-shifting and more uncertainty monitoring during their
  reasoning.

## Models

Six matched pairs across the Qwen, Llama, and Phi families:

| Reasoning model | Baseline | Family |
|---|---|---|
| R1-Distill-Qwen-7B (`deepseek-r1-7b`) | Qwen2.5-7B-Instruct (`qwen-7b`) | Qwen 2.5 |
| R1-Distill-Qwen-14B (`deepseek-r1-14b`) | Qwen2.5-14B-Instruct (`qwen-14b`) | Qwen 2.5 |
| R1-Distill-Qwen-32B (`deepseek-r1-32b`) | Qwen2.5-32B-Instruct (`qwen-32b`) | Qwen 2.5 |
| QwQ-32B (`qwq-32b`) | Qwen2.5-32B-Instruct (`qwen-32b`) | Qwen 2.5 |
| R1-Distill-Llama-8B (`r1-distill-llama-8b`) | Llama-3.1-8B-Instruct (`llama-8b`) | Llama |
| Phi-4-Reasoning (`phi-4-reasoning`) | Phi-4 (`phi-4`) | Phi |

Qwen2.5-32B-Instruct appears twice as the shared baseline for two
reasoning models.

## Contents

Three artifact families, all keyed by `problem_id`:

| Artifact | Path | Per cell |
|---|---|---|
| Problems | `problems/<domain>.parquet` | 500 rows; codeforces 7 cols, hendrycks_math 6, satbench 7 |
| CoT traces | `data/<domain>/<model>/cot.parquet` | 5 runs per problem (30 for `codeforces/deepseek-r1-7b`); 16 cols |
| Hidden states | `data/<domain>/<model>/activations.h5` | 5 evenly-spaced layers, stride 10 tokens, float16 |

```
problems/
├── codeforces.parquet
├── hendrycks_math.parquet
└── satbench.parquet
data/
├── codeforces/<model>/{cot.parquet, activations.h5}
├── hendrycks_math/<model>/{cot.parquet, activations.h5}
└── satbench/<model>/{cot.parquet, activations.h5}
```

11 models x 3 domains = 33 cells. Each cell holds one CoT parquet and
one (or four sharded) HDF5 file.

### CoT parquet schema

16 columns: `problem_id`, `run_idx`, `rating`, `seed`, `prompt`,
`trace`, `reasoning`, `response`, `has_think_tags`, `truncated`,
`correct`, `trace_length_chars`, `generation_time_seconds`, `model`,
`model_hf_id`, `domain`.

- `prompt` is the chat-template-applied string the model actually saw,
  so it varies by model. For a canonical, model-agnostic prompt, use
  `formatted_prompt` in the matching problems parquet.
- `trace` is the raw generated text. `reasoning` and `response` are the
  same text split at `<think>...</think>`. `reasoning` is empty if no
  think block was parsed.
- `has_think_tags` is the literal substring check on `<think>`.
  `truncated` flags traces that hit (or came within 5% of) the
  `max_new_tokens` cap (heuristic: 4 chars per token).
- `correct` is the pre-graded boolean correctness, joined from the
  internal `cot_analysis` table by `(problem_id, run_idx)`. Lets users
  compute per-model pass rates without re-grading.
- `model_hf_id` is the full HuggingFace model ID (e.g. `Qwen/QwQ-32B`).

### Problems parquet schemas

All three share `problem_id` (joins to CoT `problem_id`),
`pooled_difficulty` (the paper's IRT-calibrated difficulty per
problem), `formatted_prompt` (the canonical model-agnostic prompt),
and `domain`. Beyond those, each domain carries its native difficulty
signal and the ground truth needed to grade model output:

- **codeforces** (7 columns): `problem_id`, `unnorm_rating` (raw
  Codeforces Glicko-2 ELO), `pooled_difficulty`, `quintile`
  (paper-selection stratum, 1 to 5), `official_tests` (full test
  suite, list of `struct{input, output}`), `formatted_prompt`,
  `domain`.
- **hendrycks_math** (6 columns): `problem_id`, `level_int` (MATH
  level, 1 to 5), `pooled_difficulty`, `answer` (boxed ground-truth
  answer), `formatted_prompt`, `domain`.
- **satbench** (7 columns): `problem_id`, `num_clauses` (literal CNF
  clause count, the paper's SAT difficulty signal), `pooled_difficulty`,
  `num_clauses_bin` (paper-selection stratum, 1 to 5), `satisfiable`
  (ground-truth SAT/UNSAT label), `formatted_prompt`, `domain`.

### Hidden-state files

Each `activations.h5` carries self-describing root attributes:
`dataset`, `model`, `model_hf_id`, `hidden_dim`, `num_layers`,
`layers`, `stride`, `n_runs`, `seed`, `temperature`, `top_p`,
`max_new_tokens`, `dtype`, `license`. Groups are nested as
`<problem_id>/run_<idx>/layer_<idx>`, with each dataset of shape
`(n_sampled_tokens, hidden_dim)` in float16.

The hero cell `codeforces/deepseek-r1-7b` is 665 GB at 30 runs, which
exceeds HuggingFace's 500 GB per-file limit. It ships as four shards
(`activations-00001-of-00004.h5` through `activations-00004-of-00004.h5`)
with a sibling `activations.h5.index.json` mapping each `problem_id` to
its shard filename. The format matches HuggingFace's safetensors
sharding convention.

## Sizes

| Domain | Hidden-state files | Total |
|---|---:|---:|
| `codeforces` | 11 (including 4 hero shards) | 1.5 TB |
| `hendrycks_math` | 11 | 249 GB |
| `satbench` | 11 | 796 GB |
| All HDF5 | 33 cells (35 files counting shards) | ~2.5 TB |
| All parquets | 36 | ~500 MB |

## Generation

Every model used the same decoding parameters: temperature 0.6,
top-p 0.95, max 32,768 new tokens, with a fixed seed per (problem, run).
Hidden states were captured by forward hooks at 5 evenly-spaced layers,
sampled every 10th generated token, in float16.

## Known limitations

- CoT traces are verbatim model outputs and were not sanitized. A few
  `phi-4-reasoning` traces contain placeholder author emails copied
  from training data.
- Generation was capped at 32,768 new tokens. The R1-distilled models
  hit this cap on roughly 2 percent of the hardest Codeforces problems.

## Source datasets

- **Codeforces problems**: [`furonghuang-lab/Easy2Hard-Bench`](https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench) (CC-BY-SA-4.0), with extra columns merged from [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces) (CC-BY-4.0).
- **Hendrycks MATH**: [`nlile/hendrycks-MATH-benchmark`](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark), a mirror of Hendrycks et al. (2021), MIT.
- **SATBench**: [`LLM4Code/SATBench`](https://huggingface.co/datasets/LLM4Code/SATBench).

## License

CC-BY-SA-4.0 for the combined archive. The share-alike clause comes
from Easy2Hard-Bench (the Codeforces problems) and propagates to
everything else. Model outputs are also subject to each underlying
model's license; see the per-model HuggingFace card linked below.

## Citation

Withheld during anonymous review.

### Underlying models

Cite the underlying model in addition to this archive if you use its
traces in your work.

| Short name | HuggingFace card |
|---|---|
| `deepseek-r1-7b`      | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| `deepseek-r1-14b`     | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) |
| `deepseek-r1-32b`     | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| `r1-distill-llama-8b` | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |
| `qwq-32b`             | [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) |
| `phi-4-reasoning`     | [microsoft/Phi-4-reasoning](https://huggingface.co/microsoft/Phi-4-reasoning) |
| `qwen-7b`             | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| `qwen-14b`            | [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| `qwen-32b`            | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| `llama-8b`            | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| `phi-4`               | [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) |

The four R1-Distill models are from the DeepSeek-R1 technical report
([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)). QwQ-32B is from
the Qwen team's QwQ release. Phi-4-reasoning and Phi-4 are described in
Microsoft's Phi-4 technical reports. Qwen2.5-Instruct models come from
the Qwen2.5 report ([arXiv:2412.15115](https://arxiv.org/abs/2412.15115)).
Llama-3.1-8B-Instruct is from the Llama 3 report
([arXiv:2407.21783](https://arxiv.org/abs/2407.21783)).
