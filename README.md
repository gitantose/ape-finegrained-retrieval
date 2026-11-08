# Fine-grained retrieval with Atomic Proposition Embeddings

A Master's thesis project investigating whether retrieval-aware fine-tuning can
improve **question-to-proposition alignment while preserving a variable number
of proposition embeddings**.

The project builds on [Atomic Proposition Embeddings (APE)](https://github.com/SapienzaNLP/APE)
by Francesco Ortame and Roberto Navigli. APE represents a sentence as a set of
latent proposition vectors with activity scores. These vectors are embeddings,
not generated proposition text.

The intended study compares retrieval-only and joint retrieval/decomposition
training without changing APE's base architecture. It separates three questions:

- Can the model retrieve the source unit containing the requested information?
- Does the question align with its correct supporting proposition?
- Are the original decomposition capabilities preserved after fine-tuning?

## Research roadmap

The original APE reference experiment is complete and verified. The next stage
is **M2c: baseline comparisons**. Question-support data construction and
retrieval-aware fine-tuning remain planned work.

| Stage | Goal | Progress |
|---|---|---|
| M1 — Reference integration | Load pinned official APE and verify model behavior, precision and scoring | Complete and verified |
| M2a — PROPSEARCH encoding | Encode the full released candidate/query population with recoverable artifacts | Complete and verified |
| M2b — Exact retrieval | Compute and verify all gold ranks, metrics and the paper comparison | Complete and verified |
| M2c — Baseline comparisons | Compare APE with pretrained BGE-small, BM25 and Sub-Sentence Encoder (SSE) | BGE mean/CLS complete and verified; BM25/SSE planned |
| M3 — Question-support data | Construct and validate examples with document-disjoint splits, supporting propositions and reliable negatives | Planned |
| M4 — Fine-tuning | Compare retrieval-only and joint APE training, alongside same-data BGE training | Planned |
| M5 — Controlled evaluation | Measure retrieval, support alignment, decomposition preservation and generalization | Planned |
| M6 — Research review and next steps | Review findings with the supervisor and define any further experiments or revisions | To be defined after feedback |

M6 is an open review point, not a predetermined end to the project. Further work
will depend on the findings and supervisor feedback.

The first M2c comparison uses frozen pretrained BGE-small with mean and
CLS pooling. Its purpose is to compare APE's proposition representation with
single-vector encodings from the same backbone family. Query formatting is
frozen as literal `query: ` on queries only; the supplied paper, pinned code
and BGE model card describe different conditions. BM25 and SSE remain separate comparisons.

## Results: original APE on PROPSEARCH

The released benchmark contains **314,483 candidate positions** (14,483 sources
and 300,000 distractors) and **36,337 atomic-proposition queries**. All query
ranks and saved artifacts were verified, and the metrics were independently
recomputed from the saved ranks. Candidate duplicates were preserved.

The following are original-checkpoint results, with **no retrieval fine-tuning**.
They evaluate proposition-statement retrieval, not the planned question task.
Values are rounded to three decimals for display:

| Aggregate | Queries / weighting | MRR ×100 | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) |
|---|---:|---:|---:|---:|---:|
| Overall query mean | 36,337 | 96.663 | 95.894 | 97.556 | 97.941 |
| Source propositions: 1 | 1,962 | 99.842 | 99.796 | 99.898 | 99.898 |
| Source propositions: 2 | 9,195 | 98.537 | 98.303 | 98.782 | 98.880 |
| Source propositions: 3 | 8,850 | 97.662 | 97.243 | 98.124 | 98.328 |
| Source propositions: 4 | 7,040 | 96.871 | 96.094 | 97.798 | 98.153 |
| Source propositions: 5 | 3,921 | 95.291 | 94.083 | 96.710 | 97.195 |
| Source propositions: 6+ | 5,369 | 91.377 | 89.179 | 93.965 | 95.251 |
| Macro over 2–6+ | 5 equal-weight strata | 95.948 | 94.980 | 97.076 | 97.561 |

MRR is shown on a ×100 scale here; `metrics.json` stores it as a fraction.
Overall results weight each query equally. The paper macro instead gives equal
weight to source-complexity strata **2, 3, 4, 5 and 6+**, excluding singletons.

The exact local macro is **95.94750924715933**, compared with **96.0** displayed
in the supplied paper's Table 4: a difference of **−0.05249075284067 percentage
points**. Each individual stratum agrees at the paper's one-decimal precision;
the residual macro discrepancy remains unexplained. Exact numerical identity
with the paper is not claimed.

Known comparison limits include the released versus paper/card source-population
discrepancy, non-exhaustive alternative-answer labels and the original 128-token
truncation. These are not established causes of the macro difference. Full
precision values, methods, verification evidence and limitations are in
[RESEARCH_LOG.md](RESEARCH_LOG.md).

## Results: frozen pretrained BGE-small

Both pooling conditions have verified vectors and exact ranks for all **36,337
matched queries and 314,483 candidate positions**. IDs, order, duplicates and
relevance mappings match APE. The fixed query prefix is `query: `; original
pretrained weights, not APE's fine-tuned encoder, were used. Display precision
is three decimals; full values and every recall aggregate are in the run metrics.

| Model | Overall MRR ×100 | Macro MRR ×100 | Overall R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|---:|---:|
| APE H2P | 96.663 | 95.948 | 95.894 | 97.556 | 97.941 |
| BGE mean | 88.253 | 85.974 | 85.579 | 91.312 | 92.944 |
| BGE CLS | 84.050 | 81.523 | 80.747 | 87.853 | 89.925 |

| Source n_atoms | Queries | APE H2P MRR ×100 | BGE mean MRR ×100 | BGE CLS MRR ×100 |
|---|---:|---:|---:|---:|
| 1 | 1962 | 99.842 | 100.000 | 99.762 |
| 2 | 9195 | 98.537 | 95.227 | 91.742 |
| 3 | 8850 | 97.662 | 89.813 | 84.833 |
| 4 | 7040 | 96.871 | 86.328 | 81.149 |
| 5 | 3921 | 95.291 | 81.873 | 77.357 |
| 6+ | 5369 | 91.377 | 76.629 | 72.533 |

The prefix produces **5 truncated queries**, including
**1 newly over the limit**; all inputs remain.
The paper/code/model-card formatting discrepancy and historical revision limits
prevent claiming an identical historical run. These compare frozen released
models with different training objectives/data; they do not isolate architecture
or establish question-retrieval performance. See [RESEARCH_LOG.md](RESEARCH_LOG.md)
for verification, exact findings and measured resources. These BGE macros also
differ from Table 4's displayed 88.4 (mean) and 90.5 (CLS); the cause is
unresolved and no matching claim is made. M2c remains open for
BM25 and SSE; neither has been run.

## Requirements

- **Python 3.12**. `.python-version` selects 3.12.16 for reproducibility.
- **uv** for dependency/environment management; the recorded workflow uses 0.13.7.
- The dependencies in [pyproject.toml](pyproject.toml), resolved in [uv.lock](uv.lock),
  including official pinned APE and PyTorch 2.10.0. The lockfile includes the
  CUDA 12.8 runtime dependencies.
- **Linux and a CUDA-capable NVIDIA GPU with a compatible driver** for the current
  full benchmark commands. Their recovery implementation uses Linux file locking
  and signal timers; this workflow is not platform-independent.
- Enough host/GPU memory and disk for the selected configuration, dependency
  caches, model weights, dataset and saved embeddings. Resource limits are
  explicit in `configs/`; evaluation provides a memory-planning command.
- Internet access for the initial package, model and dataset downloads.

The focused model check supports CPU or CUDA, and the default tests use CPU.
The full encoding/evaluation commands require CUDA and do not fall back to CPU.
PyTorch supplies the CUDA user-space runtime; a compatible NVIDIA host driver
is required separately, without a separate CUDA toolkit installation.

## Installation

Run all commands from the repository directory containing `pyproject.toml`.
If uv was installed locally under `.tools/`, the PATH setting below finds it;
otherwise it uses the uv installation already on your PATH.

```bash
export PATH="$PWD/.tools:$PATH"
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export HF_HOME="$PWD/.cache/huggingface"

uv python install 3.12.16
uv sync --locked
```

`uv sync --locked` installs the project and its pinned official APE dependency
into `.venv/`, using the versions in [uv.lock](uv.lock). It fails if the lockfile
needs updating instead of silently resolving another environment. No environment
activation is required for the commands below.

### Verify the installation

Run the default unit tests, then download the pinned model artifacts and check
actual CUDA loading and encoding:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m ape_retrieval.check --device cuda --download --output runs/m1/cuda.json
```

For a CPU-only model check, replace `--device cuda` with `--device cpu` and use
`--output runs/m1/cpu.json`. Subsequent model checks can omit `--download`
once the artifacts are cached. Reports and tensors default to `runs/m1/<device>.json`
and `.pt`; `--output` selects another location. A CUDA failure never silently falls
back to CPU.

Optional CUDA scoring/ranking tests:

```bash
APE_TEST_DEVICE=cuda APE_EVALUATION_TEST_DEVICE=cuda .venv/bin/python -m unittest discover -s tests -v
```

The checks cover finite normalized outputs, slot selection, strongest-slot
fallback, input order, explicit truncation, scoring direction and exact rank ties.
They establish correctness on fixtures, not benchmark quality or throughput.

## Reproduce original PROPSEARCH retrieval

PROPSEARCH asks the model to retrieve the originating sentence for an atomic
proposition statement. It is distinct from the thesis's proposed question task.
The pinned release contains **314,483 candidate positions** (14,483 sources and
300,000 distractors) and **36,337 queries**.

### 1. Acquire and verify the dataset

```bash
HF_HOME="$PWD/data/propsearch/.cache/hf-home" HF_HUB_DISABLE_XET=1 .venv/bin/python -m ape_retrieval.propsearch --download
```

The acquisition command checks the pinned file hashes, row counts, unique IDs
and query-to-source mappings. Original order and duplicate candidate positions
are retained. Downloaded tables and the acquisition report stay in `data/`.

### 2. Encode the inputs

After acquiring the dataset and model, launch encoding independently of the
terminal:

```bash
mkdir -p runs/propsearch-original
nohup .venv/bin/python -u -m ape_retrieval.encode --run-dir runs/propsearch-original >> runs/propsearch-original/run.log 2>&1 < /dev/null &
```

The encoder calls official `APE.encode` and writes recoverable NumPy shards.
Model inference uses FP32, TF32 disabled, normalized vectors, activity threshold
0.6, batch size 2 and an explicit 128-token limit. The upstream strongest-slot
fallback is preserved. Untruncated token lengths and affected inputs are recorded;
no candidate is filtered out. Pair boundaries and input order survive recovery.
No model or dataset download takes place inside the encoding process.

Check all saved encoding artifacts without a GPU:

```bash
.venv/bin/python -m ape_retrieval.encode --run-dir runs/propsearch-original --verify-only
```

### 3. Evaluate the cached embeddings

Once encoding is complete, inspect the evaluation memory plan:

```bash
.venv/bin/python -m ape_retrieval.evaluate --plan-only
```

Launch exact retrieval as a separate process:

```bash
mkdir -p runs/propsearch-h2p
nohup .venv/bin/python -u -m ape_retrieval.evaluate --config configs/propsearch-evaluation.json --encoding-dir runs/propsearch-original --run-dir runs/propsearch-h2p >> runs/propsearch-h2p/run.log 2>&1 < /dev/null &
```

Evaluation directly reuses the pinned upstream scorer and rank operations. It
loads saved embeddings without loading the APE model or re-encoding text. H2P
(Hypothesis-to-Premise) receives the query first:

```text
score(query, candidate) = min over query slots of
                         max over candidate slots of cosine similarity
```

Gold rank is `1 + count(candidate_score > gold_score)`. Exact ties are optimistic;
no epsilon, rounding or approximate search is introduced. All duplicate candidate
positions remain in the population. Only compact ranks and metadata are saved,
not the full query-candidate score matrix.

## Frozen pretrained BGE-small: mean and CLS

Configuration: [configs/bge.json](configs/bge.json), original pretrained
`BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
These weights are separate from APE's fine-tuned encoder. Both pooling conditions
use the complete original release and relevance mappings, normalized FP32,
TF32 off, batch size 2 and explicit 128-token truncation. Queries receive the
literal `query: ` prefix (one trailing space); candidates receive none.

This freezes a paper-described condition: the supplied paper says `query:`,
the pinned APE evaluation code passes raw queries, and BGE's model card documents
a different retrieval instruction. Historical whitespace/teacher revision identity
is not certified. No prefix selection or benchmark-driven tuning is performed.
Prefix-induced overflow is recorded alongside original long inputs.

Acquire the pinned pretrained weights and inspect complete-task shapes:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python -m ape_retrieval.bge acquire
.venv/bin/python -m ape_retrieval.bge plan
```

Acquisition verifies the Hub LFS SHA-256 and starts the shared two-hour deadline.
All commands use `runs/bge-frozen/` by default. Repeating acquisition does not
renew the deadline. Focused real-model CUDA checks before full execution:

```bash
BGE_REAL_TEST=1 BGE_TEST_DEVICE=cuda .venv/bin/python -m unittest discover -s tests -p test_bge.py -v
```

Encode both conditions with a shared frozen forward pass, then evaluate them as
separate ordinary commands. The independent launch sequence stops on failure or
recoverable stop:

```bash
nohup bash -c '.venv/bin/python -u -m ape_retrieval.bge encode && .venv/bin/python -u -m ape_retrieval.bge_evaluate mean && .venv/bin/python -u -m ape_retrieval.bge_evaluate cls && .venv/bin/python -u -m ape_retrieval.bge_evaluate verify' >> runs/bge-frozen/run.log 2>&1 < /dev/null &
```

The shared limits are **two hours total including downtime, 6 GiB host RSS,
2 GiB allocated CUDA tensors and 4 GiB additional disk**, across both conditions.
The deadline is fixed in `budget.json`; no automatic extension, device/precision
fallback or population reduction occurs. The initial complete-task plan estimates
1.08 GB raw vectors, 0.89 GB scoring tensors and 1.53 GB additional disk; these
are shape estimates, not measured runtime or whole-process RSS.

Inspect `encoding-resources.json`, `mean/encoding/progress.json`,
`cls/encoding/progress.json` and each condition's `evaluation/progress.json`.
For a recoverable stop, inspect the current child PID before signaling it:

```bash
ps -p "$(cat runs/bge-frozen/pid)" -o pid,args
kill -TERM "$(cat runs/bge-frozen/pid)"
```

After it exits, resume with the same independent launch command, unchanged code,
configuration and inputs. Each completed phase verifies and exits; unfinished
encoding shards/query blocks resume. A crash between pooling commits recomputes
only the missing tail. The shared deadline includes downtime; expiration requires
a separately approved continuation budget. Original APE artifacts are never
written by these commands.

Verify saved vectors and final ranks without re-encoding or rescoring:

```bash
.venv/bin/python -m ape_retrieval.bge verify
.venv/bin/python -m ape_retrieval.bge_evaluate verify
```

Each condition stores packed single-vector shards under `mean/encoding/` or
`cls/encoding/`. The reused packed schema's unit activity field is a placeholder,
not a predicted APE activity. Exact cosine ranks, query IDs, original qrels,
source strata, integrity hashes and metrics are under each `evaluation/`.
`query-truncation.json` records affected query IDs and original/prefixed lengths;
`comparison.json` recomputes the metrics and matches all query identities to APE,
with per-stratum/overall/macro differences and better/equal/worse rank counts.
The scorer is upstream `_pooled_ranks`; no full score matrix is stored.

## Progress, stopping and recovery

Watch either independent job by selecting its directory:

```bash
# For evaluation, use RUN_DIR=runs/propsearch-h2p instead.
RUN_DIR=runs/propsearch-original

tail -n 5 -f "$RUN_DIR/run.log"
```

Ctrl+C stops the log viewer; the computation continues. `progress.json` records
committed coverage and status. Evaluation's scoring bars describe individual
blocks; use `processed_queries` for overall progress. A run is complete only when
its saved status is `complete`, with full input/query coverage and final artifacts.

To request a recoverable stop, inspect the recorded PID and confirm that its
command is the intended job before sending SIGTERM:

```bash
ps -p "$(cat "$RUN_DIR/pid")" -o pid,args
kill -TERM "$(cat "$RUN_DIR/pid")"
```

Wait for the process to exit before resuming. Repeat its original launch command
with unchanged input files, source code, environment and configuration. Committed
artifacts are verified on recovery; only the uncommitted tail needs recomputation.
Manifests and SHA-256 hashes bind each run to its inputs and settings.

The supplied encoding configuration limits the run to four hours and 8 GiB of
data/run storage. Evaluation allows two hours, 8 GiB peak host RSS, 3 GiB allocated
CUDA tensors and 128 MiB additional evaluation storage. These are configured
limits, not runtime predictions or installation-space requirements. Resume retains
the original deadline, including downtime; it does not grant a new time budget.
There is no automatic device, precision or population fallback.

## Outputs and metric definitions

| Output | Contents |
|---|---|
| `runs/propsearch-original/shards/` | Packed embeddings, IDs, offsets, activities and token lengths; queries also retain gold positions and source complexity |
| `runs/propsearch-original/summary.json` | Encoding coverage, slot counts and explicit truncation findings |
| `runs/propsearch-h2p/predictions.npz` | Per-query IDs, gold candidate mappings, source complexity and exact gold ranks |
| `runs/propsearch-h2p/metrics.json` | Overall, source-stratified and paper-macro retrieval metrics |
| Each run's `manifest.json` / `progress.json` | Revisions, resolved configuration, environment, integrity hashes and execution metadata |

`metrics.json` distinguishes the overall query mean, source-complexity strata
and equal-weight paper macro defined above. Missing macro buckets are an error.
Nearly tied FP32 scores can change ranks across devices or matrix geometries;
no epsilon or rounding is applied to ranking.

## Repository structure

```text
configs/
  reference.json                model identities and reference inference settings
  propsearch.json               dataset identities and encoding settings
  propsearch-evaluation.json    exact APE evaluation settings and resource limits
  bge.json                     frozen pretrained mean/CLS protocol and shared limits
src/ape_retrieval/
  model.py                      pinned model loading and explicit precision/device
  check.py                      focused real-checkpoint checks
  propsearch.py                 dataset acquisition, validation and relevance joins
  encode.py                     recoverable embedding generation
  evaluate.py                   upstream exact ranking and metric aggregation
  bge.py                        pretrained acquisition and shared-forward encoding
  bge_evaluate.py               upstream pooled ranking and matched-query verification
tests/                          scoring, encoding, evaluation and recovery fixtures
pyproject.toml, uv.lock          package definition and resolved dependencies
RESEARCH_LOG.md                  scientific methods, results and limitations
```

`.venv/`, `.tools/`, `.cache/`, `data/` and `runs/` are local, ignored artifacts.
The repository records how to obtain dependencies and inputs; it does not bundle
installed packages, model weights, datasets or generated embedding files.

## References and artifact terms

- Ortame and Navigli, [*Variable-size Discovery and Representation of Atomic Propositions*](https://aclanthology.org/2026.emnlp-main.2302/), EMNLP 2026.
- [Official APE source](https://github.com/SapienzaNLP/APE/tree/a6723d56e70a5fc92577858c9c2a2f17a6cbfe2a), reused at a pinned revision.
- [APE checkpoint](https://huggingface.co/sapienzanlp/ape-bge-small/tree/e6dc09deb8c9e18323f17bbfd35b9025d89d8044).
- [Pretrained BGE-small checkpoint/model card](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a).
- [PROPSEARCH release](https://huggingface.co/datasets/sapienzanlp/propsearch/tree/37802225e3c9795200f3695404e00a22acdb4a31).

Exact identities and hashes are in `configs/`. The pinned upstream source LICENSE
is Apache-2.0, while its README badge names CC BY-NC 4.0. The pinned APE model and
PROPSEARCH cards declare CC BY-SA 4.0; the pinned pretrained BGE card declares MIT. Source, model and data terms are separate;
the upstream discrepancy must be reconciled before redistribution. No upstream
license is implicitly assigned to this thesis's own code.
