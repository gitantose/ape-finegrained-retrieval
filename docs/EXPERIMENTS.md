# Experiments and reproducibility

Status: **prospective specification; no experiments executed** (2026-10-12).
Scientific definitions, data rules and annotations are authoritative in
[METHODOLOGY.md](METHODOLOGY.md). No dependencies, checkpoints, datasets or
scientific commands have been installed/implemented yet.

## Proposed experiment matrix

IDs below identify specifications, not successful runs. Actual runs receive a
unique suffix including date, configuration/seed and attempt; never overwrite them.

| Spec | Condition | Required evidence |
|---|---|---|
| E00 | Original APE reproduction | Verified checkpoint/config, subset then released diagnostics |
| E01 | BM25 source-unit retrieval | Frozen corpus/qrels and lexical preprocessing |
| E02 | Original APE documented scoring | Correct scoring orientation and thresholds |
| E03 | Unchanged APE single-vector query interface | Same interface as adapted model, no fine-tuning |
| E04 | Pretrained BGE-small | Verified recommended pooling/instructions |
| E05 | Same-data BGE-small fine-tuning | Same accepted pairs and applicable negative identities |
| E06 | SSE with predicted segmentation | Encoder and segmenter revisions; full cost included |
| E07 | APE retrieval-only, random negatives | Matched-slot supervision and preservation diagnostics |
| E08 | APE joint, random negatives | Same inputs/teacher; development-selected joint weight |
| E09 | Selected APE with hard negatives | Frozen training-only mining manifest |
| E10 | Selected APE with same-context negatives | Verified non-supporting proposition identities |
| E11 | Frozen/unfrozen backbone control | Exact trainable parameter groups and resource accounting |
| E12 | Prefix/scoring controls | Bounded development comparisons; no test tuning |

Annotation-assisted proposition text/span controls are separately labeled. They
must not replace source-unit retrieval baselines or hide oracle annotations.
SciQ/SNLI and capacity diagnostics are conditional, not part of the minimum matrix.

## Sequence and starting configurations

Validate evaluator/baselines on existing proposition tasks and synthetic fixtures
before training-question construction. Then validate the generator and data,
implement/debug training, and run approved bounded experiments.

Recommended starting values (not validated optima): AdamW; LR 2e-5; weight decay
0.01; retrieval temperature 0.05; clip norm 1.0; fixed activity threshold 0.6;
CPU FP32. Joint lambda candidates: 0.1 and 1.0. Original APE internal loss
weights remain 1, 1, 0.1. Seed 42 for debug/development; final central comparisons
target 13, 42, 87. Original-paper reproduction uses its actual verified config,
not these adaptation defaults.

Select joint weight on development, then compare negative strategies; do not
run a full factorial grid. Before any research run, specify/approve actual
epochs/steps, batch size, accumulation, sequence length, negative counts,
parameter groups, scheduler, save interval, maximum wall time and output budget
using measured feasibility. These values are deliberately not invented in M0.

Freeze development-selected conditions before held-out evaluation. Record
every attempted run and failed condition; do not report only favorable seeds.
Report fewer than three seeds honestly if resources constrain the final matrix.

## Planned package and interfaces

One `src/ape_retrieval` package separates data, models, training, evaluation,
runtime utilities and a CLI. `python -m ape_retrieval` will expose prepare,
validate, evaluate, train and analyze; these commands do not exist yet.
Commands take configuration paths; train/evaluate also accept device and output
overrides; train supports resume. YAML configs are version-controlled scientific
specifications and resolved overrides are saved with runs.

Normalized units/questions/qrels/manifests preserve IDs, source group/split,
complete proposition annotations where available, provenance and hashes.
Adapters return normalized embeddings and masks with stable unit/slot references;
training APE also returns all slots and activity logits. No plugin framework.

Use venv/pip, pyproject, plain PyTorch, pytest and Ruff. Check actual upstream
and Colab compatibility before fixing the tested Python/dependency range.
Freeze CPU and Colab/CUDA environments separately as needed. Do not import
notebook/Colab dependencies into core scientific modules.

## Run records and checkpoint contract

Each actual run stores an ID, parent/resume relationship, resolved config/hash,
code revision (or bundle hash plus dirty-state record), model/checkpoint hashes,
data/split/qrel versions, random seeds, software versions, hardware/device,
evaluation metrics, exit status, output locations, and durations/resource use.

Proposed run contents: resolved configuration, metadata JSON, append-only
training JSONL, predictions, evaluation JSON, last/best checkpoints and their
integrity manifests. Figures/tables derive from those artifacts, not copied
manually from console output. Commit/source revision capture must include
uncommitted scientific changes; a clean commit ID alone is insufficient.

Checkpoint model/optimizer/scheduler/scaler, all RNG states, completed optimizer
step, epoch and deterministic data-order position. Save only after completed
optimizer steps. Uncheckpointed work is rerun. Begin with zero data-loader
workers and deterministic sampler for exact mid-epoch resume tests.

Write a temporary local checkpoint, verify completion/hash and finalize; retain
the previous verified checkpoint during durable synchronization. For Drive,
copy to a new temporary name, verify readback/hash, finalize and update the
durable pointer only after success. Do not assume FUSE rename/copy is atomic.
Resume only a verified compatible checkpoint; distinguish local from durable progress.

Exact continuation is tested in the same deterministic environment. CPU/CUDA
transfer supports continuation, not guaranteed bitwise equivalence. Reject
incompatible dataset/config changes unless explicitly recorded as a new run.

## Execution environments and Colab

Device auto selects available CUDA else CPU. CPU is explicit FP32. Explicit
CUDA requests fail clearly if unavailable. Mixed precision is conditional on
CUDA hardware support. Do not silently change batch/scientific parameters on OOM.

Selected workflow: local development/verification -> reviewed scientific bundle
-> thin Colab launcher -> shared Python commands -> verified Drive artifacts
-> local analysis. No public repository is required for transport. Local files
do not automatically exist on the remote runtime; upload and verify the bundle.

Launcher responsibilities: user authentication/Drive mount, bundle upload/hash
check, package install, device report, approved command invocation, and artifact
sync/retrieval. It contains no scientific training/evaluation logic. CPU fallback
uses identical implementation but may be too slow for full-scale workloads.

Documentation findings inspected 2026-10-12, **not live compatibility validation**:

- Official Colab MCP needs a local client handling tool-list-change notifications;
  Codex incompatibility has been reported. Test actual tools/output, not connection text.
- The inspected MCP pyproject requires Python >=3.13; isolate its environment.
- Official Colab CLI declares native Windows unsupported; exclude it initially.
- VS Code's extension supports notebook kernels/uploads/Drive and an experimental
  terminal. This does not redirect Codex's ordinary local shell to Colab.

MCP spike, capped at half a day: record versions, authenticate a disposable
notebook, verify post-connection tools, read/add/update/run a cell and observe
`print(2 + 2)` output, execute a CPU fixture, save/retrieve a durable artifact,
reconnect and verify recovery, optionally check available GPU, then end runtime.
Failure retains manual launcher execution; no custom bridge or default fork.
Free Colab availability/duration is not guaranteed. No keep-alive/limit bypasses.

References: [MCP](https://github.com/googlecolab/colab-mcp),
[MCP requirements](https://github.com/googlecolab/colab-mcp/blob/main/pyproject.toml),
[compatibility report](https://github.com/googlecolab/colab-mcp/issues/50),
[CLI](https://github.com/googlecolab/google-colab-cli),
[VS Code guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide),
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Verification and reporting

Planned meaningful tests: matching/permutation invariance, masked pooling,
score orientation and symmetry, empty-slot failure, exact metric fixtures,
document separation/qrels consistency, finite gradients, frozen parameter
integrity, overfit tiny fixture, checkpoint/restored RNG-data order agreement,
metadata/hash compatibility, and available CPU/CUDA smoke parity.

Do not equate subset smoke checks with paper reproduction or generator validity.
Before final claims, regenerate metrics from predictions, audit candidate
completeness, preserve original diagnostic cohorts, compute paired group
confidence intervals, and check seed/config consistency and baseline fairness.

No results table is populated in M0. Subsequent experiment records must link to
verified artifacts; failures/incomplete runs cannot be labeled successful results.
