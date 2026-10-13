# Research log

The working scientific record for this project. Develop the pipeline first;
write the thesis/paper later using versioned prompts, configurations and saved
run evidence. This file records methods, important findings and their rationale;
it is not a manuscript or an agent task log.

No datasets have been generated and no scientific experiments have run.
The initial protocol below is provisional and not supervisor-approved.

## How to record scientific work

Append a dated entry after a significant dataset or experiment task. Keep it
concise and link to exact files/IDs instead of duplicating their contents.

- Purpose/question and change from the previous condition.
- Code commit or source hash, including uncommitted scientific changes.
- Prompt template path/revision/hash and model revision. Save the exact rendered
  prompt/messages or the immutable template plus exact inputs needed to recreate
  them. No dataset-generation prompt has been implemented yet.
- Dataset/source/split/qrel version, provenance, filters and annotation decisions.
- Run ID and resolved configuration, including actual command-line overrides,
  decoding settings, seeds, software, device and checkpoint versions.
- Verified outcomes, actual checks, failures, interpretation and limitations.
- Paths to preserved generated outputs, predictions, metrics and checkpoints.

When implemented, prompts/ and configs/ hold version-controlled scientific
inputs; runs/ holds actual run snapshots, outputs and metadata (local, ignored).
Data generation must preserve outputs/validation records because regeneration
may differ even with the same prompt and seed. Never store secrets or account
credentials in these records. Git captures revisions, not all external data or
runtime choices. Do not rely on today's code/config to reconstruct yesterday's run.

### Entry template

```text
YYYY-MM-DD | dataset/run ID | planned / completed / failed
Purpose:
Code/source revision:
Prompt/model/dataset/config references:
Actual execution and checks:
Outputs and observations:
Interpretation, limitations, next scientific step:
```

## 2026-10-12 ? Initial research protocol (planned, unexecuted)

### Question and hypotheses — submitted proposal

Can retrieval-aware fine-tuning improve question-to-proposition alignment while
preserving APE's variable-size representation? The intended contribution is
data construction, supervision, and evaluation without substantial architecture changes.

| Hypothesis | Required measurement |
|---|---|
| Retrieval supervision improves unchanged APE | Paired source-unit retrieval on identical corpora/qrels |
| Gains identify the requested proposition | Separate conditional and end-to-end support alignment |
| Joint loss preserves decomposition better than retrieval-only | Cardinality, slot diagnostics, PROPSEARCH, PAWS-WIKI |
| Benefits increase with complexity | Stratification by independent gold proposition counts |

Negative findings are valid. No efficacy, preservation, or generalization claim
is authorized before evidence and prespecified interpretation criteria.

### APE foundation — source facts

The manuscript describes encoder tokens attended by learned decoder slots.
Embedding/activity heads yield a variable-size bag, bounded by slot capacity.
The main setting has eight slots, three decoder layers, and a BGE-small-en-v1.5
backbone; training is restricted to at most five propositions. The activity
threshold is calibrated to 0.6. Actual release/checkpoint values remain unverified.

With frozen teacher proposition embeddings t_k, predicted slots e_i, and
Hungarian assignment pi minimizing cosine distance:

$$L_{emb}=\frac{1}{K}\sum_{k=1}^K(1-\cos(e_{\pi(k)},t_k)).$$

Matched slots receive activity target 1; other slots receive 0. Activity uses
binary cross-entropy over all slots. Repulsion penalizes positive cosine
similarity for slot pairs where at least one is active:

$$L_{rep}=\frac{1}{|P|}\sum_{(i,j)\in P}\max(0,\cos(e_i,e_j)),$$
$$P=\{(i,j):i<j,\sigma(a_i)>\tau\ \lor\ \sigma(a_j)>\tau\}.$$

The main loss is L_APE = L_emb + L_act + 0.1 L_rep. Implementation must define
zero-pair repulsion as zero and reject incomplete/empty supervision for this loss.
Matching is discrete; gradients flow through selected predictions, not matching.

The teacher mean-pools independently encoded proposition text. Training data is
LLM-decomposed WikiSplit with entailment filtering and singleton augmentation.
PROPSEARCH uses proposition statements to retrieve source sentences, not questions.
Its reported 96.0 MRR is macro-averaged over complexity classes, not query-averaged.
The published trainings use single runs. Longer passages are not validated by
the paper, and causal/temporal/discourse relationships are deliberately discarded.

#### Original scoring — source facts

For normalized active embedding sets with pairwise cosine matrix S:

- HA: optimal one-to-one matching of the smaller set into the larger, divided
  by the smaller set size; symmetric.
- MaxSim: one half of the sum of mean row maxima and mean column maxima;
  symmetric. Do not confuse this formula with directional ColBERT aggregation.
- H2P(premise, hypothesis): min over hypothesis slots of max over premise slots;
  asymmetric. A one-proposition hypothesis reduces it to the best premise match.

Reproduction must preserve direction, masks, threshold, normalization and task
aggregation. Latent slots are not generated proposition text.

### Minimum adaptation — recommendation, awaiting scientific approval

Use unchanged APE architecture. Accept complete English passages with 1-5
validated propositions, adequate coverage, no tokenizer truncation, and a
single-support question. Report exclusions and complexity/length distribution.
Do not chop a passage while retaining claims outside it or retain only five
claims from a longer passage. Six-to-eight-proposition capacity diagnostics
need separate approval; larger capacity/long-context changes are outside scope.

Keep a fixed copy of the checkpoint's original teacher. Match every annotated
proposition to all predicted passage slots independently of the question. The
gold proposition's assigned slot is the retrieval positive. Record assignment
ambiguity; do not select the positive solely through current query similarity.

Initially encode a question using normalized masked mean pooling of the existing
backbone, with configurable `query:` prefix. No new parameters are added. This
query interface is a recommendation, not original APE behavior. Evaluate both
original APE scoring and an unchanged-checkpoint control with this same interface.

For normalized q, matched positive e+, verified negatives N, and temperature T:

$$L_{retrieval}=-\log\frac{\exp(q^Te^+/T)}{\exp(q^Te^+/T)+\sum_{e^-\in N}\exp(q^Te^-/T)}.$$

Compare retrieval-only with L_retrieval + lambda L_APE. Use stable log-sum-exp,
BCE-with-logits, finite-value checks, and normalization safeguards. Use assigned
annotated slots as initial training candidates. Frozen targets stay identical
across conditions. Joint-loss full-proposition targets must pass coverage checks.

Inference scores a unit by max cosine over active slots, returning unit ID,
slot ID, and score. An empty active set is an invalid retrieval candidate and
a counted decomposition failure; never silently activate a fallback slot.

### Datasets and construction

| Dataset | Role | Limitations/adaptation |
|---|---|---|
| OASIS claim extraction | Training and in-distribution questions | title/text/claims, nominal five-sentence passages; no support questions |
| PROPSEGMENT/SSE retrieval | Broader original proposition task; interrogative diagnostic | Preserve token spans/cluster IDs; converted questions can change qrels |
| APE decomposition/held-out data | Loss verification and original diagnostics | Exact revision/splits/capacity filters unverified |
| PROPSEARCH | Original retrieval preservation diagnostic | Preserve released corpus, confounders, qrels, complexity aggregation |
| PAWS-WIKI | Initial representation diagnostic | Reproduce documented pair scoring and AUC/AP |
| SciQ | Conditional additional OOD test | Natural questions, passage support sometimes empty; no gold proposition labels |
| SNLI | Conditional entailment diagnostic | Not a minimum-project dependency |

OASIS is synthetic supervision requiring validation, not guaranteed atomic gold.
Its card declares CC BY-NC-SA 4.0. PROPSEGMENT's Wikipedia text and news retrieval
instructions differ; verify accessible coverage and spans, and disclose omissions.
Never silently replace unavailable news text or retain stale token alignments.

1. Pin source revisions/licenses and stable document/unit/proposition IDs.
2. Group OASIS by normalized Wikipedia title and preserve PROPSEGMENT clusters.
3. Deduplicate/group before assigning splits and before generating questions.
   Default OASIS groups: 80% training, 10% development, 10% test, seed 42.
   Preserve public benchmark splits and cluster integrity; do not randomly
   repartition related benchmark documents across roles.
4. Audit exact/near duplicates across splits/sources and recoverable overlap with
   APE pretraining. Do not claim independence from unavailable pretraining records.
5. Validate atomicity, self-containment, entailment/support and passage coverage.
6. Generate one question and expected answer per selected proposition.
7. Reject malformed, unanswerable, ambiguous and multi-support pairs.
8. Audit the prescribed sample, refining rules on pilot/development only.
9. Freeze questions, split manifests, qrels, negatives and annotation rules.

Multiple valid corpus answers may be included in qrels without adding
multi-support training. A changed query requires rechecking relevance; inherited
qrels are not automatically valid after interrogative conversion.

#### Candidate generator — recommendation

Use a pinned open-weight Qwen/Qwen3-4B-Instruct-2507 as a candidate for a separate
free-Colab generation pilot, not as a validated generator. Begin with at most
200 generated pairs; quality/runtime review and authorization precede scaling
to at most 1,000 accepted training pairs. No paid API or undocumented chat
generation. Record prompts, outputs and checks. If quality or resources are
inadequate, stop scaling and revise scope with approval.

#### Annotation guidelines — proposed minimum

Unit of review: question, expected answer, source passage, candidate gold
proposition, and all other annotated propositions. Annotators do not inspect
model predictions when producing held-out gold.

| Criterion | Record |
|---|---|
| Answerability | yes/no/uncertain; answer must be supported by the given unit |
| Single support | yes/no/uncertain; one atomic proposition sufficient |
| Alignment | supporting proposition ID(s), or unresolvable |
| Ambiguity | absent/present/uncertain and reason |
| Alternative support | all observed equivalents/alternative supporting IDs |
| Atomicity and coverage | complete/partial/uncertain, with unsupported/missing content |

Accept single-support examples only when answerability/single support are yes,
alignment is resolved, and ambiguity absent. Reject uncertain cases from clean
training/gold sets; retain rejection reason/count. Duplicate synonymous support
requires equivalent labels or exclusion, not a false negative. Coverage-complete
annotations are required for joint decomposition supervision and matching metrics.

| Allocation | Unique examples | Source default |
|---|---:|---|
| Pilot/guideline refinement | 30 | 15 OASIS / 15 PROPSEGMENT |
| Synthetic training audit | 50 | OASIS training only |
| Development/error analysis | 50 | 25 OASIS / 25 PROPSEGMENT |
| Held-out gold diagnostic | 120 | 60 OASIS / 60 PROPSEGMENT |

Keep all role documents disjoint. Use fixed sampled IDs before reviewing model
performance; preserve exclusions, replacements, and denominators. Pilot examples
never become final gold. Independently double-annotate 24-30 held-out examples
where possible (additional judgments on the same IDs). Report raw and criterion
agreement, kappa where meaningful, and pre-adjudication results; adjudicate with
recorded rationale. An LLM is not an independent human annotator. If no second
annotator is available, report that limitation rather than fabricate agreement.

The small gold set complements broader public/automatically validated evaluation;
it cannot alone justify broad generalization. Expansion and multi-support labels
are outside the initial annotation commitment.

#### Negatives

Start with verified random negatives. Mine hard negatives on training only;
freeze the list for comparisons. Same-context negatives are irrelevant
propositions for the slot objective, not negative labels for the positive source
unit. Exclude known positives, equivalent claims, and alternative support. Log
candidate identities and filters. Microbatch accumulation does not enlarge an
in-batch negative pool by itself.

### Evaluation and essential baselines

Include BM25, original APE/documented scoring, unchanged APE/new question
interface, pretrained BGE-small, same-data fine-tuned BGE-small, and SSE with
predicted segmentation. SSE is an essential baseline, not deferred work.
If unavailable, record the missing comparison and obtain scope approval.

BGE trains on the same accepted question/support relationships and compatible
negative identities and retrieves original units at inference. Gold proposition
text/masks are annotation-assisted representation controls, not regular retrieval.
SSE's complete pipeline includes segmentation latency/memory; isolate its legacy
dependencies instead of adopting its framework for the APE trainer.

Separate three endpoints:

- Unit retrieval: query-averaged MRR, Recall@1/5/10, nDCG@10 with frozen qrels;
  document-balanced summaries and gold-complexity strata.
- Support alignment: query-independent Hungarian matching of complete gold
  propositions; accuracy within known correct unit and end-to-end correct-unit
  plus correct-support accuracy. Report missing/inactive/unmatched cases as
  failures. This is a teacher-space diagnostic, not proof of human slot semantics.
- Preservation: cardinality exact and +/-1 accuracy, MAE, active-slot distribution,
  duplicate-slot similarity, PROPSEARCH MRR/Recall and PAWS-WIKI AUC/AP.

Use a fixed corpus per comparison and exact chunked scoring. Training negatives
are not the evaluation corpus. Define relevance granularity, unjudged-candidate
handling, tie-breaking and truncation before final evaluation. Initial ties use
stable unit/slot IDs. Report unknown relevance as a limitation, not verified
irrelevance. PROPSEARCH macro strata remain separate from query averages.

Compute paired document/cluster bootstrap intervals; retain all questions in a
resampled group. Report actual seed counts and variability separately: query
bootstrap does not estimate training variability. Use gold complexity, not a
model-dependent count, for comparing complexity strata. Freeze selection on
development before test. Preservation margins and interpretation criteria need
supervisor approval before test inspection.

### Provenance, disclosure and limits

Each scientific dataset/run must record source revisions/hashes, licenses,
generator/checker names and revisions, scientific prompts, decoding settings,
filters, seeds, annotation criteria, accepted/rejected counts, and all materially
LLM-assisted evaluation/interpretation. Follow applicable institutional and
publication AI-development disclosure policy once verified. Never misrepresent
code, data, annotations, or results. Do not redistribute upstream artifacts
without verified terms; no software license has been selected yet.

Known risks: synthetic annotation bias; teacher-space evaluation bias; incomplete
qrels; passage filtering bias; pretraining overlap; non-unique atomic granularity;
slot collapse; query-interface changes; small gold sample; free GPU availability.
Explicitly distinguish in-distribution evaluation, benchmark/domain transfer,
and any claimed out-of-distribution condition.

### References

- Ortame/Navigli, *Variable-size Discovery and Representation of Atomic
  Propositions*, supplied manuscript, sections 3-6 and Appendix A.
  [Cited release](https://github.com/SapienzaNLP/APE) (access unverified).
- [SSE paper](https://aclanthology.org/2024.naacl-long.89/) and
  [official code](https://github.com/schen149/sub-sentence-encoder).
- [OASIS card](https://huggingface.co/datasets/Babelscape/LLM-Oasis_claim_extraction).
- [PROPSEGMENT documentation](https://github.com/google-research-datasets/PropSegmEnt).
- [SciQ card](https://huggingface.co/datasets/allenai/sciq).
- [Candidate generator card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

These links are references, not evidence that artifacts have been acquired or executed.

## 2026-10-12 ? Execution and reproducibility protocol (planned, unexecuted)

### Proposed experiment matrix

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

### Sequence and starting configurations

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

### Planned package and interfaces

One `src/ape_retrieval` package within this research repository separates data, models, training, evaluation,
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

### Run records and checkpoint contract

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

### Execution environments and Colab

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

### Verification and reporting

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
