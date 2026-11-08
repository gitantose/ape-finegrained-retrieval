# Research log

This file records scientific methods, verified findings and limitations.
Installation and commands are in [README.md](README.md). Exact configurations,
inputs, environment versions and artifact hashes remain in the referenced run
records; this summary does not replace those records.

## Reference protocol

- Reuse official APE source, checkpoint and backbone configuration/tokenizer at
  the immutable revisions and hashes in [configs/reference.json](configs/reference.json).
  The full APE checkpoint contains its fine-tuned encoder weights; pretrained
  BGE weights are not substituted. Local loading adds revision pinning and hash
  verification, while model inference and scoring reuse upstream implementations.
- Use FP32, TF32 disabled, L2-normalized vectors, activity threshold 0.6,
  explicit 128-token truncation, encoding batch size 2, seed 42 and four CPU threads.
  Preserve the upstream strongest-slot fallback. APE slots are latent vectors,
  not generated proposition text.
- Preserve all released candidate positions, duplicates, query IDs, relevance
  mappings and original source-complexity labels. Queries remain proposition
  statements; PROPSEARCH is evaluation data, not thesis training supervision.
- Score query-first H2P: the minimum over query slots of their best candidate-slot
  cosine similarity. Gold rank is `1 + count(score > gold_score)`, with optimistic
  exact ties and no epsilon, approximate search or precision fallback.

Reference execution used Python 3.12.16, uv 0.13.7, PyTorch 2.10.0 / CUDA 12.8
and an RTX 3050 Laptop GPU (4 GiB) with NVIDIA driver 595.99.02 on Ubuntu 22.04.5.
[uv.lock](uv.lock) and each run manifest retain the full dependency/environment
identities. Resource measurements below describe these runs, not universal
hardware requirements.

## 2026-11-06 — M1: official model integration verified

Strict checkpoint loading, source/artifact identity checks and cached offline
loading passed. Three scoring tests passed on CPU and CUDA. Real-model checks
covered finite normalized output, activities, strongest-slot fallback, empty
inputs, batching/order and explicit truncation; known-answer scoring checked
H2P direction, variable slots, duplicates and exact/adjacent-FP32 ties.

CPU/CUDA fixture slot counts matched **[2, 1, 1]**. Maximum embedding difference
was **6.984919309616089e-7**, passing comparison at atol=rtol=2e-5 without changing
precision. CUDA fixture peak allocated tensor memory was **169.1 MiB**. These
fixtures establish integration correctness, not benchmark quality or throughput.

Evidence: local `runs/m1/cpu.json` / `.pt`, `runs/m1/cuda.json` / `.pt` and
`runs/m1/device-comparison.json`. They retain exact inputs, tensors, activities,
versions, hashes and numerical comparisons.

## 2026-11-06 — M2a: full PROPSEARCH encoding verified

Dataset revision and file hashes are in [configs/propsearch.json](configs/propsearch.json).
All **14,483 sources + 300,000 distractors = 314,483 candidate positions** and
**36,337 queries** passed release/hash/relevance validation and were encoded.
The 246 duplicate distractor-text groups retained their separate positions;
no text deduplication or population correction was performed.

Consecutive input pairs were sent to official `APE.encode`, preserving pair
boundaries/order across recovery. This geometry need not be bitwise identical
to a benchmark call that globally sorts unique texts. Saved shards retain vectors,
activities, IDs, offsets and original token lengths; queries also retain gold
positions and source complexity. Focused tests included real CUDA interrupted/
resumed consistency. Independent closure verified all shards without re-encoding.

| Encoding evidence | Result |
|---|---:|
| Candidate / query shards | 615 / 71 |
| Candidate / query active vectors | 1,005,488 / 51,178 |
| Active slots per input | 1–5 on both sides |
| Raw candidate / query FP32 vector bytes | 1,544,429,568 / 78,609,408 |
| Recorded wall time, including final verification | 1,270.186113357544 s (21m 10.19s) |
| Data + encoding storage before closure receipt | 1,677,695,727 bytes (1.56 GiB) |
| Peak allocated CUDA tensors | 177,371,648 bytes (169.15 MiB) |
| Host RSS / reserved CUDA peak | Not recorded |

Evidence: `data/propsearch/acquisition.json` and local
`runs/propsearch-original/{manifest.json,progress.json,summary.json,closure-verification.json}`
plus `shards/`. Exact coverage, hashes, slot histograms and affected truncation
IDs/token lengths remain in these records. Encoding alone does not establish
retrieval effectiveness.

## 2026-11-06 — M2b: exact retrieval and paper comparison

Reused upstream `_multi_vector_ranks` (H2P only) and `_retrieval_metrics`, with
query chunks of 8 and recoverable blocks of 1,024 queries. Actual candidate vectors
fit the in-memory scorer; no custom similarity kernel was needed. Gold scores
came from the same chunk as competing scores. No model was loaded or text
re-encoded. Configuration: [configs/propsearch-evaluation.json](configs/propsearch-evaluation.json).

All **36,337 ranks**, 36 saved blocks, final hashes, original IDs/order/relevance
mappings and complexity labels passed closure checks. Final ranks match their
checkpoint blocks. Metrics reproduce from saved ranks; a separate NumPy
calculation agrees to float64 precision. Recall@1/5/10 hit counts are
**34,845 / 35,449 / 35,589**. No retrieval scores were rerun during closure.

MRR below is a **fraction**; recalls are **percentages**. The overall mean weights
each query equally. The paper macro weights the five 2/3/4/5/6+ strata equally,
excluding singletons; missing macro buckets are an error.

| Aggregate / source n_atoms | Queries / weights | MRR (fraction) | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|---:|---:|
| Overall query mean | 36337 | 0.9666338582690565 | 95.89399234939593 | 97.55620992376916 | 97.94149214299475 |
| 1 | 1962 | 0.9984231096916956 | 99.79612640163099 | 99.89806320081549 | 99.89806320081549 |
| 2 | 9195 | 0.9853722777417575 | 98.30342577487765 | 98.78194671016857 | 98.87982599238717 |
| 3 | 8850 | 0.9766217887020662 | 97.24293785310735 | 98.12429378531074 | 98.32768361581921 |
| 4 | 7040 | 0.968709536733753 | 96.09375 | 97.79829545454545 | 98.1534090909091 |
| 5 | 3921 | 0.9529067042032343 | 94.08314205559806 | 96.71002295332823 | 97.19459321601633 |
| 6+ | 5369 | 0.913765154977156 | 89.17861799217731 | 93.96535667722108 | 95.25051219966474 |
| Paper macro 2–6+ | 5 equal-weight buckets | 0.9594750924715933 | 94.98037473515207 | 97.07598311611483 | 97.5612048229593 |

### Comparison with supplied paper, Table 4

The supplied paper reports MRR on a ×100 scale. Its `Tot` is the five-stratum
macro, not the overall query mean. It reports no overall query-mean MRR,
singleton MRR or Recall@5/10 in Table 4; no missing paper values are invented.

| Table 4 column | Local MRR (%) | Paper displayed MRR (%) | Local minus displayed paper (percentage points) |
|---|---:|---:|---:|
| Tot | 95.94750924715933 | 96.0 | -0.05249075284066862 |
| 2 | 98.53722777417575 | 98.5 | 0.037227774175747186 |
| 3 | 97.66217887020662 | 97.7 | -0.03782112979338592 |
| 4 | 96.87095367337531 | 96.9 | -0.02904632662469453 |
| 5 | 95.29067042032344 | 95.3 | -0.009329579676560229 |
| 6+ | 91.3765154977156 | 91.4 | -0.02348450228440413 |

All five local stratum MRRs match the paper's one-decimal display precision.
The exact local macro is **95.94750924715933** on the ×100 scale and displays as
**95.9**, whereas the paper displays **96.0**. The residual **−0.05249075284066862
percentage-point** difference is unexplained. Raw paper ranks/unrounded metrics
are unavailable; no causal rounding, hardware or release-population explanation
has been established. No method was adjusted to match the published number.
This is verified released-population evaluation with close agreement, not exact
numerical reproduction.

### Measured evaluation resources

| Resource | Recorded value |
|---|---:|
| Manifest-to-completion interval | 178.85967898368835 s (2m 58.86s) |
| Peak host RSS | 6,671,441,920 bytes (6.21 GiB) |
| Peak allocated CUDA tensors | 1,748,832,768 bytes (1.63 GiB) |
| Peak reserved CUDA memory | 2,023,751,680 bytes (1.88 GiB) |
| Evaluation directory before closure receipt | 4,430,962 bytes (4.23 MiB) |

The interval includes post-manifest verification, cache loading, scoring and
output writes; initial imports/pre-manifest setup and earlier encoding are
excluded. Whole-process wall time was not separately captured. CUDA allocator
peaks exclude some driver/library/context memory. Recorded runtime/memory/storage
allowances were respected; exact limits remain in the run configuration.

Evidence: local `runs/propsearch-h2p/{manifest.json,progress.json,closure-verification.json}`,
`ranks-*.npz`, `predictions.npz` and `metrics.json`. These retain exact ranks,
result/encoding/code/environment fingerprints, independent metric recomputation,
paper differences and resource records. Local run artifacts are ignored by Git;
only this scientific summary and the reproduction code are included here.

## Scientific limitations

- The released source population is **14,483**, versus **14,867** in the supplied
  paper/card histogram. No rows were invented or removed. The separate 13,020-row
  counting subset is not the retrieval corpus and was not acquired for this run.
- Each query has one labelled gold source; alternative valid answers are not
  exhaustively labelled. Duplicate candidate positions remain in the evaluation.
- Explicit truncation affects **92 sources, 333 distractors and 4 queries**;
  **391 queries** have truncated gold sources. Maximum original candidate/query
  lengths are 548 / 222 tokens. Affected examples were retained and recorded.
- FP32 device/batching geometry can change near-tie ranks, but this possibility
  is not an established explanation of the paper discrepancy.
- The supplied paper's byte identity with the published version, historical
  teacher/training revisions and all original numerical settings are not fully
  reconciled. Supplied PDF SHA-256:
  `db4f09e8960a1aae3fcb76873db1b62e2bd6d77850547731ce674503ff352407`.
- These results concern the unchanged APE checkpoint on proposition statements.
  They establish neither question-retrieval improvement nor preservation after
  fine-tuning. Latent slots do not guarantee human-readable interpretations.

For later experiments, add only the method change, validation, main results,
limitations and evidence locations. Keep installation commands in README and
exact per-run details in machine-readable artifacts.

## 2026-11-07 — M2c.1: frozen pretrained BGE mean/CLS protocol

Configuration: [configs/bge.json](configs/bge.json). Original pretrained
BGE-small at the recorded backbone revision, verified safetensors SHA-256
`3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`
(133,466,304 bytes), distinct from APE's fine-tuned encoder. All original input
IDs/order/duplicates/qrels and original source strata are retained. Conditions
are masked mean and CLS, unit-normalized FP32, TF32 disabled, CUDA batch 2,
128 tokens and literal `query: ` on queries only; candidates are unprefixed.

The supplied paper says `query:`, pinned APE evaluation code passes raw queries,
and BGE's card documents a different instruction. This formatting was frozen
before execution, with no test-result tuning; historical formatting/revision
identity is not certified. Prefix-induced overflow is recorded against M2a's
unprefixed token lengths, without filtering affected inputs.

Upstream `pool` computes both conditions from one frozen forward pass. Separate
encoding/evaluation commands reuse the durable project stores and upstream
`_pooled_ranks`; gold scores and competing scores use the same exact cosine
chunk with strict-greater ranks. The single-vector packed schema's unit activity
is a placeholder, not an APE prediction. Custom code adds only incremental
ID-preserving storage, provenance and shared-budget recovery missing upstream.

Before launch, 20 tests ran (19 passed, one unrelated opt-in APE test skipped).
Actual CUDA tests covered both pooling formulas, masking, normalization, prefix
truncation and bitwise interrupted/resumed consistency. Fixtures also covered
order/duplicates/qrels, strict ties, asymmetric commits, corruption and rank
recovery. An initial offline snapshot lookup failure was fixed by requesting
only acquired files. Exact commands and logs remain in runs/bge-frozen/.

Both conditions completed and passed explicit CPU closure verification without
re-encoding or rescoring. All 686 encoding shards per condition (615 candidate,
71 query), 36 rank blocks per condition, final hashes, unit norms, IDs/order/qrels,
full query coverage and matched APE gold mappings pass. All 737 original APE
artifact files retain their fingerprints. Metrics recompute exactly; independent
NumPy reciprocal-rank/recall calculations agree within 3e-14. Candidate token
lengths equal M2a; both pooling audits agree.

MRR below is a fraction; source strata and macro retain their original definitions.
Deltas are baseline minus APE in percentage points (rounded to three decimals;
exact differences and all recall aggregates remain in comparison.json).

| Aggregate / source n_atoms | BGE mean MRR | BGE CLS MRR | Mean − APE (pp) | CLS − APE (pp) |
|---|---:|---:|---:|---:|
| Overall query mean | 0.8825331514943522 | 0.8404956914694713 | -8.410 | -12.614 |
| 1 | 1.0 | 0.9976177779573885 | 0.158 | -0.081 |
| 2 | 0.9522730709963803 | 0.917418100854284 | -3.310 | -6.795 |
| 3 | 0.8981344580346655 | 0.8483337935165697 | -7.849 | -12.829 |
| 4 | 0.8632784386548419 | 0.8114884405148186 | -10.543 | -15.722 |
| 5 | 0.8187333129724688 | 0.7735696889183796 | -13.417 | -17.934 |
| 6+ | 0.7662940503141421 | 0.7253319014944524 | -14.747 | -18.843 |
| Macro 2–6+ | 0.8597426661944997 | 0.8152283850597009 | -9.973 | -14.425 |

| Condition / aggregate | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---:|---:|---:|
| mean / overall | 85.57943693755676 | 91.31188595646312 | 92.9438313564686 |
| mean / macro 2–6+ | 82.85150140021327 | 89.56380394385185 | 91.4983636183076 |
| cls / overall | 80.7468971021273 | 87.8526020309877 | 89.92486996725101 |
| cls / macro 2–6+ | 77.7941489380796 | 85.82352189117866 | 88.18489756813015 |

Matched-query rank counts (baseline better / equal / worse than APE):

- mean: 777 / 30745 / 4815.
- cls: 754 / 28997 / 6586.

Truncation: unchanged 92 sources and 333 distractors; 5
prefixed queries, 1 newly over 128 tokens. Original
unprefixed queries over the limit: 4; 391 queries retain truncated gold sources.
Query length delta histogram: {'2': 36337}.
Affected query IDs and original/prefixed lengths are in query-truncation.json;
all candidate/query affected inputs also remain in encoding summaries.

| Measured resource / time | Value |
|---|---:|
| Independent launch through automatic verification, including imports/preparation | 1548.93688082695 s |
| Acquisition-start through explicit closure (includes tests/pre-launch work/downtime) | 1713.6424703598022 s |
| Peak host RSS across encoding/evaluations/CPU closure | 2735710208 bytes |
| Peak allocated CUDA tensors | 1017844224 bytes |
| Peak reserved CUDA allocator memory | 1356857344 bytes |
| Both run conditions/logs/receipts + newly acquired weights, before closure receipt | 1252920143 logical bytes |
| Same files, allocated blocks | 1255059456 bytes |
| Conservative total additional-storage upper bound, including auxiliary files and growth reserve | 1274343424 bytes |

All approved shared limits were respected: two hours, 6 GiB host, 2 GiB allocated
CUDA and 4 GiB additional disk. Storage excludes pre-existing data/environment/
model metadata and directory entries; CUDA peaks exclude driver/context/library
memory outside the allocator. Per-phase timings/resources, exact environment/
source/weight fingerprints, predictions and independent comparisons are in
runs/bge-frozen/{acquisition.json,budget.json,encoding-resources.json,comparison.json,
closure-verification.json}, and each condition's encoding/evaluation manifests.

APE's earlier Table 4 macro discrepancy remains unresolved. The BGE formatting
and historical-teacher limitations also remain; no causal rounding/hardware
explanation is asserted. These frozen-model comparisons do not isolate training
from architecture, evaluate question retrieval or demonstrate fine-tuning gains.
M2c.1 is complete; M2c stays open for separately approved BM25/SSE work.

### BGE comparison with supplied Table 4

The fixed conditions do not reproduce the supplied paper's displayed baselines:

| Condition | Local macro MRR ×100 | Paper displayed macro | Local − paper (pp) |
|---|---:|---:|---:|
| BGE mean | 85.97426661944996 | 88.4 | -2.4257333805500423 |
| BGE cls | 81.52283850597009 | 90.5 | -8.977161494029914 |

The local mean/CLS ordering also differs from Table 4. The cause is unresolved:
there are no paper raw ranks/unrounded metrics or controlled formatting/revision
comparisons here. The known paper/code/card formatting discrepancy is a limit,
not demonstrated causal evidence. Do not substitute the query mean for the macro,
attribute the gaps to hardware/rounding, or tune the frozen condition to match.
