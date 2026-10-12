# Methodology

Status: **provisional, unexecuted, not supervisor-approved** (2026-10-12).
This is a prospective protocol, not a report of completed experiments. Source
facts below describe the supplied APE manuscript; recommendations are not
established APE behavior. Experiment scheduling/reproducibility lives in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Question and hypotheses — submitted proposal

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

## APE foundation — source facts

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

### Original scoring — source facts

For normalized active embedding sets with pairwise cosine matrix S:

- HA: optimal one-to-one matching of the smaller set into the larger, divided
  by the smaller set size; symmetric.
- MaxSim: one half of the sum of mean row maxima and mean column maxima;
  symmetric. Do not confuse this formula with directional ColBERT aggregation.
- H2P(premise, hypothesis): min over hypothesis slots of max over premise slots;
  asymmetric. A one-proposition hypothesis reduces it to the best premise match.

Reproduction must preserve direction, masks, threshold, normalization and task
aggregation. Latent slots are not generated proposition text.

## Minimum adaptation — recommendation, awaiting scientific approval

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

## Datasets and construction

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

### Candidate generator — recommendation

Use a pinned open-weight Qwen/Qwen3-4B-Instruct-2507 as a candidate for a separate
free-Colab generation pilot, not as a validated generator. Begin with at most
200 generated pairs; quality/runtime review and authorization precede scaling
to at most 1,000 accepted training pairs. No paid API or undocumented chat
generation. Record prompts, outputs and checks. If quality or resources are
inadequate, stop scaling and revise scope with approval.

### Annotation guidelines — proposed minimum

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

### Negatives

Start with verified random negatives. Mine hard negatives on training only;
freeze the list for comparisons. Same-context negatives are irrelevant
propositions for the slot objective, not negative labels for the positive source
unit. Exclude known positives, equivalent claims, and alternative support. Log
candidate identities and filters. Microbatch accumulation does not enlarge an
in-batch negative pool by itself.

## Evaluation and essential baselines

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

## Provenance, disclosure and limits

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

## References

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
