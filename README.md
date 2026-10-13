# Retrieval-Aware Atomic Proposition Embeddings

Master's thesis research project investigating whether retrieval-aware
fine-tuning improves question-to-proposition alignment while preserving APE's
variable-size representation, without substantial architectural changes.

**Current state:** provisional design, not yet supervisor-approved. No scientific
code, dataset generation, model reproduction or experiments have been executed.

## Development and scientific evidence

This repository is the single current scientific project. Develop and verify
the pipeline first, then write the thesis/paper using the recorded evidence.

[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) contains the initial protocol and
subsequent scientific observations. It links to exact prompts, configurations,
dataset versions, run metadata and outputs when these are implemented. A log
entry does not substitute for preserving the actual inputs and results.

Planned directories, added when needed:

| Path | Purpose |
|---|---|
| src/ | Shared Python/PyTorch scientific implementation |
| tests/ | Meaningful correctness and reproducibility checks |
| prompts/ | Exact version-controlled scientific prompt templates |
| configs/ | Version-controlled experiment configurations |
| runs/ | Actual resolved configurations, prompt/input snapshots, metrics and outputs; local and ignored |
| data/ | Local datasets and caches; version/provenance recorded in runs |

Essential comparisons include BM25, original APE, unchanged APE with the proposed
query interface, pretrained/same-data fine-tuned BGE-small and SSE. Retrieval,
support alignment and preservation are separate measurements. Negative findings
are valid scientific outcomes.

## Installation and reproduction

Runnable commands and tested dependencies will be added after upstream artifact
and environment checks. The planned implementation supports local CPU and Colab
CPU/CUDA with a thin launcher notebook sharing the same scientific code.

## References and provenance

- Ortame/Navigli, *Variable-size Discovery and Representation of Atomic
  Propositions*: [cited APE release](https://github.com/SapienzaNLP/APE)
  (artifact access/version unverified).
- [Sub-Sentence Encoder](https://aclanthology.org/2024.naacl-long.89/).
- [LLM-OASIS](https://huggingface.co/datasets/Babelscape/LLM-Oasis_claim_extraction).
- [PropSegmEnt](https://github.com/google-research-datasets/PropSegmEnt).

No data, model weights or scientific results are distributed here. Upstream
licenses/attribution must be verified before redistribution; no software license
has been selected. Scientifically material LLM involvement will be documented
accurately, following applicable institutional and publication disclosure policy.
