# Retrieval-Aware Atomic Proposition Embeddings

Research design for a Master's thesis investigating whether retrieval-aware
fine-tuning improves question-to-proposition alignment while preserving APE's
variable-size representation. The intended contribution concerns supervision,
dataset construction, and evaluation, without substantial architecture changes.

**Status: provisional design, not yet supervisor-approved.** No scientific
implementation, dataset construction, model reproduction, or experiments have
been executed. This repository does not yet provide runnable research commands.

## Scientific protocol

- [Methodology](docs/METHODOLOGY.md): research questions, proposed losses,
  dataset construction, annotation, metrics, and provenance requirements.
- [Experiments](docs/EXPERIMENTS.md): proposed comparisons, reproducibility,
  verification requirements, and local/Colab execution design.

The minimum study includes original APE, BM25, pretrained and same-data
fine-tuned BGE-small, and Sub-Sentence Encoder. It separates source-unit
retrieval, supporting-proposition alignment, and preservation diagnostics.
Negative results are valid scientific outcomes.

## Installation and reproduction

Installation instructions will be added after the upstream artifact and
environment checks. Do not infer supported dependency versions, commands, or
results from the proposed design. Planned execution uses one Python package
with plain PyTorch on local CPU or Colab CPU/CUDA; notebooks will only launch
the shared implementation.

## Primary references

- Ortame and Navigli, *Variable-size Discovery and Representation of Atomic
  Propositions*, supplied APE manuscript. The cited artifact location is
  [SapienzaNLP/APE](https://github.com/SapienzaNLP/APE); access and versions
  remain to be verified.
- Chen et al., [Sub-Sentence Encoder](https://aclanthology.org/2024.naacl-long.89/).
- [LLM-OASIS claim extraction](https://huggingface.co/datasets/Babelscape/LLM-Oasis_claim_extraction).
- [PropSegmEnt](https://github.com/google-research-datasets/PropSegmEnt).

## Provenance and licensing

No data, model weights, or scientific results are distributed here. Upstream
licenses and attribution must be checked before acquisition or redistribution.
No software license is selected yet. Scientifically material LLM involvement
will be documented in the methodology; applicable institutional disclosure
requirements remain to be confirmed.
