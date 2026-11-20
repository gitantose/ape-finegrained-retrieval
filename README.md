# Fine-grained retrieval with Atomic Proposition Embeddings

This thesis studies whether retrieval-specific fine-tuning can improve
[Atomic Proposition Embeddings (APE)](https://github.com/SapienzaNLP/APE) while
preserving their variable-size representations. The current implementation
evaluates frozen
[APE](https://huggingface.co/sapienzanlp/ape-bge-small/tree/e6dc09deb8c9e18323f17bbfd35b9025d89d8044),
unsupervised
[SimCSE](https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased/tree/218baf6bc48c463a53f5b4e1c568299537fab124)
and unsupervised
[Contriever](https://huggingface.co/facebook/contriever/tree/2bd46a25019aeea091fd42d1f0fd4801675cf699).


## Document retrieval

Each benchmark ranks its own full candidate corpus. Dataset revisions are
pinned in [the benchmark configuration](configs/benchmarks.json).

- [**NQ Simplified**](https://huggingface.co/datasets/LLukas22/nq-simplified) uses original questions to retrieve supplied context
  passages. The dataset-linked context is relevant when a native answer span
  matches the source text at its recorded offset.
- [**SQuAD v1.1**](https://huggingface.co/datasets/rajpurkar/squad) uses original extractive questions to retrieve paragraphs.
  Relevance comes from the annotated paragraph with a verified answer span.
- [**PROPSEARCH**](https://huggingface.co/datasets/sapienzanlp/propsearch) uses released atomic statements to retrieve their source
  sentences from a corpus of source sentences and distractors. The released
  source sentence ID defines relevance.

APE scores query and document slots with Hungarian Average (HA), symmetric
MaxSim, or hypothesis-to-premise (H2P) with the query as hypothesis. Long
documents contribute slots from all segments. SimCSE scores with cosine
similarity, computed as a dot product of unit-normalized vectors, while
Contriever takes the dot product of unnormalized vectors.

Mean reciprocal rank (MRR) averages the inverse rank of the first relevant
candidate over queries, using the full ranking. Recall@k averages the fraction
of relevant candidates retrieved in the top *k*. All scores are fractions on
a 0–1 scale, rounded to four decimal places.

The tables use NQ Simplified test (16,039 queries), SQuAD v1.1 validation
(10,570 queries), and PROPSEARCH's released `train` split (36,337 queries).

### MRR

| Model | NQ Simplified | SQuAD v1.1 | PROPSEARCH |
| --- | ---: | ---: | ---: |
| APE (HA) | 0.4253 | **0.7239** | 0.9645 |
| APE (MaxSim) | **0.4427** | 0.6889 | 0.9529 |
| APE (H2P) | 0.4236 | 0.7235 | **0.9685** |
| SimCSE | 0.2322 | 0.4149 | 0.7680 |
| Contriever | 0.4383 | 0.6463 | 0.9508 |

### Recall

#### NQ Simplified

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| APE (HA) | 0.3140 | 0.5538 | 0.6442 | 0.7143 |
| APE (Symmetric MaxSim) | **0.3301** | 0.5750 | 0.6584 | 0.7296 |
| APE (H2P) | 0.3124 | 0.5520 | 0.6427 | 0.7133 |
| SimCSE | 0.1570 | 0.3117 | 0.3813 | 0.4459 |
| Contriever | 0.3208 | **0.5774** | **0.6658** | **0.7340** |

#### SQuAD v1.1

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| APE (HA) | **0.6334** | **0.8350** | **0.8863** | 0.9207 |
| APE (Symmetric MaxSim) | 0.5841 | 0.8203 | 0.8814 | **0.9211** |
| APE (H2P) | 0.6333 | 0.8342 | 0.8850 | 0.9191 |
| SimCSE | 0.3077 | 0.5368 | 0.6255 | 0.7062 |
| Contriever | 0.5339 | 0.7838 | 0.8529 | 0.9032 |

#### PROPSEARCH

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| APE (HA) | 0.9558 | 0.9745 | 0.9791 | 0.9826 |
| APE (Symmetric MaxSim) | 0.9397 | 0.9685 | 0.9753 | 0.9800 |
| APE (H2P) | **0.9611** | **0.9769** | **0.9807** | 0.9840 |
| SimCSE | 0.7314 | 0.8095 | 0.8344 | 0.8573 |
| Contriever | 0.9333 | 0.9723 | 0.9801 | **0.9858** |

Among the configurations tested, APE achieves the highest MRR on all three
document retrieval benchmarks. Contriever leads at some higher Recall cutoffs.

## SupportQA within-context support retrieval

SupportQA supplies the correct context with each original question. It uses
short contexts from NQ Simplified test and SQuAD v1.1 validation. Propositionizer
generates propositions from each context without seeing its questions.
Model-assisted review identifies minimal sufficient sets of generated
propositions as silver references. A question is single-support if the
smallest sufficient set contains one proposition and multi-support if it
contains two or more.

APE ranks active context slots by their maximum cosine similarity to any
query slot. A query-independent teacher alignment maps silver proposition IDs
to those slots. SimCSE ranks the generated proposition texts by cosine
similarity, and Contriever ranks them by dot product.

The frozen selection contains **8,967 contexts** and **10,142 query outcomes**.
The tables combine both sources over the **6,022 queries eligible for all three
models (59.4%)**, with 3,758 single-support and 2,264 multi-support queries.
The other queries had insufficient context (2,297), ambiguous support (1,001),
a decomposition error (821), or a reference proposition that exceeded the APE
alignment teacher's tokenizer limit (1).

Here, Recall measures the fraction of a silver reference support set retrieved.
All-required Success@k measures the share of queries for which the top *k*
contains every proposition in at least one silver reference support set.
Each metric uses the best available reference set for each query.

### SupportQA MRR

| Model | Single-support MRR | Multi-support MRR |
| --- | ---: | ---: |
| APE | 0.6976 | 0.8172 |
| SimCSE | **0.7729** | **0.8712** |
| Contriever | 0.7580 | 0.8614 |

### SupportQA Recall

| Support class | Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| Single | APE | 0.5512 | 0.9072 | 0.9072 | 0.9072 |
| Single | SimCSE | **0.6305** | **0.9674** | **0.9989** | **0.9997** |
| Single | Contriever | 0.6103 | 0.9602 | 0.9979 | **0.9997** |
| Multi | APE | 0.3189 | 0.8443 | 0.8443 | 0.8443 |
| Multi | SimCSE | **0.3590** | **0.8855** | **0.9954** | 0.9998 |
| Multi | Contriever | 0.3531 | 0.8701 | 0.9940 | **1.0000** |

### SupportQA all-required Success

| Support class | Model | Success@1 | Success@5 | Success@10 | Success@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| Single | APE | 0.5460 | 0.9066 | 0.9066 | 0.9066 |
| Single | SimCSE | **0.6261** | **0.9673** | **0.9989** | **0.9997** |
| Single | Contriever | 0.6051 | 0.9598 | 0.9979 | **0.9997** |
| Multi | APE | **0.0000** | 0.6784 | 0.6784 | 0.6784 |
| Multi | SimCSE | **0.0000** | **0.7597** | **0.9907** | 0.9996 |
| Multi | Contriever | **0.0000** | 0.7301 | 0.9872 | **1.0000** |

SimCSE achieves the highest MRR and all-required Success@5 in both support
classes on the common eligible set.

## Limitations and next steps

Document relevance labels can miss other valid candidates. SupportQA covers
short contexts, and its silver labels await human validation. Alignment adds
uncertainty to APE slot evaluation, and unmapped propositions count as misses.
APE has at most five active slots per SupportQA context in this run, so its
Recall and all-required Success stay flat from cutoff 5 onward.

Exact reconstruction of the frozen SupportQA dataset currently requires local
artifacts. A public reproducibility package is future work, alongside human
validation and evaluation of retrieval-specific fine-tuning against these
frozen results.
