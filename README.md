# Fine-grained retrieval with Atomic Proposition Embeddings

This thesis studies whether retrieval-specific fine-tuning can improve
[Atomic Proposition Embeddings (APE)](https://github.com/SapienzaNLP/APE) while
preserving their variable-size representations. The current implementation
evaluates frozen
[APE](https://huggingface.co/sapienzanlp/ape-bge-small/tree/e6dc09deb8c9e18323f17bbfd35b9025d89d8044),
unsupervised
[SimCSE](https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased/tree/218baf6bc48c463a53f5b4e1c568299537fab124),
unsupervised
[Contriever](https://huggingface.co/facebook/contriever/tree/2bd46a25019aeea091fd42d1f0fd4801675cf699),
and the frozen
[BGE-small backbone](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a).

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
documents contribute slots from all segments. SimCSE and BGE-small use
cosine similarity, and Contriever takes the dot product of unnormalized
vectors. BGE-small uses normalized final-layer CLS vectors, with the original
queries unprefixed. For inputs longer than 512 tokens, it averages the unit
segment vectors and normalizes the resulting vector.

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
| APE (MaxSim) | 0.4427 | 0.6889 | 0.9529 |
| APE (H2P) | 0.4236 | 0.7235 | **0.9685** |
| SimCSE | 0.2322 | 0.4149 | 0.7680 |
| Contriever | 0.4383 | 0.6463 | 0.9508 |
| BGE-small | **0.4921** | 0.6976 | 0.8853 |

### Recall

#### NQ Simplified

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| APE (HA) | 0.3140 | 0.5538 | 0.6442 | 0.7143 |
| APE (Symmetric MaxSim) | 0.3301 | 0.5750 | 0.6584 | 0.7296 |
| APE (H2P) | 0.3124 | 0.5520 | 0.6427 | 0.7133 |
| SimCSE | 0.1570 | 0.3117 | 0.3813 | 0.4459 |
| Contriever | 0.3208 | 0.5774 | 0.6658 | 0.7340 |
| BGE-small | **0.3791** | **0.6280** | **0.7029** | **0.7618** |

#### SQuAD v1.1

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| APE (HA) | **0.6334** | **0.8350** | **0.8863** | 0.9207 |
| APE (Symmetric MaxSim) | 0.5841 | 0.8203 | 0.8814 | **0.9211** |
| APE (H2P) | 0.6333 | 0.8342 | 0.8850 | 0.9191 |
| SimCSE | 0.3077 | 0.5368 | 0.6255 | 0.7062 |
| Contriever | 0.5339 | 0.7838 | 0.8529 | 0.9032 |
| BGE-small | 0.6015 | 0.8154 | 0.8720 | 0.9093 |

#### PROPSEARCH

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| APE (HA) | 0.9558 | 0.9745 | 0.9791 | 0.9826 |
| APE (Symmetric MaxSim) | 0.9397 | 0.9685 | 0.9753 | 0.9800 |
| APE (H2P) | **0.9611** | **0.9769** | **0.9807** | 0.9840 |
| SimCSE | 0.7314 | 0.8095 | 0.8344 | 0.8573 |
| Contriever | 0.9333 | 0.9723 | 0.9801 | **0.9858** |
| BGE-small | 0.8591 | 0.9162 | 0.9314 | 0.9445 |

The highest MRR comes from BGE-small on NQ Simplified, APE (HA) on
SQuAD v1.1, and APE (H2P) on PROPSEARCH.

## SupportQA v1 within-context support retrieval

SupportQA supplies the correct context with each original question. It uses
short contexts from the repackaged `test` split of `LLukas22/nq-simplified`
and SQuAD v1.1 validation. Propositionizer generates propositions from each
context without seeing its questions.
Model-assisted review identifies minimal sufficient sets of generated
propositions as silver references. A question is single-support if the
smallest sufficient set contains one proposition and multi-support if it
contains two or more.

APE ranks active context slots by their maximum cosine similarity to any
query slot. The frozen BGE-small checkpoint also serves as the
query-independent alignment teacher. In that role it embeds proposition
references with masked mean pooling to match their IDs to APE slots. As a
standalone retriever, BGE-small uses CLS pooling and ranks the generated
proposition texts directly by cosine similarity, as does SimCSE. Contriever
ranks proposition texts by dot product.

The frozen selection contains **8,967 contexts** and **10,142 query outcomes**.
The tables combine both sources over the **6,022 queries eligible for all four
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
| SimCSE | 0.7729 | 0.8712 |
| Contriever | 0.7580 | 0.8614 |
| BGE-small | **0.8651** | **0.9129** |

### SupportQA Recall

| Support class | Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| Single | APE | 0.5512 | 0.9072 | 0.9072 | 0.9072 |
| Single | SimCSE | 0.6305 | 0.9674 | 0.9989 | 0.9997 |
| Single | Contriever | 0.6103 | 0.9602 | 0.9979 | 0.9997 |
| Single | BGE-small | **0.7637** | **0.9895** | **0.9995** | **1.0000** |
| Multi | APE | 0.3189 | 0.8443 | 0.8443 | 0.8443 |
| Multi | SimCSE | 0.3590 | 0.8855 | 0.9954 | 0.9998 |
| Multi | Contriever | 0.3531 | 0.8701 | 0.9940 | **1.0000** |
| Multi | BGE-small | **0.3940** | **0.9157** | **0.9972** | **1.0000** |

### SupportQA all-required Success

| Support class | Model | Success@1 | Success@5 | Success@10 | Success@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| Single | APE | 0.5460 | 0.9066 | 0.9066 | 0.9066 |
| Single | SimCSE | 0.6261 | 0.9673 | 0.9989 | 0.9997 |
| Single | Contriever | 0.6051 | 0.9598 | 0.9979 | 0.9997 |
| Single | BGE-small | **0.7586** | **0.9891** | **0.9995** | **1.0000** |
| Multi | APE | **0.0000** | 0.6784 | 0.6784 | 0.6784 |
| Multi | SimCSE | **0.0000** | 0.7597 | 0.9907 | 0.9996 |
| Multi | Contriever | **0.0000** | 0.7301 | 0.9872 | **1.0000** |
| Multi | BGE-small | **0.0000** | **0.8180** | **0.9934** | **1.0000** |

BGE-small has the highest MRR and all-required Success@5 in both support
classes on the common set.

The backbone control shares APE’s pretrained starting point. Differences in
training, representation and scoring can all affect the results, so this
comparison cannot isolate the effect of the slot architecture.

## Human review and error audit

SupportQA v1 keeps the original silver outcomes for all 10,142 queries.
The 70 confirmed human judgments form a separate layer joined by `query_id`.
The thesis author made the final judgments, with AI-assisted discussion or
note drafting in some cases. Human review changed the sufficient support
sets on 27 of the 70 reviewed queries. Nineteen gained a sufficient set where
the silver review had none, and one lost the set accepted by the silver review.

Of the 70 reviewed queries, 57 had a human-confirmed sufficient set. The
table uses the 38 of those queries with usable saved rankings from all four
models. The 19 supported queries without saved rankings are excluded from
retrieval scores. These results describe the reviewed subset.

| Model | First-positive MRR | Complete-set Success@5 |
| --- | ---: | ---: |
| APE | 0.6430 | 31/38 |
| BGE-small | 0.7829 | 37/38 |
| SimCSE | 0.8364 | 33/38 |
| Contriever | 0.7226 | 33/38 |

First-positive MRR measures how early the first relevant item appears.
Complete-set Success@5 counts queries for which the top five contain every
element of at least one human-confirmed sufficient set.

In 11 cases, APE assigned a single sufficient proposition to a slot but
ranked that slot below first. Reranking the existing slots could improve
those cases. In seven cases, even all active slots together did not cover a
complete set under the saved assignment. The saved evidence leaves open
whether the matching missed the evidence or the slots failed to represent it.

[SupportQA v1](https://huggingface.co/datasets/antoserra/supportqa-v1) is
available on Hugging Face. The
[evaluation report](https://huggingface.co/datasets/antoserra/supportqa-v1/blob/main/evaluation-report.md)
documents the 70 human-reviewed queries and the diagnostic audit on the 38
queries with saved rankings from all four models.

## Limitations and next steps

Document relevance labels can miss other valid candidates. SupportQA covers
short contexts, with human judgments available only for the reviewed subset.
Alignment adds uncertainty to APE slot evaluation, and unmapped propositions
count as misses. APE has at most five active slots per SupportQA context in
this run, so its Recall and all-required Success stay flat from cutoff 5 onward.
Learning to rank mapped support earlier is a hypothesis for a future
fine-tuning experiment.
