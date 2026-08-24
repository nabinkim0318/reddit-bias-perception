# Topic-model stability

This document describes the **supported exploratory topic-modeling contract**.
It is a methodology note, not a completed BERTopic study and not a claim that
a real Reddit topic solution is stable or semantically valid.

## What the repository provides

The repository provides a reproducible framework for:

- fitting **one** configured BERTopic model (`analysis.bertopic_model`)
- recording run configuration and aggregate provenance
- measuring **structural** assignment stability across configured seeds
  (`analysis.topic_stability`)
- binding topic-id → category maps to a **specific** topic run

Public tests use synthetic assignments only. Default CI does **not** fit
BERTopic, download sentence-transformer models, or require a GPU.

## Topic identities

Topic IDs are labels local to one fitted model. They are not scientific
categories and must not be reused across unrelated stochastic runs as if
“topic 3” were the same entity.

`nr_topics="auto"` is an **exploratory** reduction choice. Topic count and
outlier rate may vary across seeds. A single automatically reduced solution
is not canonical without a stability evaluation, and even a stable partition
is not a semantic validation.

A fixed `nr_topics` integer may be set for controlled experiments.

## Stochastic configuration

Important settings are explicit in `config/topic_model.json` and
`analysis.topic_config.TopicModelConfig`:

| Component | Seed / determinism |
|---|---|
| UMAP | `random_state` set from the configured topic-model seed |
| HDBSCAN | No `random_state` in the supported APIs; treated as deterministic given embeddings |
| CountVectorizer | Deterministic given documents and its configuration |
| Embedding model | Identifier recorded; revision recorded only when genuinely available |

Do not invent unsupported `random_state` arguments.

## Assigned-topic probability

Assigned probabilities come from BERTopic’s document-information API
(`get_document_info(...)["Probability"]`). Numeric topic IDs are **never**
used as offsets into a probability matrix. Outlier topic `-1` is recorded as
missing assigned-topic confidence.

## Stability metric

Pairwise assignment agreement uses **Adjusted Rand Index (ARI)**, which is
invariant to label permutation.

The report includes:

- inlier topic count and outlier rate per seed
- topic-count and outlier-rate distributions
- pairwise ARI on all assignments (treating `-1` as a cluster label)
- pairwise ARI on documents that are inliers in **both** compared runs, with
  that document count as the denominator
- min / median / max summaries

Outliers are not silently dropped. The software does **not** declare a model
“stable” merely because a run completed, and it does not apply a universal
ARI threshold such as 0.8.

## Mapping provenance

A topic-to-category mapping must include:

- `topic_run_id`
- `topic_assignment_checksum`
- `mapping_version`
- `labels`

Applying a mapping created for a different run is an error. Statistics that
use mapped categories are conditional on that exploratory grouping; they do
not validate BERTopic.

## Local vs public execution

Real fitting:

```text
python -m analysis.bertopic_model --input LOCAL.csv --output-dir LOCAL_DIR
python -m analysis.topic_stability_fit --input LOCAL.csv --output-dir LOCAL_DIR
```

These commands may download embedding assets and read local research data.
They are not part of `make demo`.

Aggregate stability from already-computed assignment vectors:

```text
python -m analysis.topic_stability --assignments-json runs.json --output report.json
```

## What this does not establish

- semantic validity of topics
- that a real Reddit BERTopic solution is stable, unless a governed
  multi-seed run on local data was actually completed and reported separately
- ground-truth bias categories
- causal structure in Reddit discourse
