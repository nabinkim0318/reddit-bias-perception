# Statistical methodology

This document describes the **supported exploratory inferential path** for
emotion scores grouped by mapped topic categories. It is a methodology
contract, not a completed confirmatory analysis and not a set of study
findings.

## Observational unit

The observational unit is the **Reddit post**.

Posts from the same subreddit cannot automatically be assumed independent.
Ordinary independent-row ANOVA is not treated as sufficient without
qualification.

## Supported model

For each emotion outcome the supported specification is:

```text
emotion_score ~ C(mapped_topic_category)
```

with cluster-robust covariance by subreddit when clustering is supportable
(statsmodels OLS, `cov_type="cluster"`, `groups=subreddit`).

This retains an interpretable group-comparison mean model while allowing
within-subreddit residual dependence. It is **not** a mixed-effects model,
**not** a randomized design, and **not** causal inference.

Mapped topic categories are **exploratory topic-derived groupings**. They are
not ground-truth bias categories. Tests conditional on that grouping do not
validate BERTopic.

## Clustering contract

Clustered inference requires:

- a cluster column (default: `subreddit`)
- no missing cluster labels among evaluable rows
- at least two clusters

The number of clusters is always reported. Dataset-level `n_clusters` in
`analysis_manifest.json` is the mapped evaluable table. Each emotion model's
`n_clusters` and `inference_status` are recomputed **after** dropping missing
scores for that outcome, so a sparse emotion is not labelled `cluster_robust`
merely because the full table had many subreddits.

If clustered inference is requested and those checks fail, the analysis
**refuses** rather than silently falling back to independent-row ANOVA.

Few-cluster cluster-robust standard errors can be unstable. This project
flags `n_clusters < 10` as `limited_few_clusters`. That cutoff is a **project
heuristic**, not a universal statistical law. Literature often discusses
larger cluster counts as more comfortable.

## Multiple testing

There are multiple emotion outcomes. The family of emotion-level **omnibus**
tests is corrected with **Benjamini-Hochberg FDR**.

Every tested emotion appears in the output regardless of p-value. Null
results are not dropped. Both raw p and BH q are reported, together with the
family size.

The primary inferential question per emotion is whether the mapped category
term contributes overall (Wald / omnibus under the clustered covariance), not
whether one treatment-coded coefficient happens to be significant. Pairwise
contrasts of every category pair are not the default.

BH correction does not eliminate researcher degrees of freedom.

## Effects and intervals

Per mapped category the descriptive table reports:

- N posts
- N subreddits
- mean emotion score
- standard deviation

Regression contrasts versus the reference category report estimate, standard
error, and a 95% confidence interval from the **cluster-robust** fit when
available. If an interval cannot be estimated, it is emitted as unavailable
with a reason. An independent-row interval is not substituted and labeled
cluster-aware.

A small effect is not scientifically important merely because p < .05.

## MANOVA

The legacy MANOVA in `analysis.emotion_manova` is an **unadjusted exploratory
diagnostic**. It does not account for subreddit clustering and is **not** the
supported confirmatory or primary inferential path.

## Outputs

Local aggregate outputs (no source text, no Reddit IDs):

- `emotion_descriptives.csv`
- `emotion_clustered_tests.csv`
- `emotion_contrasts.csv`
- `analysis_manifest.json`

CLI:

```text
python -m analysis.emotion_statistics \
  --input LOCAL_TABLE.csv \
  --group-column mapped_topic_category \
  --cluster-column subreddit \
  --output-dir LOCAL_STATS_DIR \
  --topic-mapping LOCAL_MAPPING.json \
  --topic-run-id RUN_ID \
  --topic-assignment-checksum CHECKSUM
```

Importing the module does not read research files or run models.

## GoEmotions execution

`analysis.sentiment_analysis` loads the GoEmotions tokenizer/model and VADER
only when an execution path is called. The default model ID is explicit
(`SamLowe/roberta-base-go_emotions`). An optional Hugging Face revision may be
supplied. When it is omitted, provenance records `model_revision` as unset:
the model ID is explicit, but no immutable revision was specified. A commit
SHA or tag makes that identity provenance stronger. `revision=None` does not
make the run scientifically reproducible, and neither setting validates
GoEmotions for Reddit.

Lazy loading and revision configuration are engineering controls. They do not
calibrate scores or change 512-token truncation. Predictions remain
exploratory local tooling.

## What this does not establish

- causal effects of “bias categories”
- representativeness of Reddit
- validity of GoEmotions
- semantic validity of BERTopic
- that synthetic clustered-regression tests are real study conclusions
