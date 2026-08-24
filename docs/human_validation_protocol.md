# Human-validation protocol

**Protocol version:** 1
**Construct:** `visual_identity_bias_in_ai_generated_images` / version 1
**Codebook:** [annotation_codebook.md](annotation_codebook.md)

This document defines a reproducible procedure for evaluating **automated
yes/no predictions** against **human judgments**. It is a methodology
contract, not a completed study.

**The existence of this protocol does not mean human validation has been
completed.** No completed human-validation result is claimed unless a
separately governed evaluation is documented with real annotations, hashes,
and a privacy-reviewed aggregate report.

---

## Scientific questions this protocol can address

| Question | This protocol |
|---|---|
| Construct definition | Uses the codebook derived from the current prompt |
| Annotation reliability | Inter-annotator agreement on double-coded tasks |
| Automated-label accuracy | Model vs resolved human `yes`/`no` |
| Prevalence of the construct on Reddit | **Not** established here |
| Causal effects / “which subreddit is biased” | **Not** in scope |
| Objective bias in an AI system | **Not** in scope |

Human agreement does not prove construct validity. High model-vs-human F1
does not prove that the construct captures “AI bias.” A Reddit user’s
perception does not prove objective bias in an underlying generator.
Keyword-filtered posts are not a random sample of Reddit. A validation
sample is not representative of all Reddit discourse unless the sampling
design independently supports that claim.

---

## 1. Sampling

Source: a **local/private** automated-annotation artifact (typically the
combined LLM results CSV: `id`, `subreddit`, `clean_text`, `status`,
`pred_label`, plus optional `subreddit_group` and `matched_bias_types`).

Primary scientific population:

- `status == success` and `pred_label` in `{yes, no}`

`parse_error` and `model_error` rows are **execution failures**. They must
not enter yes/no classification metrics. An optional separate QA sample of
failures may be drawn; it stays out of the scientific denominator.

Procedure (implemented by `python -m validation.sample`):

1. Deduplicate source rows by original record id (first row wins). Duplicate
   source rows must not create duplicate tasks.
2. Stratify success rows by predicted label, subreddit group, and primary
   keyword/bias category when those fields exist.
3. Allocate the requested size as evenly as possible across non-empty
   strata, capped by stratum size, using a fixed seed for within-stratum
   shuffles.
4. Assign blinded `task_id` values that are not the original record ids.

**Sample-size adequacy is a study-design decision.** The tooling will not
claim that any configured `N` is statistically sufficient. If a stratum is
smaller than the allocation, take all available rows and record the
shortfall. Do not invent extra rows.

Real sampled text remains local/private and gitignored.

---

## 2. Blinding

Annotator-facing files contain only:

- `task_id`
- `text_to_annotate`
- `human_label` (empty until filled)
- `notes` (optional)

They must **not** contain model `pred_label`, `status`, `raw_output`,
`llm_reasoning`, original record ids, or expected answers.

A private sampling index (also local/private) retains the linkage needed to
score the model later.

---

## 3. Independent annotation

At least **two** annotators (`A` and `B`) label each sampled task using this
codebook, independently, without seeing each other’s labels or the model
outputs.

Annotators must use the vocabulary `yes` / `no` / `uncertain` /
`insufficient_context`. Unsupported labels are validation-input errors;
they are not coerced.

---

## 4. Annotation quality checks

Before scoring, the evaluator must:

- reject duplicate `task_id`s in a file
- reject unsupported labels
- detect missing annotations (index task absent or unlabeled)
- align records by `task_id`, never by row order
- fail clearly rather than dropping malformed rows silently

Optional drift checks (spot-audit a small random subset, timing, or
instruction quizzes) may be added by a study protocol; they are not
implemented as automation in this repository.

---

## 5. Disagreement handling

Report disagreements explicitly (count and rate). Do not silently keep one
annotator as “ground truth.”

Percent agreement alone must not be described as high reliability.
Cohen’s kappa is also reported when defined; kappa can be low when labels
are highly imbalanced even if percent agreement is high (prevalence /
marginal-distribution limitation). If kappa is mathematically undefined
(degenerate margins), report it as unavailable — do not invent a number.

---

## 6. Adjudication

Adjudication is **optional**.

If used, write a **separate** artifact that preserves original A/B labels:

- `task_id`
- `annotator_a_label`
- `annotator_b_label`
- `adjudicated_label`
- `adjudication_status` (`resolved` / `unresolved`)

Do not overwrite A/B files. A third reader (or a documented consensus rule)
assigns `adjudicated_label`. Unresolved items, and items adjudicated to
`uncertain` or `insufficient_context`, stay out of binary performance
denominators. Every `task_id` in the adjudication file must exist in the
sampling index; unknown IDs are rejected rather than ignored.
`unresolved` rows must not carry an `adjudicated_label`.

If adjudication is **not** used:

- still report inter-annotator agreement
- model-vs-human evaluation is limited to items where A and B **exactly
  agree** on `yes` or `no`
- that limitation must be stated in the report

---

## 7. Model-vs-human evaluation

Binary metrics use only tasks where **both** are true:

1. Model row is `success` with `pred_label` `yes` or `no`
2. Human reference is a resolved `yes` or `no` (exact A/B agreement, or
   adjudicated binary label)

Compute a confusion matrix (`yes` as the positive class), precision, recall,
F1, accuracy (secondary), and, where defined, specificity and balanced
accuracy. Zero denominators yield null / unavailable, not fabricated zeros
that pretend to be estimates. A defined F1 of 0 (no true positives, but
`FP + FN > 0`) is reported as 0.0, not unavailable.

Optional deterministic bootstrap confidence intervals may be attached to
precision, recall, F1, and accuracy. They are percentile intervals under a
fixed seed; they are not a license to overclaim.

Subgroup tables, if present, report **counts only** unless a future study
pre-registers adequate sample sizes. This protocol does not rank subreddits,
test group differences, or make causal claims.

---

## 8. Exclusions

Exclude from the binary scientific denominator:

- `parse_error` and `model_error` (report as coverage / execution failures)
- human `uncertain` and `insufficient_context`
- human disagreements that were not binarily adjudicated
- tasks that fail input validation (do not recode them as `no`)

A model failure must never be counted as a true negative or false negative.

---

## 9. Uncertainty handling

`uncertain` and `insufficient_context` are first-class outcomes. They must
appear in the aggregate counts. They must not be collapsed into `yes` or
`no` to inflate agreement or F1.

---

## 10. Reporting

Public or shareable output is an **aggregate** JSON report:

- schema / construct / codebook versions
- sampling seed and configuration hash (no absolute machine paths)
- checksums of the source artifact and annotation files
- counts listed below
- agreement metrics and confusion matrix
- exclusions and model execution-status counts
- timestamp and code SHA when available

The report must **not** contain Reddit text, titles, comments, usernames,
permalinks, stable Reddit ids, model raw outputs, human notes, per-record
labels, or task-to-record maps.

Recommended count fields:

`sampled`, `double_annotated`, `agreed_yes`, `agreed_no`,
`human_disagreements`, `uncertain`, `insufficient_context`, `adjudicated`,
`unresolved`, `model_success`, `model_parse_error`, `model_model_error`,
`binary_evaluable`

The binary-metric denominator must be recoverable from those counts
(`binary_evaluable`).

Default real reports stay local/gitignored until separately reviewed for
privacy and publication suitability.

---

## 11. Privacy and governance

- Real Reddit-derived validation text is local/private research data.
- Public tests use fictional synthetic records only (`SYNTH-` ids).
- Do not commit annotator task CSVs, sampling indexes, or adjudication
  files that contain source text.
- Removing usernames does not anonymize post text.
- This repository’s public-data policy is unchanged; see
  [data_statement.md](data_statement.md).

---

## What “done” would require

A completed human-validation **study** would additionally require, at
minimum: governed local annotations, the hashes above, documented annotator
training, a pre-specified `N` and inclusion rules, and a privacy-reviewed
aggregate report. **This repository does not currently claim that study.**

Tooling:

```bash
python -m validation.sample \
  --input LOCAL_MODEL_RESULTS.csv \
  --output-dir LOCAL_VALIDATION_DIR \
  --sample-size N \
  --seed 42

python -m validation.evaluate \
  --sampling-index LOCAL_VALIDATION_DIR/sampling_index.csv \
  --annotations-a LOCAL_VALIDATION_DIR/annotator_a.csv \
  --annotations-b LOCAL_VALIDATION_DIR/annotator_b.csv \
  --adjudication LOCAL_VALIDATION_DIR/adjudication.csv \
  --output LOCAL_VALIDATION_DIR/validation_report.json
```
