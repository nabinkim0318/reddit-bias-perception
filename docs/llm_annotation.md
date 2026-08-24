# LLM annotation contract

Automated few-shot labels are **model predictions**, not human-validated
ground truth. This document describes schema version 2.

## Operational construct

The yes/no label answers whether the post **discusses visual-identity bias
in AI-generated images** (unfair, distorted, or missing portrayal of human
identity in generated images). It does **not** measure whether an AI system
objectively exhibits bias, and it is not a general “social bias” detector.

See [annotation_codebook.md](annotation_codebook.md) and
[human_validation_protocol.md](human_validation_protocol.md). This
repository does not claim that a human-validation study has been completed.

## Status and label

| `status` | `pred_label` | Meaning |
|---|---|---|
| `success` | `yes` or `no` | Parsed a valid contract response |
| `parse_error` | null | Output was empty, malformed, or ambiguous |
| `model_error` | null | Load, generation, or cardinality failure |

`pred_label` is the scientific yes/no construct. It is set only on `success`.
Failures are never recorded as `"no"`.

`refusal` and `timeout` are **not** assigned. The current Hugging Face
`generate` path cannot distinguish those states deterministically. A
recognizable refusal or empty reply is `parse_error`.

`error_type` is a compact machine-readable code (`empty_output`,
`malformed_output`, `ambiguous_output`, `generation_exception`,
`model_unavailable`, `output_cardinality_mismatch`, `validation_error`).
Raw model text is not copied into logs or error fields.

## Parser

After trim, surrounding quotes, and case folding, the parser accepts:

- the entire response `yes` or `no`
- a line `Label: yes` or `Label: no` (the prompt contract)

It does **not** guess from prose, hedging, both labels, JSON objects, or
substring matches such as “contains yes”. Those cases are `parse_error`
with a null label.

Existing artifacts without `status` are a prior schema. Re-run annotation
rather than migrating them.

## Split outputs

- `success` + `yes` → `*_filtered_ai_bias.csv`
- `success` + `no` → `*_filtered_ai_non_bias.csv`
- anything else, including `success` with a null label → `*_filtered_ai_unclassified.csv`
- all rows → `*_llm_classification_results.csv`

Failed records are excluded from yes/no prevalence denominators. Run logs
report `success_yes`, `success_no`, `parse_error`, and `model_error`
separately.
