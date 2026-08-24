# Reddit Bias Perception

A modular pipeline to collect, filter, and analyze Reddit posts for bias in AI-generated images.

The project supports data collection via Reddit API, text preprocessing, few-shot classification (LLM), keyword filtering, sentiment analysis, and topic modeling.

**Public repository policy:** this Git tree is intended to distribute source code, configuration, schemas, documentation, and synthetic fixtures. It does **not** distribute record-level Reddit text, Reddit IDs, usernames, or permalinks. See [docs/data_statement.md](docs/data_statement.md), [docs/git_history_audit.md](docs/git_history_audit.md), and [tests/fixtures/synthetic/README.md](tests/fixtures/synthetic/README.md).

---

## Project Structure

```
reddit-bias-perception/
├── Makefile                        # install, format, test, demo
├── README.md
├── main.py                         # shim to processing.run_pipeline
│
├── processing/
│   ├── run_pipeline.py             # canonical CLI (synthetic + legacy private)
│   ├── synthetic_pipeline.py       # offline synthetic workflow
│   ├── keyword_filter.py
│   ├── clean_text.py
│   ├── llm_annotation.py           # yes/no + failure-status contract
│   └── manifest.py                 # run provenance
│
├── validation/                     # blinded sampling + human-vs-model evaluation
├── docs/annotation_codebook.md
├── docs/human_validation_protocol.md
├── tests/fixtures/synthetic/       # fully synthetic public fixtures
├── artifacts/                      # generated demo output (gitignored)
```

---

## Reproducible synthetic demo

This is the **supported public/offline path**. It exercises preprocessing,
keyword filtering, the annotation contract, aggregation, and provenance using
fully synthetic fixtures.

It does **not** reproduce study findings, validate bias results, or recreate
Reddit analyses. Demo outputs are pipeline-validation artifacts.

### Prerequisites

- Python 3.10–3.12
- Poetry

```bash
poetry install --with=dev
```

Dependency installation may need package-index access. After install, the demo
itself is runtime-offline.

### Canonical command

```bash
make demo
```

Equivalent:

```bash
PYTHONPATH=. poetry run python -m processing.run_pipeline --synthetic \
  --input tests/fixtures/synthetic/posts.json \
  --output-dir artifacts/synthetic_demo
```

Expected output directory: `artifacts/synthetic_demo/`

- `synthetic_demo_aggregate.json` — compact counts only
- `synthetic_demo_manifest.json` — input checksum, code SHA, config hash, stage counts

Canonical resume reuses those outputs only when the manifest matches the current
input checksum, config hash, and schema version **and** the recorded aggregate
SHA-256 still matches the file on disk. `code_sha` is stored for provenance but
is not a cache key, so a documentation-only commit does not by itself
invalidate a matching synthetic run.

### What the demo exercises

- loading tracked synthetic fixtures
- canonical text preprocessing and date-window filtering
- keyword filtering and deduplication
- deterministic fake annotation (`synthetic-demo-annotator/v1`)
- aggregate summary + provenance manifest

### What it deliberately does not exercise

- Reddit API / private Reddit corpora
- Hugging Face / Llama / other real LLMs
- BERTopic, GoEmotions, VADER
- GPU or model downloads
- network access at runtime

Demo data are synthetic. No Reddit source text is distributed. No real LLM is
required. Results are not research findings.

`make test` runs the meaningful offline suite, including this demo path.

---

## Real Reddit data (local only)

Real Reddit-derived corpora are obtained separately, remain on the researcher's
machine (typically under `data/`), and are **not** required for the public
synthetic demo. They must not be committed. Provenance and governance for those
files are a research-data concern; this repository does **not** claim that the
full real-data workflow is reproducible from a clean clone.

The legacy runner `python -m processing.run_pipeline --subreddit <name>` still
exists for local private dumps. It may load a real LLM, assumes files under
`data/`, and resumes on file existence only (not provenance). It is not the
supported public command.

---

## Pipelines (legacy / exploratory)

Sentiment analysis and BERTopic modules remain in `analysis/` for local
research. They are **not** part of the canonical demo and are not invoked by
`make demo` or `main.py`.

---

## Development Commands

| Command        | Description                                      |
|----------------|--------------------------------------------------|
| `make install` | Install dependencies with Poetry                 |
| `make format`  | Run black and isort                              |
| `make check`   | Check formatting and config                      |
| `make test`    | Offline tests (`not slow and not integration_external`) |
| `make demo`    | Canonical synthetic end-to-end workflow          |

---

## Configuration

Environment variables and paths are defined in:

- `config/config.py` → all file paths and constants
- `.env` → API keys and credentials

### Example `.env`
```
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=script:reddit_bias:v1.0 (by u/yourname)
HF_TOKEN=hf_your_token
```

---

## Outputs

Pipeline scripts may write record-level files under local `data/` (raw dumps, cleaned corpora, per-record labels). **Those files are local research artifacts and are gitignored.** They are not published in this repository.

Public examples for tests and documentation are synthetic only (`tests/fixtures/synthetic/`). `.gitignore` is written to block accidental recommits of Reddit corpora, archives, databases, and per-record outputs; it cannot erase objects that already exist in Git history. See [docs/git_history_audit.md](docs/git_history_audit.md).

### LLM annotation contract

Few-shot labels are model predictions, not human-validated ground truth.

- On success, `pred_label` is `yes` or `no`.
- On parse or model failure, `pred_label` is null and `status` records the failure.
- Malformed replies are not guessed. Failures are never stored as `"no"`.
- Unsuccessful rows are written to `*_filtered_ai_unclassified.csv` and excluded from yes/no prevalence denominators.

See [docs/llm_annotation.md](docs/llm_annotation.md). Artifacts without `status` are a prior schema; re-run annotation rather than migrating them.

### Annotation validity

The operational yes/no construct is whether a post **discusses visual-identity
bias in AI-generated images** (see [docs/annotation_codebook.md](docs/annotation_codebook.md)).
That is a discourse/perception label. It is not a detector of objective AI-system
bias, and the project name should not be read as a claim that the classifier
“detects AI bias.”

- Automated `yes`/`no` values are **model predictions**, not human-validated
  ground truth.
- This repository includes a reproducible human-validation **framework**
  (blinded deterministic sampling, double annotation, optional adjudication,
  agreement metrics, and aggregate model-vs-human evaluation). See
  [docs/human_validation_protocol.md](docs/human_validation_protocol.md).
- **No completed human-validation result is claimed by this repository.**
  Shipping the framework is not the same as completing a study.
- Human `uncertain` and `insufficient_context` labels are counted but excluded
  from binary evaluation denominators. They are not forced into `yes`/`no`.
- Model `parse_error` / `model_error` rows are reported as execution failures
  and are never treated as scientific `"no"`.

Real validation task files (source text, sampling indexes, annotator CSVs)
remain local/private. Public tests use fictional synthetic fixtures only.

---

## Tests

Test files are under `tests/` and can be run with:

```bash
make test
```

Includes:

- Annotation-contract and parser tests
- Public-artifact privacy regression
- Synthetic end-to-end pipeline regression
- Human-validation framework tests (sampling, agreement, synthetic E2E)
- Other fast offline unit tests

Live Reddit API tests, model downloads, GPU jobs, and BERTopic training are
not part of the default suite.

---

## License

MIT License for the software in this repository. The license does **not** grant rights to redistribute Reddit users' content. See [docs/data_statement.md](docs/data_statement.md).
