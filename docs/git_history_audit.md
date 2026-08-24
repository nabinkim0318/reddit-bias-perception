# Git history audit (privacy)

**Original audit date:** 2026-08-24  
**Starting branch / HEAD audited (pre-rewrite):** `main` @
`7d0da0f40adba57c6cf1c412fde94ee46607c52d`  
**Additional refs inspected then:** `origin/main`, `origin/local-backup`,
`origin/sentiment`, and the working branch created from that HEAD.

This note records a **non-disclosing** audit: path names, object metadata
(size, blob hash), commit identifiers, and scanner finding *types* only. It
does not quote Reddit text, IDs, or credential values.

## Commands / tools used (original audit)

- `git log --all --pretty=format: --name-only` — unique historical paths
- `git rev-list --objects --all` plus `git cat-file --batch-check` — blob type,
  hash, and size
- `git log --all --oneline -- data/` and pathspecs for named sensitive files
- `git for-each-ref` — local and remote refs
- Custom Python regex scan of **source-like** blobs (≤400 KiB; excluding
  `data/` dumps, archives, and lockfiles) for common credential patterns
- Current-tree `git ls-files` inventory of `data/` (names and sizes only)

The original audit did **not** rewrite history. The follow-up sanitization
is recorded below.

## Categories of historical files found (pre-rewrite)

Record-level or archive-like paths appeared in Git history, including:

| Category | Examples (path classes only) |
| --- | --- |
| Raw Reddit exports | `data/raw/` JSON dumps |
| Compressed archives | `data/raw/` `.zst` |
| Extracted corpora | `data/extracted/` JSONL |
| Filtered / processed record files | `data/filtered/` keyword-filtered, cleaned, preview, and full-filtered tables |
| Per-subreddit processed CSVs | `data/processed/subreddit_csv/` |
| Per-record LLM / filter outputs | `data/processed/` classification and filter result tables |
| Cleaned record-level tables | `data/processed/` cleaned bias tables |
| Sentiment / result tables | `data/results/` labeled outputs |
| Summary tables that can include examples | `data/filtered/_summaries/` |
| Databases | root-level DuckDB files |
| Other named research CSVs | root-level `*_posts*.csv` and similar research dumps |

Approximate sensitive-path blob mass in the object database at original audit
time: **~651 MiB** across **118** matching blobs (plus trees in the earlier
count). Largest blobs were processed CSV/JSONL corpora under `data/`, not
source code.

Those paths were removed from the **current tree** by GitHub PR #2
(`privacy/remove-record-level-reddit-data`). Before sanitization they remained
in **history**.

## Record-level Reddit artifacts in history (pre-rewrite)

**Yes, before sanitization.** Record-level Reddit artifacts were recoverable
from reachable Git history of the published branches.

Deleting files from the current revision does not remove those blobs from the
object database. That is why history rewriting was required.

## Credential exposure (original audit)

**No verified credential exposure** in the source-like blobs scanned.

Findings:

- The only env-like tracked path in history is `.env.example` (placeholders).
- No `.env`, `.pem`, `id_rsa`, or `credentials` filenames appeared in
  historical path lists.
- Regex scan of 393 source-like blobs for common token shapes (AWS access-key
  IDs, GitHub `ghp_` / `github_pat_`, Hugging Face `hf_`, OpenAI-style `sk-`,
  PEM private-key headers, Slack tokens, Google API keys, and non-placeholder
  `REDDIT_CLIENT_*` / `HF_TOKEN` assignments) returned **zero** matches.

This is **not** a claim that no secret has ever existed in any blob: large
`data/` dumps were not content-scanned (to avoid reproducing record-level
text), and scanners cannot prove absence.

## History sanitization (PR 1.5)

**Date:** 2026-08-24  
**Tool:** `git filter-repo` 2.47.0  
**Strategy:** A — path-based removal (plus one historical notebook blob strip)  
**Pre-rewrite default HEAD:** `d7f047f2d9390f1c950d7436c7154621fc746a0c`
(merge of GitHub PR #2 onto `main`)  
**Post-rewrite default HEAD (rewrite commit):**
`4f7d8be7ed8fe3ee58070e8cb025107077b06303`

The current-tree snapshot of PR #2 was preserved exactly (path list and blob
IDs identical before vs after rewrite). One later documentation commit on
rewritten `main` may follow this rewrite HEAD; use `origin/main` as the
canonical next-base.

### Why Strategy A

Sensitive material was concentrated in identifiable data-path classes and a
small set of root-level research dumps. Code, configuration, schemas, docs,
and synthetic fixtures were retainable. Strategy B (orphan / re-root reset)
would have discarded useful code history, including unique collaborator
commits on `sentiment`, without a matching privacy gain once those path
classes and the oversized notebook blob were removed.

### Path classes removed from reachable rewritten history

- `data/raw/`, `data/extracted/`, `data/processed/`, `data/filtered/`,
  `data/results/` (kept `data/README.md`)
- Root-level research dumps matching `*_posts*.csv` and DuckDB files
- Archive/database globs: `*.zst`, `*.duckdb`, `*.sqlite`, `*.sqlite3`,
  `*.parquet`
- Record-level filter / classification / sentiment / topic filename classes
- One historical `summary.ipynb` blob (~433 KiB) that contained embedded
  outputs; current scrubbed notebooks were kept

Synthetic fixtures under `tests/fixtures/synthetic/` were **not** removed.

### Branches and tags

| Ref | Action |
| --- | --- |
| `main` | Rewritten and force-updated (`--force-with-lease`) |
| `sentiment` | Rewritten and force-updated (unique non-data commits retained) |
| `local-backup` | Deleted from remote (obsolete pointer into old history) |
| `privacy/remove-record-level-reddit-data` | Deleted from remote (PR #2 already merged) |
| tags | None existed; none created |

Do **not** start future feature branches from pre-rewrite local copies.
Branch PR 2 work from rewritten `origin/main` only.

### Known sensitive-object reachability

- Pre-rewrite: 118 identified sensitive blobs (~621 MiB blob payload;
  ~651 MiB in the earlier inclusive audit estimate) were reachable from
  published history.
- Post-rewrite published refs / fresh clone: **0 / 118 reachable**. All 118
  were missing from the rewritten object database.

### Size (supporting evidence only)

| Location | Pack size (approx.) |
| --- | --- |
| Pre-rewrite development clone | 115 MiB pack |
| Dedicated rewrite clone after filter-repo | 1.84 MiB pack |
| Fresh post-push clone (ordinary refs) | 584 KiB pack |

Size reduction is supporting evidence, not the acceptance criterion.
Acceptance is path-history + blob-reachability + fresh-clone checks.

### Post-rewrite credential scan

Custom regex scan of 435 source-like blobs (≤400 KiB) in the rewritten
repository: **zero** verified credential findings. Historical env-like paths
remain `.env.example` and `.envrc` only.

### Fresh-clone verification

A new clone of `github.com/nabinkim0318/reddit-bias-perception` after the
force-update:

- Default HEAD is the rewritten `main`
- Ordinary fetched refs: `main`, `sentiment` (no tags; stale branches gone)
- Prohibited historical path classes: **no matches**
- Known sensitive blobs: **not present / not reachable**
- Privacy regression tests: **9 passed, 1 skipped**
- Current notebooks: scrubbed (no stored outputs)
- Synthetic fixture IDs remain `SYNTH-*`

Ordinary `git clone` / `git fetch` of branch tips does **not** download
GitHub `refs/pull/*`.

### Residual remote exposure (not the same as ordinary Git history)

Verified remaining GitHub-side exposure after ref rewrite:

- `refs/pull/1/head` and `refs/pull/2/head` still exist on the GitHub remote
  and still name **pre-rewrite** commit objects.
- GitHub REST/HTML still returned those old commit objects, and API blob
  lookup still returned known sensitive blob objects, after the branch
  force-update.

**GitHub Support data-removal is recommended** so cached commits, PR refs,
and blob objects can be purged. Rewriting published branches does not by
itself expire GitHub caches.

Forks at sanitization time: **0**. Rewriting origin would not rewrite a
third-party fork if one appears later.

Old local clones (including any collaborator clone) still contain the
pre-rewrite object database until they are discarded and replaced. **Do not
push those clones back** to this repository.

### Recontamination controls

- `.gitignore` privacy rules from PR #2 remain in force.
- `tests/unit/test_public_artifact_privacy.py` remains in CI.
- Future PRs must branch from rewritten `origin/main`.
- The private pre-rewrite mirror/bundle is a recovery artifact only; it is
  not part of the public repository.

## Current recommendation

Ordinary published branch/tag history in a fresh clone is **sanitized**.

The repository **is** a safe base for new PRs from rewritten `main`.

Republication remains **conditional** until GitHub Support removes cached
PR refs / commit / blob objects, and until old clones are replaced. Do not
claim that GitHub-hosted cached views are gone; they were still reachable
by old object ID after the force-update.
