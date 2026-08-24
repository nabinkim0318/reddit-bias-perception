# Git history audit (privacy)

**Audit date:** 2026-08-24  
**Starting branch / HEAD audited:** `main` @
`7d0da0f40adba57c6cf1c412fde94ee46607c52d`  
**Additional refs inspected:** `origin/main`, `origin/local-backup`,
`origin/sentiment`, and the working branch created from that HEAD.

This note records a **non-disclosing** audit: path names, object metadata
(size, blob hash), commit identifiers, and scanner finding *types* only. It
does not quote Reddit text, IDs, or credential values.

## Commands / tools used

- `git log --all --pretty=format: --name-only` — unique historical paths
- `git rev-list --objects --all` plus `git cat-file --batch-check` — blob type,
  hash, and size
- `git log --all --oneline -- data/` and pathspecs for named sensitive files
- `git for-each-ref` — local and remote refs
- Custom Python regex scan of **source-like** blobs (≤400 KiB; excluding
  `data/` dumps, archives, and lockfiles) for common credential patterns
- Current-tree `git ls-files` inventory of `data/` (names and sizes only)

No history-rewriting tools (`git filter-repo`, BFG, force-push, forced branch
replacement) were run as part of this audit.

## Categories of historical files found

Record-level or archive-like paths appeared in Git history, including:

| Category | Examples (paths only) |
| --- | --- |
| Raw Reddit exports | `data/raw/reddit_raw.json` (blobs up to ~10 MiB) |
| Compressed archives | `data/raw/aiwars_submissions.zst` (~7 MiB) |
| Extracted corpora | `data/extracted/aiwars.jsonl` (~52 MiB), `data/extracted/sample_ai_posts.jsonl` |
| Filtered / processed record files | `data/filtered/*_keyword_filtered.{json,csv}`, `*_filtered_cleaned.csv`, `*_duckdb_processed.csv`, `*_sample_preview.csv`, `*_full_filtered_posts*.csv` |
| Per-subreddit processed CSVs | `data/processed/subreddit_csv/*.csv` |
| Per-record LLM / filter outputs | `data/processed/fewshot_classification_results.csv`, `filtered_ai_bias.csv`, `classified_*.csv`, `keywords_filtered_ai_bias.*` |
| Cleaned record-level tables | `data/processed/reddit_bias_data_clean.csv` (blobs up to ~11 MiB) |
| Sentiment / result tables | `data/results/sentiment_labeled.csv`, `sentiment_labeled_output.csv` |
| Summary tables that can include examples | `data/filtered/_summaries/examples_per_category.csv` (~1.5 MiB) |
| Databases | `reddit.duckdb` (~1.3 MiB) |
| Other named research CSV | `depression_posts.csv` (present in history; not in current HEAD) |

Approximate sensitive-path blob mass in the object database at audit time:
**~651 MiB** across **180** matching objects (including trees). Largest blobs
were processed CSV/JSONL corpora under `data/`, not source code.

Current HEAD (pre-remediation) still tracked `data/filtered/*_keyword_filtered.{json,csv}`
files whose schema included Reddit `id` plus `clean_text` (record-level). Those
paths are removed from the **current tree** by the privacy PR that added this
document; they remain in **history**.

## Record-level Reddit artifacts in history

**Yes. Record-level Reddit artifacts remain recoverable in Git history.**

Anyone with a clone of this repository (or a fork / backup that still contains
the relevant commits) can check out older commits and recover filtered text,
stable post IDs, raw JSON, archives, and related per-record outputs listed
above. Deleting files from the current revision does not remove those blobs
from the object database.

## Credential exposure

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

## Limitations

- Blob **contents** of Reddit corpora were not printed or excerpted.
- Secret scanning skipped large and `data/` blobs by design.
- GitHub Issues, Pull Request comments, Actions logs, wikis, and third-party
  forks were not audited.
- Unreachable dangling objects without a path, if any, were not exhaustively
  classified beyond size-ranked blobs from `rev-list --objects --all`.
- Packfile compression means deleted-from-HEAD files can still be present
  until history is rewritten **and** clones are replaced.

## Recommendation

**History rewriting is recommended as a follow-up**, not as part of the current
privacy PR.

There is **no** immediate verified credential exposure that would justify an
emergency rewrite in this change set. There **is** a publication blocker:
record-level Reddit research artifacts remain in Git history and should be
removed with a planned sanitization (for example `git filter-repo` or BFG),
followed by rotating any credentials if a later scan finds them, coordinating
force-updates of published branches, and asking collaborators to re-clone.

Until that follow-up is done, **do not describe the repository as safe to
republish** solely because the current tree has been cleaned.
