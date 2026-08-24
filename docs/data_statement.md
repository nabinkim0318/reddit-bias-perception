# Data statement

This document describes how **Reddit Bias Perception** treats source data,
synthetic fixtures, and public repository artifacts. It is a data-governance
statement, not a software license and not a full methodology paper.

## Project purpose

The project studies discourse about visual-identity representation and bias in
AI-generated imagery: how people talk about portrayal, stereotyping, and
erasure of identity traits in image-generation tools.

## Software license vs data rights

The repository's software license (see `LICENSE` if present, otherwise the
license declared in `pyproject.toml` / `README.md`) covers **source code** and
other files the maintainers choose to distribute as software.

That license **does not** grant rights in third-party content. In particular,
it does not establish permission to redistribute Reddit users' posts, comments,
or other Reddit-derived text. Public accessibility of a Git repository is not
the same thing as a right to republish crawled or archived social-media
records.

This project does **not** claim Reddit redistribution rights from repository
evidence.

## Data source

Research workflows **may** operate on Reddit-derived data acquired separately
and stored only on local research machines (for example under `data/`).

**Record-level Reddit text and stable Reddit identifiers are not distributed
in the public repository.**

Local inputs can include raw exports, extracted archives, filtered corpora, and
per-record model outputs. Those files are private research inputs, not public
repository artifacts.

## Public repository policy

The public repository is intended to contain:

- source code
- configuration and schemas
- synthetic fixtures
- documentation
- governed aggregate outputs, if maintainers later designate specific
  sanitized files for publication

The public repository is **not** intended to contain:

- Reddit post or comment text (raw or cleaned)
- Reddit post IDs or comment IDs
- usernames / authors
- permalinks
- raw archives (including compressed dumps)
- record-level annotations tied to identifiable Reddit records

## Source data vs synthetic fixtures

[`tests/fixtures/synthetic/`](../tests/fixtures/synthetic/) holds **fully
synthetic** records written for tests and documentation. They are not derived
from Reddit users. They must not be replaced with real exports.

## Privacy / re-identification

Removing usernames alone does **not** make Reddit text anonymous.

Verbatim or near-verbatim post text, and stable identifiers (post IDs, comment
IDs, permalinks), can enable re-identification: the same string or ID can be
looked up on Reddit or in third-party archives. Cleaned text that remains
close to the original can carry the same risk.

That is why this repository treats record-level Reddit content and IDs as
nonpublic, even when usernames are absent.

## Sampling limitations

Any local corpus used with this code is purposive and **not** a representative
sample of Reddit or of public opinion. Subreddit selection, keyword selection,
date restrictions, platform visibility (including removed or never-indexed
posts), and collection mechanisms (API, archives, or other local sources) all
shape what appears in a run. This statement does not replace a later
methodology write-up.

## Automated labels

Model-generated labels (for example LLM classifications, sentiment scores, or
topic assignments) are **automated annotations**. They should not be treated as
human-validated ground truth unless a separate validation process has been
documented and applied.

## Takedown / contact

If you believe this repository still exposes content that should not be public:

1. Do **not** open a public issue that pastes the sensitive text or identifiers.
2. Use GitHub's private vulnerability reporting for this repository if it is
   enabled (repository **Security** → **Advisories** / **Report a
   vulnerability**).
3. Otherwise contact the repository owner through GitHub (the owner's profile
   contact options, or any private security contact the repository lists).

This procedure is repository-level. It does not invent a personal email
address.

## Historical data

Older Git history of this project **did** contain record-level research
artifacts (filtered corpora, raw JSON, archives, databases, and per-record
outputs). Removing files from the current tree does not erase them from prior
commits.

The audit performed for this privacy remediation is documented in
[`docs/git_history_audit.md`](git_history_audit.md). Until a follow-up history
sanitization is completed, those historical objects remain recoverable by
anyone with a clone of the repository (including forks and backups made from
it).
