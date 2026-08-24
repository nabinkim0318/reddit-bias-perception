# Synthetic fixtures

All records in this directory are **fully synthetic**.

They are not copied, adapted, or paraphrased from Reddit users, Reddit posts,
or Reddit comments. Identifiers use an unmistakably synthetic `SYNTH-` prefix
and must not be treated as real Reddit IDs.

These fixtures exist so tests, documentation, and later end-to-end
reproducibility work can run **without** private Reddit corpora.

Schema coverage (fictional content, representative fields):

- a normal post that would match configured AI/bias keyword filters
- empty / missing body
- deleted / removed-style title and body
- a duplicated synthetic ID
- a non-matching post (no AI/bias keyword overlap)
- posts labeled with each configured subreddit group name
- crawler-shaped fields (`title`, `selftext`, `comments`, scores) without
  `author`, `username`, or `permalink`
- keyword-filtered output fields (`clean_text`, `matched_keywords`, …)
- synthetic subreddit-group assignments used by the offline demo
- methodology fixtures under `methodology/` (fictional topic assignments and emotion scores)

Do not replace these records with real Reddit exports.
