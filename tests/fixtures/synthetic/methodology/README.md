# Synthetic methodology fixtures

All records here are **fully synthetic**. Identifiers use the `SYNTH-STAT-`
prefix. They are not copied or paraphrased from Reddit users.

These files exist so topic-stability and clustered-emotion tests can run
without private Reddit corpora, embedding downloads, or GPU jobs.

- `emotion_scores.csv` — fictional posts, mapped categories, and emotion scores
- `topic_assignments.json` — synthetic multi-seed topic labels for the same row order
- `topic_to_category.json` — a run-specific mapping bound to the seed-11 solution
  (checksum is filled by tests from the assignment vector)

Do not replace these records with real exports or real BERTopic output.
