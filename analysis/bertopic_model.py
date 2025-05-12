"""
BERTopic modeling on final AI bias dataset.

This module performs topic modeling using BERTopic with a custom vectorizer configuration.
It generates two outputs:
1. A per-topic summary with top representative words and topic statistics.
2. A per-document topic assignment file with metadata and topic probability.

Output files:
- `bertopic_topic_info.csv`: Summary of discovered topics.
- `bertopic_post_topics.csv`: Document-to-topic mapping with metadata and confidence.

Schema:

📄 `bertopic_topic_info.csv`
| topic | count | representative_words                          |
|-------|-------|------------------------------------------------|
| 0     | 210   | ["bias", "image", "ai", "gender", "representation"] |
| 1     | 180   | ["model", "training", "data", "epoch", "cuda"]     |
| -1    |  35   | []                                             |

📄 `bertopic_post_topics.csv`
| id     | subreddit | clean_text          | topic | topic_probability | bias_types        | score | num_comments | upvote_ratio | flair | created_utc |
|--------|-----------|---------------------|-------|-------------------|-------------------|-------|--------------|--------------|--------|--------------|
| abc123 | aiart     | all outputs are men | 0     | 0.85              | ["gender"]        | 120   | 15           | 0.97         | null   | 1687450394   |
| def456 | machineAI | cuda crashes again  | 1     | 0.92              | ["general_bias"]  | 14    | 3            | 0.74         | null   | 1687450100   |

Notes:
- Topics with `topic == -1` are considered outliers and are excluded from topic summaries.
- `representative_words` are the top terms per topic (from BERTopic).
- `topic_probability` reflects confidence in the topic assignment for each document.
- `bias_types` are inherited from keyword-based filtering step.
- Other metadata is merged from the cleaned Reddit dataset.
"""

import logging
import os

import pandas as pd
from bertopic import BERTopic
from dotenv import load_dotenv

from config.config import (
    CLEANED_DATA,  # Path to full cleaned Reddit data (includes metadata)
)
from config.config import (
    FINAL_ANALYSIS_INPUT,  # Path to filtered posts (AI bias-related)
)
from config.config import (
    TOPIC_ASSIGNMENT_PATH,  # Output path for document-topic mapping
)
from config.config import TOPIC_MODEL_PATH  # Path to save the trained BERTopic model
from config.config import TOPIC_OUTPUT  # Output path for topic-level summary
from config.vectorizer_config import vectorizer_model  # Custom vectorizer for BERTopic

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)


def run_bertopic_model(df):
    """
    Run BERTopic modeling on the input DataFrame and return topic model, topic summary, and document-topic mappings.

    Args:
        df (pd.DataFrame): Preprocessed and filtered dataset containing 'clean_text' and metadata.

    Returns:
        BERTopic: trained topic model
        pd.DataFrame: topic-level summary (topic, count, representative_words, avg_probability)
        pd.DataFrame: post-topic mapping with topic assignment and metadata
    """
    # Prepare text for modeling
    docs = df["clean_text"].fillna("").astype(str).tolist()

    # Instantiate and fit BERTopic model
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=True,
        nr_topics="auto",
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(docs)

    # Build document-topic assignment DataFrame
    doc_info = topic_model.get_document_info(docs)
    doc_info["id"] = df.get("id", pd.Series(range(len(df))))
    doc_info["clean_text"] = df["clean_text"]
    doc_info["topic"] = topics

    # Assign per-document topic probability (skip if topic is -1 or probs missing)
    doc_info["topic_probability"] = [
        p[t] if p is not None and t != -1 else None for p, t in zip(probs, topics)
    ]

    # Merge metadata and bias labels into the document info
    optional_cols = [
        "subreddit",
        "bias_types",
        "score",
        "num_comments",
        "upvote_ratio",
        "flair",
        "created_utc",
    ]
    for col in optional_cols:
        if col in df.columns:
            doc_info[col] = df[col].values

    # Get topic-level summary
    topic_info = topic_model.get_topic_info()
    topic_info["representative_words"] = topic_info["Topic"].apply(
        lambda t: [w for w, _ in topic_model.get_topic(t)]
    )

    # Compute average topic probability across all assigned docs
    prob_df = pd.DataFrame({"topic": topics, "prob": doc_info["topic_probability"]})
    avg_probs = prob_df.groupby("topic")["prob"].mean().rename("avg_probability")
    topic_info = topic_info.merge(
        avg_probs, left_on="Topic", right_on="topic", how="left"
    ).drop(columns=["topic"])

    logging.info(f"Discovered {len(topic_info)} topics (including outlier -1)")
    return topic_model, topic_info, doc_info


def main():
    """
    Load input datasets, run BERTopic, and save results to disk.
    Merges filtered bias data with full metadata and outputs:
        - topic_info summary
        - post-topic mappings
        - serialized BERTopic model
    """
    # Load filtered AI bias posts and clean metadata
    df_filtered = pd.read_csv(FINAL_ANALYSIS_INPUT)
    df_meta = pd.read_csv(CLEANED_DATA)

    # Merge metadata (score, comments, etc.) into filtered dataset
    df = pd.merge(
        df_filtered,
        df_meta[
            ["id", "score", "num_comments", "upvote_ratio", "flair", "created_utc"]
        ],
        on="id",
        how="left",
    )
    logging.info(f"Loaded {len(df)} records after merging filtered + metadata")

    # Run BERTopic modeling
    topic_model, topic_info, doc_topics = run_bertopic_model(df)

    # Save all outputs
    os.makedirs(os.path.dirname(TOPIC_OUTPUT), exist_ok=True)
    topic_info.to_csv(TOPIC_OUTPUT, index=False)
    doc_topics.to_csv(TOPIC_ASSIGNMENT_PATH, index=False)
    topic_model.save(TOPIC_MODEL_PATH)

    logging.info(f"✅ Topic summary saved to: {TOPIC_OUTPUT}")
    logging.info(f"✅ Post-topic assignments saved to: {TOPIC_ASSIGNMENT_PATH}")
    logging.info("✅ BERTopic model saved to: models/bertopic_model")


if __name__ == "__main__":
    main()
