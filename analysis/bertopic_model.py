### modeling/bertopic_model.py
"""
BERTopic modeling on final AI bias dataset.
Performs topic modeling using BERTopic with reusable vectorizer config.
Also saves topic info and document-topic assignments with probabilities.
"""

import os

import pandas as pd
from bertopic import BERTopic
from vectorizer_config import vectorizer_model

from config import AI_KEYWORDS

INPUT_PATH = os.getenv("BERTopic_INPUT", "data/processed/ai_bias_final.csv")
TOPIC_INFO_PATH = os.getenv("BERTopic_OUTPUT", "data/results/bertopic_topic_info.csv")
TOPIC_ASSIGNMENT_PATH = os.getenv(
    "BERTopic_TOPIC_DOCS", "data/results/bertopic_post_topics.csv"
)


def run_bertopic_model(df):
    docs = df["clean_text"].fillna("").astype(str).tolist()

    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=True,
        nr_topics="auto",
        verbose=True,
    )
    topic_model.fit(docs)
    topic_info = topic_model.get_topic_info()

    # 🔍 document-topic info extraction (BERTopic >= 0.14.0)
    doc_info = topic_model.get_document_info(docs)
    doc_info["post_id"] = df.get("id", pd.Series(range(len(df))))
    doc_info["input_text"] = df["clean_text"]
    doc_info["raw_text"] = df["full_text"]

    return topic_model, topic_info, doc_info


def main():
    df = pd.read_csv(INPUT_PATH)
    topic_model, topic_info, doc_topics = run_bertopic_model(df)
    topic_info.to_csv(TOPIC_INFO_PATH, index=False)
    doc_topics.to_csv(TOPIC_ASSIGNMENT_PATH, index=False)

    print(f"✅ BERTopic modeling complete.")
    print(f"- Topic summary saved to: {TOPIC_INFO_PATH}")
    print(f"- Post-topic assignments saved to: {TOPIC_ASSIGNMENT_PATH}")


if __name__ == "__main__":
    main()
