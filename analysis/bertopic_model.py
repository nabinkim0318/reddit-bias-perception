### modeling/bertopic_model.py
"""
BERTopic modeling on final AI bias dataset.
Performs topic modeling using BERTopic with custom stopwords and saves topic info.
Also saves post-topic assignments with probabilities.
"""

import os

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from config import AI_KEYWORDS

INPUT_PATH = os.getenv("BERTopic_INPUT", "data/processed/ai_bias_final.csv")
TOPIC_INFO_PATH = os.getenv("BERTopic_OUTPUT", "data/results/bertopic_topic_info.csv")
TOPIC_ASSIGNMENT_PATH = os.getenv(
    "BERTopic_TOPIC_DOCS", "data/results/bertopic_post_topics.csv"
)

# Combine default and custom stopwords
CUSTOM_STOPWORDS = set(
    [
        "use",
        "know",
        "thing",
        "like",
        "just",
        "really",
        "even",
        "think",
        "get",
        "used",
        "dont",
        "make",
        "want",
        "need",
        "see",
        "im",
        "people",
        "post",
        "prompt",
        "prompts",
        "image",
        "images",
        "mj",
        "midjourney",
        "dalle",
        "openai",
        "transformer",
        "resnets",
        "layer",
        "weight",
        "param",
        "train",
        "loss",
        "epoch",
        "tokenizer",
        "attention",
        "model",
        "error",
        "ckpt",
        "yaml",
        "diffusers",
        "setup",
        "api",
        "automatic1111",
        "torch",
        "cuda",
        "rtx",
        "comfyui",
        "training",
        "checkpoint",
        "ai",
        "models",
        "sd",
        "sdxl",
        "movie",
        "movies",
        "human",
        "new",
        "good",
        "time",
        "feel",
        "using",
        "seen",
        "character",
        "thing",
        "image",
    ]
)

STOPWORDS = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)


def run_bertopic_model(df):
    docs = df["text"].fillna("").astype(str).tolist()
    vectorizer_model = CountVectorizer(stop_words=STOPWORDS, min_df=2)
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=True,
        nr_topics="auto",
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(docs)
    topic_info = topic_model.get_topic_info()

    doc_topics = pd.DataFrame(
        {
            "post_id": df.get("id", pd.Series(range(len(df)))),
            "topic": topics,
            "probability": [
                max(p) if isinstance(p, list) and p else None for p in probs
            ],
        }
    )
    doc_topics = pd.merge(
        doc_topics, topic_info, how="left", left_on="topic", right_on="Topic"
    )

    return {"model": topic_model, "topic_info": topic_info, "doc_topics": doc_topics}


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
