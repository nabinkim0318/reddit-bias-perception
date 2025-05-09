"""
BERTopic modeling on final AI bias dataset.
Performs topic modeling using BERTopic with reusable vectorizer config.
Also saves topic info and document-topic assignments with probabilities.

| post_id | topic | probability | Name    | Representative Words                                  | raw_text                                                                                     |
| -------- | ----- | ----------- | ------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| abc123   | 0     | 0.85        | Topic 0 | `["bias", "image", "ai", "gender", "representation"]` | "I noticed all AI-generated portraits show women with the same unrealistic proportions." |
| def456   | 1     | 0.92        | Topic 1 | `["model", "training", "data", "epoch", "cuda"]`      | "I'm training a diffusion model and keep getting CUDA errors at epoch 3."                |
| xyz789   | -1    | None        | Outlier | `[]`                                                  | "Sometimes I wonder if machines dream in color or code."                                 |

"""

import pandas as pd
from bertopic import BERTopic
from dotenv import load_dotenv

from config.config import FINAL_ANALYSIS_INPUT, TOPIC_ASSIGNMENT_PATH, TOPIC_OUTPUT
from config.vectorizer_config import vectorizer_model

load_dotenv()


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
    df = pd.read_csv(FINAL_ANALYSIS_INPUT)
    topic_model, topic_info, doc_topics = run_bertopic_model(df)
    topic_info.to_csv(TOPIC_OUTPUT, index=False)
    doc_topics.to_csv(TOPIC_ASSIGNMENT_PATH, index=False)

    print(f"✅ BERTopic modeling complete.")
    print(f"- Topic summary saved to: {TOPIC_OUTPUT}")
    print(f"- Post-topic assignments saved to: {TOPIC_ASSIGNMENT_PATH}")


if __name__ == "__main__":
    main()
