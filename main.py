import pandas as pd

from analysis.bertopic_model import run_bertopic_model
from analysis.sentiment_analysis import run_sentiment_analysis
from config.config import (
    FINAL_ANALYSIS_INPUT,
    SENTIMENT_OUTPUT,
    TOPIC_ASSIGNMENT_PATH,
    TOPIC_OUTPUT,
)


def main():
    # 1. Load final filtered dataset
    df = pd.read_csv(FINAL_ANALYSIS_INPUT)
    print(f"📥 Loaded {len(df)} filtered posts for analysis")

    # 2. Run sentiment analysis (GoEmotions + VADER)
    df = run_sentiment_analysis(df)
    df.to_csv(SENTIMENT_OUTPUT, index=False)
    print(f"✅ Sentiment results saved to: {SENTIMENT_OUTPUT}")

    # 3. Run BERTopic modeling
    topic_model, topic_info, doc_topics = run_bertopic_model(df)
    topic_info.to_csv(TOPIC_OUTPUT, index=False)
    doc_topics.to_csv(TOPIC_ASSIGNMENT_PATH, index=False)
    print(f"✅ Topic modeling complete. Saved to: {TOPIC_OUTPUT}")

    print("🎉 Analysis pipeline complete.")


if __name__ == "__main__":
    main()
