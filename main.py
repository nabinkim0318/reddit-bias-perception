### main.py

import pandas as pd
from modeling.bertopic_model import run_bertopic
from modeling.lda_model import run_lda
from preprocessing.clean_text import preprocess_dataframe

from analysis.topic_filter import extract_bias_topics
from analysis.visualization import visualize_topics
from config import DATA_PATH
from reddit_crawler.fetch_data import fetch_all_subreddits


def main():
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} posts")

    # 2. Preprocess
    df = preprocess_dataframe(df)

    # 3. BERTopic Modeling
    topic_model, topic_info, df_with_topics = run_bertopic(df)

    # 4. Filter bias topics
    bias_df = extract_bias_topics(topic_info, df_with_topics)

    # 5. Visualization
    visualize_topics(topic_model, bias_df)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
