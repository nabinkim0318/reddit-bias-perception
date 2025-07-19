import logging
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stopwords = set(ENGLISH_STOP_WORDS)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stopwords])


# DuckDB → Python: remove stopwords
def main():
    subreddit = "aiwars"
    df = pd.read_csv(f"data/filtered/{subreddit}_full_filtered_posts.csv")
    df["clean_text"] = df["clean_text"].apply(remove_stopwords)
    df["matched_bias_types"] = df["matched_bias_types"].apply(
        lambda x: ", ".join(x) if x else ""
    )
    df["matched_keywords"] = df["matched_keywords"].apply(
        lambda x: ", ".join(x) if x else ""
    )
    df.to_csv(
        f"data/filtered/{subreddit}_full_filtered_posts_cleaned_posts.csv", index=False
    )
    logging.info(
        f"Saved cleaned posts to {f'data/filtered/{subreddit}_full_filtered_posts_cleaned_posts.csv'}"
    )
    logging.info(f"Total number of posts: {len(df)}")


if __name__ == "__main__":
    main()
