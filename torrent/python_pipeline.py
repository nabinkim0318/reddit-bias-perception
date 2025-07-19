# torrent/python_pipeline.py
import logging
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from config.config import BASE_DIR

stopwords = set(ENGLISH_STOP_WORDS)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stopwords])


def run(subreddit: str):
    input_path = f"{BASE_DIR}/processed/filtered/{subreddit}_full_filtered_posts.csv"
    output_path = f"{BASE_DIR}/processed/filtered/{subreddit}_full_filtered_posts_cleaned_posts.csv"

    df = pd.read_csv(input_path)
    df["clean_text"] = df["clean_text"].apply(remove_stopwords)

    df["matched_bias_types"] = df["matched_bias_types"].apply(
        lambda x: ", ".join(eval(x)) if isinstance(x, str) and x.startswith("[") else x
    )
    df["matched_keywords"] = df["matched_keywords"].apply(
        lambda x: ", ".join(eval(x)) if isinstance(x, str) and x.startswith("[") else x
    )

    df.to_csv(output_path, index=False)

    logging.info(f"✅ Saved cleaned posts to {output_path}")
    logging.info(f"📄 Total number of posts: {len(df)}")


if __name__ == "__main__":
    run("aiwars")