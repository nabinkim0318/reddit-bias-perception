### processing/clean_text.py
"""
Text preprocessing module for AI bias Reddit project.
Performs lowercasing, URL removal, punctuation stripping, and stopword filtering.
Also removes deleted or empty posts.
"""

import json
import re

import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config.config import CLEANED_DATA, RAW_REDDIT_DATA

load_dotenv()

STOPWORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([w for w in text.split() if w not in STOPWORDS])
    return text


def is_valid_content(text):
    return str(text).strip().lower() not in {
        "[deleted]",
        "[removed]",
        "",
        "none",
        "null",
    }


def preprocess_dataframe(df):
    # Clean comments to ensure consistency
    df["comments"] = df["comments"].apply(lambda x: x if isinstance(x, list) else [])

    # Remove deleted/empty content
    df = df[
        df["title"].apply(is_valid_content) | df["selftext"].apply(is_valid_content)
    ]

    # Construct and clean text
    df["full_text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")
    df["clean_text"] = df["full_text"].apply(preprocess_text)

    # Order columns for clarity
    columns_to_keep = [
        "id",
        "subreddit",
        "title",
        "selftext",
        "comments",
        "full_text",
        "clean_text",
    ]
    return df[columns_to_keep]


def main(input_path=None):
    path = input_path or RAW_REDDIT_DATA
    with open(path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    df = pd.DataFrame(posts)
    df = preprocess_dataframe(df)
    df.to_csv(CLEANED_DATA, index=False)
    print(f"✅ Preprocessed text saved to {CLEANED_DATA}")


if __name__ == "__main__":
    main()
