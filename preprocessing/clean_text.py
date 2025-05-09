### preprocessing/clean_text.py
"""
Text preprocessing module for AI bias Reddit project.
Performs lowercasing, URL removal, punctuation stripping, and stopword filtering.
Also removes deleted or empty posts.
"""

import json
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import DATA_PATH

STOPWORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([w for w in text.split() if w not in STOPWORDS])
    return text


def preprocess_dataframe(df):
    # Remove deleted/empty content
    df = df[~df["title"].isin(["[deleted]", "[removed]"])]
    df = df[~df["selftext"].isin(["[deleted]", "[removed]"])]
    df = df[df["title"].notnull() | df["selftext"].notnull()]

    # Construct and clean text
    df["full_text"] = (
        df["title"].fillna("")
        + " "
        + df["selftext"].fillna("")
        + " "
        + df["comments"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    )
    df["clean_text"] = df["full_text"].apply(preprocess_text)
    return df


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        posts = json.load(f)
    df = pd.DataFrame(posts)
    df = preprocess_dataframe(df)
    df.to_csv(DATA_PATH.replace(".json", "_clean.csv"), index=False)
    print(f"Preprocessed text saved to {DATA_PATH.replace('.json', '_clean.csv')}")


if __name__ == "__main__":
    main()
