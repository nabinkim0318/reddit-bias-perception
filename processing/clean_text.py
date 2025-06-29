### processing/clean_text.py
"""
Text preprocessing module for AI bias Reddit project.
Performs lowercasing, URL removal, punctuation stripping, and stopword filtering.
Also removes deleted or empty posts and cleans top 10 comments.
Logs length changes throughout the pipeline.
"""


import json
import logging
import re

import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config.config import CLEANED_DATA, RAW_REDDIT_DATA, TOP_COMMENT_NUM

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

STOPWORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text):
    """
    Clean and normalize a text string.

    Applies lowercasing, URL and HTML removal, punctuation stripping,
    and stopword filtering.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned and processed text.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([w for w in text.split() if w not in STOPWORDS])
    return text


def is_valid_content(text):
    """
    Check whether the text content is valid (not deleted or empty).

    Args:
        text (str): Post title or selftext.

    Returns:
        bool: True if the content is valid, False otherwise.
    """
    return str(text).strip().lower() not in {
        "[deleted]",
        "[removed]",
        "",
        "none",
        "null",
    }


def clean_comment_list(comments):
    """
    Preprocess up to the first 10(TOP_COMMENT_NUM) comments in a comment list.

    Args:
        comments (list[str]): List of comment strings.

    Returns:
        list[str]: Cleaned comments.
    """
    if not isinstance(comments, list):
        return []
    return [preprocess_text(c) for c in comments[:TOP_COMMENT_NUM]]


def preprocess_dataframe(df):
    """
    Perform full text preprocessing on a Reddit posts DataFrame.

    This includes:
    - Filtering out deleted/empty content
    - Creating full_text and clean_text
    - Preprocessing top n comments
    - Ensuring all required schema columns

    Args:
        df (pd.DataFrame): Raw Reddit post data.

    Returns:
        pd.DataFrame: Cleaned and structured DataFrame.
    """
    logging.info(f"Initial number of rows: {len(df)}")

    # Clean comments to ensure consistency
    df["comments"] = df["comments"].apply(lambda x: x if isinstance(x, list) else [])

    # Filter invalid posts
    df_valid = df[
        df["title"].apply(is_valid_content) | df["selftext"].apply(is_valid_content)
    ].copy()
    logging.info(f"Rows after filtering invalid content: {len(df_valid)}")

    # Full text fields
    df_valid["full_text"] = (
        df_valid["title"].fillna("") + " " + df_valid["selftext"].fillna("")
    )
    df_valid["clean_text"] = df_valid["full_text"].apply(preprocess_text)

    # Process top 10 comments
    df_valid["top_comments"] = df_valid["comments"].apply(clean_comment_list)

    logging.info(f"Rows after preprocessing: {len(df_valid)}")

    # Ensure all expected output columns exist
    expected_columns = [
        "id",
        "subreddit",
        "title",
        "selftext",
        "comments",
        "top_comments",
        "full_text",
        "clean_text",
        "score",
        "num_comments",
        "upvote_ratio",
        "flair",
        "created_utc",
    ]
    for col in expected_columns:
        if col not in df_valid.columns:
            df_valid[col] = None

    return df_valid[expected_columns]


def main(input_path=None):
    """
    Main entry point for cleaning Reddit data.

    Reads from the input JSON, processes it, and saves as CSV.

    Args:
        input_path (str, optional): Path to input JSON file.
            Defaults to RAW_REDDIT_DATA.
    """
    path = input_path or RAW_REDDIT_DATA
    with open(path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    df = pd.DataFrame(posts)
    df = preprocess_dataframe(df[:100])
    df.to_csv(CLEANED_DATA, index=False)
    logging.info(f"✅ Preprocessed text saved to {CLEANED_DATA}")
    print(f"✅ Preprocessed text saved to {CLEANED_DATA}")


if __name__ == "__main__":
    main()
