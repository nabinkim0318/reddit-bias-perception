### processing/keyword_filter.py
"""
Keyword-based filtering module for AI bias Reddit project.
Filters posts based on presence of bias-related keywords and AI relevance.
"""

import json
import re

import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError

from config.config import (
    AI_KEYWORDS,
    BIAS_KEYWORDS,
    CLEANED_DATA,
    KEYWORDS_FILTERED_DATA,
    SUBREDDIT_GROUPS_PATH,
)
from processing.schema import FilteredAIBiasPost

load_dotenv()


def flatten_keywords(keyword_dict):
    return list({kw.lower() for values in keyword_dict.values() for kw in values})


def match_keywords(text, keywords):
    text = str(text).lower()
    return [kw for kw in keywords if kw in text]


def load_subreddit_groups() -> dict:
    with open(SUBREDDIT_GROUPS_PATH, "r") as f:
        return json.load(f)


def get_subreddit_category(subreddit: str, subreddit_groups: dict) -> str:
    subreddit = subreddit.lower()
    for category, subs in subreddit_groups.items():
        if subreddit in subs:
            return category
    return "unknown"


def infer_bias_types(text, bias_keywords_dict):
    text = str(text).lower()
    matched_types = []
    for bias_type, keywords in bias_keywords_dict.items():
        if any(kw in text for kw in keywords):
            matched_types.append(bias_type)
    return matched_types


def filter_posts_by_keywords(posts, bias_keywords_dict, ai_keywords):
    filtered = []
    ai_flat = flatten_keywords({"ai": ai_keywords})
    bias_flat = flatten_keywords(bias_keywords_dict)
    subreddit_groups = load_subreddit_groups()

    for post in posts:
        content = f"{post.get('title', '')} {post.get('selftext', '')} {' '.join(post.get('comments', []))}".lower()

        # ==== Keywords Matching ===
        matched_bias_keywords = match_keywords(content, bias_flat)
        matched_ai_keywords = match_keywords(content, ai_flat)
        bias_types = infer_bias_types(content, bias_keywords_dict)

        # ==== Required Keywords based on Subreddit Category ===
        subreddit = post["subreddit"]
        category = get_subreddit_category(subreddit, subreddit_groups)

        if category == "casual":
            condition = bool(matched_bias_keywords and matched_ai_keywords)
        elif category == "expert":
            condition = bool(matched_bias_keywords)
        else:
            condition = False

        if condition:
            try:
                enriched = {
                    "id": post["id"],
                    "subreddit": subreddit,
                    "clean_text": post["clean_text"],
                    "bias_types": bias_types,
                    "matched_keywords": list(
                        set(matched_bias_keywords + matched_ai_keywords)
                    ),
                }
                validated = FilteredAIBiasPost(**enriched)
                filtered.append(validated.model_dump())
            except ValidationError as e:
                print(
                    f"❌ Validation error for post {post.get('id')} from r/{subreddit}: {e}"
                )
    return filtered


def main():
    df = pd.read_csv(CLEANED_DATA)
    raw_data = df.to_dict("records")

    keywords_filtered_data = filter_posts_by_keywords(
        raw_data, BIAS_KEYWORDS, AI_KEYWORDS
    )

    # === Save filtered results to JSON for compatibility ===
    with open(KEYWORDS_FILTERED_DATA, "w", encoding="utf-8") as f:
        json.dump(keywords_filtered_data, f, ensure_ascii=False, indent=2)

    # === Also save to CSV for downstream analysis (e.g., topic distribution, sentiment) ===
    df_filtered = pd.DataFrame(keywords_filtered_data)
    filtered_csv_path = KEYWORDS_FILTERED_DATA.replace(".json", ".csv")
    df_filtered.to_csv(filtered_csv_path, index=False)

    print(
        f"Filtered {len(keywords_filtered_data)} posts → saved to:\n"
        f"- JSON: {KEYWORDS_FILTERED_DATA}\n"
        f"- CSV : {filtered_csv_path}"
    )


if __name__ == "__main__":
    main()
