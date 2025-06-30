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

from config.config import AI_KEYWORDS, BIAS_KEYWORDS, CLASSIFIED_YES, FILTERED_DATA
from processing.schema import FilteredAIBiasPost

load_dotenv()


def flatten_keywords(keyword_dict):
    return list({kw.lower() for values in keyword_dict.values() for kw in values})


def match_keywords(text, keywords):
    text = str(text).lower()
    return [kw for kw in keywords if kw in text]


def infer_bias_types(text, bias_keywords_dict):
    text = str(text).lower()
    matched_types = []
    for bias_type, keywords in bias_keywords_dict.items():
        if any(kw in text for kw in keywords):
            matched_types.append(bias_type)
    return matched_types


def filter_posts(posts, bias_keywords_dict, ai_keywords):
    filtered = []
    ai_flat = flatten_keywords({"ai": ai_keywords})

    for post in posts:
        content = f"{post.get('title', '')} {post.get('selftext', '')} {' '.join(post.get('comments', []))}".lower()

        matched_bias_keywords = match_keywords(
            content, flatten_keywords(bias_keywords_dict)
        )
        matched_ai_keywords = match_keywords(content, ai_flat)
        bias_types = infer_bias_types(content, bias_keywords_dict)

        if matched_bias_keywords and matched_ai_keywords:
            try:
                enriched = {
                    "id": post["id"],
                    "subreddit": post["subreddit"],
                    "clean_text": post["clean_text"],
                    "matched_keywords": list(
                        set(matched_bias_keywords + matched_ai_keywords)
                    ),
                    "bias_types": bias_types,
                }
                validated = FilteredAIBiasPost(**enriched)
                filtered.append(validated.model_dump())
            except ValidationError as e:
                print(f"❌ Validation error for post {post.get('id')}: {e}")
    return filtered


def main():
    df = pd.read_csv(CLASSIFIED_YES)
    raw_data = df.to_dict("records")

    filtered_data = filter_posts(raw_data, BIAS_KEYWORDS, AI_KEYWORDS)

    # === Save filtered results to JSON for compatibility ===
    with open(FILTERED_DATA, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    # === Also save to CSV for downstream analysis (e.g., topic distribution, sentiment) ===
    df_filtered = pd.DataFrame(filtered_data)
    filtered_csv_path = FILTERED_DATA.replace(".json", ".csv")
    df_filtered.to_csv(filtered_csv_path, index=False)

    print(
        f"Filtered {len(filtered_data)} posts → saved to:\n"
        f"- JSON: {FILTERED_DATA}\n"
        f"- CSV : {filtered_csv_path}"
    )


if __name__ == "__main__":
    main()
