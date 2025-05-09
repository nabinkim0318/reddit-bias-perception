### filtering/keyword_filter.py
"""
Keyword-based filtering module for AI bias Reddit project.
Filters posts based on presence of bias-related keywords and AI relevance.
"""

import json
import re

from config import AI_KEYWORDS, BIAS_KEYWORDS, DATA_PATH, RAW_DATA_PATH


def flatten_keywords(keyword_dict):
    return list({kw.lower() for values in keyword_dict.values() for kw in values})


def contains_keywords(text, keywords):
    text = str(text).lower()
    return any(kw in text for kw in keywords)


def filter_posts(posts, bias_keywords, ai_keywords):
    filtered = []
    for post in posts:
        content = f"{post.get('title', '')} {post.get('selftext', '')} {' '.join(post.get('comments', []))}".lower()
        if contains_keywords(content, bias_keywords) and contains_keywords(
            content, ai_keywords
        ):
            filtered.append(post)
    return filtered


def main():
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    bias_flat = flatten_keywords(BIAS_KEYWORDS)
    filtered_data = filter_posts(raw_data, bias_flat, AI_KEYWORDS)

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(
        f"Filtered {len(filtered_data)} posts from {len(raw_data)} raw posts → saved to {DATA_PATH}"
    )


if __name__ == "__main__":
    main()
