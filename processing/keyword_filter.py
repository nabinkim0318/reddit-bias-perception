### processing/keyword_filter.py
"""
Keyword-based filtering module for AI bias Reddit project.
Filters posts based on presence of bias-related keywords and AI relevance.
"""

import json
import re

from dotenv import load_dotenv

from config.config import AI_KEYWORDS, BIAS_KEYWORDS, CLASSIFIED_BIAS, FILTERED_DATA

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

        # Mandatory condition: need to have
        matched_bias_keywords = match_keywords(
            content, flatten_keywords(bias_keywords_dict)
        )
        matched_ai_keywords = match_keywords(content, ai_flat)
        bias_types = infer_bias_types(content, bias_keywords_dict)

        if matched_bias_keywords and matched_ai_keywords:
            post["matched_bias_keywords"] = matched_bias_keywords
            post["matched_ai_keywords"] = matched_ai_keywords
            post["bias_types"] = ";".join(bias_types)
            filtered.append(post)
    return filtered


def main():
    with open(CLASSIFIED_BIAS, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    bias_flat = flatten_keywords(BIAS_KEYWORDS)
    filtered_data = filter_posts(raw_data, bias_flat, AI_KEYWORDS)

    with open(FILTERED_DATA, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(
        f"Filtered {len(filtered_data)} posts from {len(raw_data)} raw posts → saved to {FILTERED_DATA}"
    )


if __name__ == "__main__":
    main()
