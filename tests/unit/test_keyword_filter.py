## tests/unit/test_keyword_filter.py

import pytest

from processing.keyword_filter import (
    filter_posts,
    flatten_keywords,
    infer_bias_types,
    match_keywords,
)

BIAS_KEYWORDS = {
    "gender": ["gender", "woman", "man", "female", "male"],
    "age": ["old", "young"],
    "race": ["black", "white", "asian"],
    "general_bias": ["bias", "discrimination", "stereotype"],
}

AI_KEYWORDS = [
    "ai",
    "artificial intelligence",
    "diffusion",
    "deep learning",
    "machine learning",
    "neural network",
    "stable diffusion",
    "dalle",
    "midjourney",
    "openai",
    "text-to-image",
    "image generation",
    "chatgpt",
]


def test_flatten_keywords():
    result = flatten_keywords(BIAS_KEYWORDS)
    assert isinstance(result, list)
    assert "woman" in result and "bias" in result
    assert len(result) == len(set(result))


@pytest.mark.parametrize(
    "text,keywords,expected",
    [
        ("The AI seems biased", ["ai", "bias", "woman"], ["ai", "bias"]),
        ("I tested dalle and diffusion", AI_KEYWORDS, ["dalle", "diffusion"]),
        ("Nothing AI-related", ["openai", "chatgpt"], []),
    ],
)
def test_match_keywords(text, keywords, expected):
    result = match_keywords(text, keywords)
    assert set(result) == set(expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This AI only generates white males", ["gender", "race"]),
        ("Discrimination in models", ["general_bias"]),
        ("Nothing here", []),
    ],
)
def test_infer_bias_types(text, expected):
    result = infer_bias_types(text, BIAS_KEYWORDS)
    assert set(result) == set(expected)


def test_filter_posts_realistic():
    posts = [
        {
            "id": "abc123",
            "subreddit": "aiart",
            "title": "AI generates young white males",
            "selftext": "Feels stereotypical",
            "comments": ["No diversity"],
            "clean_text": "AI generates young white males Feels stereotypical No diversity",
        },
        {
            "id": "xyz789",
            "subreddit": "news",
            "title": "Not related",
            "selftext": "No AI here",
            "comments": [],
            "clean_text": "No AI here",
        },
    ]
    filtered = filter_posts(posts, BIAS_KEYWORDS, AI_KEYWORDS)
    assert len(filtered) == 1
    assert filtered[0]["id"] == "abc123"
    assert isinstance(filtered[0]["bias_types"], list)
    assert set(filtered[0]["bias_types"]) & {"gender", "race", "general_bias"}
