# tests/unit/test_clean_text.py

import os
import sys

import pandas as pd
import pytest

# Ensure module path is correct
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing.clean_text import clean_comment_list, is_valid_content, preprocess_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Check this out: https://example.com", "check"),
        ("<div>Hello!</div>", "hello"),
        ("Just... wow!!!", "just wow"),
        ("THE AI IS BIASED", "ai biased"),
    ],
)
def test_preprocess_text(text, expected):
    assert preprocess_text(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("[deleted]", False),
        ("[removed]", False),
        ("null", False),
        ("Actually a real post", True),
        ("", False),
        (None, False),
    ],
)
def test_is_valid_content(text, expected):
    assert is_valid_content(text) == expected


@pytest.mark.parametrize(
    "comments,expected",
    [
        (
            ["Nice!", "Wow!!", "Helpful link: http://foo.com"],
            [
                preprocess_text("Nice!"),
                preprocess_text("Wow!!"),
                preprocess_text("Helpful link: http://foo.com"),
            ],
        ),
        (None, []),
        ("not a list", []),
        ([], []),
    ],
)
def test_clean_comment_list(comments, expected):
    assert clean_comment_list(comments) == expected
