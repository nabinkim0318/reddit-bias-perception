# tests/unit/test_clean_text.py

import pytest

from processing.clean_text import is_valid_content, preprocess_text


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
