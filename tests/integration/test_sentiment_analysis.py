import pytest

from analysis.sentiment_analysis import run_goemotions, run_vader


@pytest.mark.parametrize(
    "text",
    ["This image looks biased.", "I love the output!", "Why are all characters male?"],
)
def test_goemotions_output(text):
    result = run_goemotions(text)
    print("DEBUG result:", result)

    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], list)
    assert all(
        isinstance(em, tuple) and isinstance(em[0], str) and isinstance(em[1], float)
        for em in result[0]
    )


@pytest.mark.parametrize(
    "text",
    [
        "This image looks biased.",
        "I love the output!",
    ],
)
def test_vader_output(text):
    scores = run_vader(text)
    assert isinstance(scores, dict)
    assert "compound" in scores
    assert -1.0 <= scores["compound"] <= 1.0
