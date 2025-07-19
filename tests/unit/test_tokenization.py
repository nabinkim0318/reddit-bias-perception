# tests/unit/test_tokenization.py
"""
import pytest
from transformers import AutoTokenizer

from utils.tokenize import batch_tokenize


@pytest.mark.parametrize(
    "input_texts,expected_len",
    [
        (["Hello world!"], 1),
        (["", "  ", None], 3),
        (["test"] * 5, 5),
    ],
)
def test_batch_tokenize_shapes(input_texts, expected_len):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    output = batch_tokenize(input_texts, tokenizer)
    assert output["input_ids"].shape[0] == expected_len
"""