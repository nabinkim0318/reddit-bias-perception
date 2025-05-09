import os

import pytest
from dotenv import load_dotenv

from processing import llm_few_shot

load_dotenv()  # Load .env where HF_TOKEN is defined


@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN not set in environment")
def test_classify_post_real_model():
    # Load real model and tokenizer
    tokenizer, model = llm_few_shot.load_model(llm_few_shot.MODEL_ID)

    # Simple post that should ideally return Yes or No
    sample_post = "This AI image generator mostly creates white people. Feels biased."

    label, output = llm_few_shot.classify_post(sample_post, tokenizer, model)

    print(f"Label: {label}")
    print(f"Raw Output: {output}")

    assert label in ["Yes", "No", "Uncertain"]  # Should be valid classification
