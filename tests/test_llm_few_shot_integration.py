import os

import pytest
from dotenv import load_dotenv

load_dotenv()  # Load .env where HF_TOKEN is defined
os.environ["TEMPLATE_PATH"] = "tests/assets/test_prompt_template.j2"
from processing import llm_few_shot


@pytest.fixture(scope="session", autouse=True)
def create_test_template():
    os.makedirs("tests/test_assets", exist_ok=True)
    with open("tests/test_assets/test_template.j2", "w") as f:
        f.write("Dummy template: {{ instruction }} {{ post }}")


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
