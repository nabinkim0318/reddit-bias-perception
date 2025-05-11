import os

import pytest
from dotenv import load_dotenv

load_dotenv()  # Load .env where HF_TOKEN is defined
os.environ["TEMPLATE_PATH"] = "tests/assets/test_prompt_template.j2"
from processing import llm_few_shot
from processing.schema import ClassificationResult

# -----------------------------------------------------------------------------
# Setup: Fixture to generate a dummy template for isolated prompt tests
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def create_test_template():
    os.makedirs("tests/test_assets", exist_ok=True)
    with open("tests/test_assets/test_template.j2", "w") as f:
        f.write(
            "{{ instruction }}\nPOST:\n{{ post }}\nCRITERIA:\n{{ yes_criteria }}\n{{ no_criteria }}"
        )


# -----------------------------------------------------------------------------
# Prompt Rendering Test
# -----------------------------------------------------------------------------


def test_prompt_rendering_contains_criteria():
    prompt = llm_few_shot.build_prompt("This AI image generator feels biased.")
    assert "AI-generated images" in prompt
    assert "bias" in prompt.lower()
    assert "Output:" not in prompt  # prompt should not contain a pre-written answer


# -----------------------------------------------------------------------------
# Schema Validation Tests
# -----------------------------------------------------------------------------


def test_schema_validation_valid_row():
    row = ClassificationResult(
        id="123",
        subreddit="aiart",
        clean_text="Faces in AI images are mostly white males.",
        pred_label="bias",
        llm_reasoning="Output: 1 — This reflects bias in racial representation.",
    )
    assert row.pred_label == "bias"


def test_schema_validation_invalid_label():
    with pytest.raises(ValueError):
        ClassificationResult(
            id="123",
            subreddit="aiart",
            clean_text="Example",
            pred_label="yes",  # Invalid label
            llm_reasoning="Invalid label test",
        )


# -----------------------------------------------------------------------------
# Full classification test using the real model (optional/slow)
# -----------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN not set in environment")
def test_classify_post_real_model():
    tokenizer, model = llm_few_shot.load_model(llm_few_shot.MODEL_ID)

    post = "This AI image generator mostly creates white people. Feels biased."
    label_output = llm_few_shot.classify_post([post], tokenizer, model)

    label, reasoning = label_output[0]
    print(f"🔍 Label: {label}")
    print(f"🧠 Output: {reasoning}")

    assert label in ["Yes", "No", "Uncertain"]
    assert isinstance(reasoning, str)
    assert len(reasoning.strip()) > 0
