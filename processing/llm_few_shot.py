### processing/llm_few_shot.py
"""
Few-shot classification using Gemma 2B model to determine whether a Reddit post discusses bias in AI-generated images.
"""

import logging
import os
import re
import traceback
from functools import lru_cache  # Lazy load template
from typing import Literal, cast

import pandas as pd
import torch
from dotenv import load_dotenv
from jinja2 import Template
from pydantic import ValidationError
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from config.config import (
    BATCH_SIZE,
    BIAS_UNCERTAIN,
    CLASSIFIED_BIAS,
    CLASSIFIED_NONBIAS,
    CLEANED_DATA,
    FEWSHOT_RESULT,
    MODEL_ID,
    OUTPUT_DIR,
    TEMPLATE_PATH,
)
from processing.schema import ClassificationResult
from utils.tokenize import batch_tokenize

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Classification instruction (system prompt)
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Your job is to classify whether the following Reddit post "
    "reflects a **concern, observation, or critique about bias** in AI-generated images."
)

YES_CRITERIA = (
    "- A complaint or observation about misrepresentation related to race, gender, body type, culture, disability, or religion\n"
    "- Frustration about the lack of diversity or overuse of stereotypical visuals in AI output\n"
    "- Emotional, social, or ethical concerns about fairness, exclusion, or how certain identities are depicted or erased\n"
    "- Even if subtle, the post reflects discomfort or surprise with how identity is visualized or how bias manifests in image generation"
)

NO_CRITERIA = (
    "- Focuses on technical details, creative tools, styles, rendering techniques, or model features\n"
    "- Shares AI-generated art, music, or fictional work without discussing fairness, identity, or social representation\n"
    "- Describes how the AI responded to prompts with no identity-related concern\n"
    "- Reflects on the creative process or philosophical/artistic value of AI output without reference to social bias or exclusion"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.info("🔍 Loading model...")


def load_model(model_id):
    """
    Load the Hugging Face tokenizer and model for few-shot classification.

    Args:
        model_id (str): The identifier for the pre-trained model.

    Returns:
        tuple: (tokenizer, model) ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    return tokenizer, model


@lru_cache
def get_template():
    """
    Load and cache the Jinja2 prompt template from file.

    Returns:
        Template: Jinja2 template object used to render few-shot prompts.
    """
    with open(TEMPLATE_PATH) as f:
        return Template(f.read())


def build_prompt(post_text):
    """
    Construct a classification prompt using the system instruction, criteria, and post text.

    Args:
        post_text (str): Reddit post's cleaned content.

    Returns:
        str: Fully rendered prompt ready for tokenization.
    """
    return get_template().render(
        instruction=SYSTEM_INSTRUCTION,
        yes_criteria=YES_CRITERIA,
        no_criteria=NO_CRITERIA,
        post=post_text.strip(),
        include_yes_criteria=True,
        include_no_criteria=True,
    )


def extract_label(decoded_output):
    match = re.search(
        r"Label:\s*(bias|non-bias|uncertain)", decoded_output, re.IGNORECASE
    )
    return match.group(1).lower() if match else "uncertain"


def classify_post(batch_texts, tokenizer, model):
    """
    Perform few-shot classification on a batch of Reddit post texts.

    Args:
        batch_texts (list of str): Cleaned Reddit post texts.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face causal language model.

    Returns:
        list of tuple: Each tuple is (label, model_output) where label is "Yes", "No", or "Uncertain".
    """
    prompts = [build_prompt(text) for text in batch_texts]
    inputs = batch_tokenize(prompts, tokenizer).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

    labels = []
    for decoded in decoded_outputs:
        label = extract_label(decoded)
        labels.append((label, decoded))

    return labels


def main():
    """
    Main execution for few-shot classification.

    Loads the model and cleaned Reddit data, applies few-shot classification to each post,
    and writes labeled results into three output files:
      - classified_bias.csv
      - classified_nonbias.csv
      - bias_uncertain.csv

    Output schema:
        - id (str)
        - subreddit (str)
        - clean_text (str)
        - pred_label (str): 'bias', 'non-bias', or 'uncertain'
        - llm_reasoning (str): LLM-generated reasoning for the classification
    """
    tokenizer, model = load_model(MODEL_ID)
    df = pd.read_csv(CLEANED_DATA)
    texts = df["clean_text"].fillna("").astype(str).tolist()
    subreddits = df["subreddit"] if "subreddit" in df.columns else ["unknown"] * len(df)
    ids = df["id"] if "id" in df.columns else [f"unknown_{i}" for i in range(len(df))]

    batch_size = BATCH_SIZE
    results = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        batch_subreddits = pd.Series(subreddits)[i : i + batch_size].reset_index(
            drop=True
        )
        batch_ids = pd.Series(ids)[i : i + batch_size].reset_index(drop=True)

        try:
            label_output_pairs = classify_post(batch_texts, tokenizer, model)
        except Exception:
            label_output_pairs = [("uncertain", traceback.format_exc())] * len(
                batch_texts
            )

        for j, (label, output) in enumerate(label_output_pairs):
            try:
                # pred_label validation and normalization
                normalized_label = (
                    label.lower() if isinstance(label, str) else "uncertain"
                )
                if normalized_label not in ["bias", "non-bias", "uncertain"]:
                    normalized_label = "uncertain"

                # Type checker for casting
                pred_label: Literal["bias", "non-bias", "uncertain"] = normalized_label  # type: ignore

                row = ClassificationResult(
                    id=batch_ids[j],
                    subreddit=batch_subreddits[j],
                    clean_text=batch_texts[j],
                    pred_label=pred_label,
                    llm_reasoning=output,
                )
                results.append(row.model_dump())
            except ValidationError as e:
                logging.error(f"Validation error at row {i + j}: {e}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(FEWSHOT_RESULT, index=False)

    result_df[result_df["pred_label"] == "bias"].to_csv(CLASSIFIED_BIAS, index=False)
    result_df[result_df["pred_label"] == "non-bias"].to_csv(
        CLASSIFIED_NONBIAS, index=False
    )
    result_df[result_df["pred_label"] == "uncertain"].to_csv(
        BIAS_UNCERTAIN, index=False
    )

    logging.info("Few-shot classification complete. Saved:")
    logging.info("- fewshot_classification_results.csv")
    logging.info("- classified_bias.csv")
    logging.info("- classified_nonbias.csv")
    logging.info("- bias_uncertain.csv")


# === SINGLE POST CLASSIFICATION ===
def classify_single_post(
    post_text, subreddit="unknown", post_id="unknown", tokenizer=None, model=None
):
    """
    Perform bias classification for a single Reddit post.

    Args:
        post_text (str): Text of the Reddit post to classify
        subreddit (str): Subreddit name (default: "unknown")
        post_id (str): Post ID (default: "unknown")
        tokenizer: Hugging Face tokenizer (auto-load if None)
        model: Hugging Face model (auto-load if None)

    Returns:
        dict: Dictionary
        {
            'id': str,
            'subreddit': str,
            'clean_text': str,
            'pred_label': str,
            'llm_reasoning': str
        }
    """
    # If model and tokenizer are not loaded, load them
    if tokenizer is None or model is None:
        logging.info("🔍 Loading model and tokenizer...")
        tokenizer, model = load_model(MODEL_ID)

    try:
        # Create prompt
        prompt = build_prompt(post_text)

        # Tokenize
        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode result
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract label
        label = extract_label(decoded_output)

        # Construct result
        result = {
            "id": post_id,
            "subreddit": subreddit,
            "clean_text": post_text,
            "pred_label": label,
            "llm_reasoning": decoded_output,
        }

        logging.info(f"✅ Classification complete: {label}")
        return result

    except Exception as e:
        logging.error(f"❌ Error during classification: {e}")
        return {
            "id": post_id,
            "subreddit": subreddit,
            "clean_text": post_text,
            "pred_label": "uncertain",
            "llm_reasoning": f"Error: {str(e)}",
        }


def example_single_classification():
    """
    A function to show an example of single post classification.
    """
    # Example text
    example_text = """
    I tried to generate images of 'doctors' and 'nurses' using AI,
    but all the doctors came out as white men and all the nurses as women.
    This feels really biased and doesn't represent the diversity we see in real healthcare.
    """

    print("🔍 Running example of single post classification...")
    print(f"📝 Input text: {example_text.strip()}")

    # Classification run
    result = classify_single_post(
        post_text=example_text, subreddit="artificial", post_id="example_001"
    )

    print("\nClassification result:")
    print(f"Label: {result['pred_label']}\n")
    print(f"Subreddit: {result['subreddit']}\n")
    print(f"ID: {result['id']}\n")
    print(f"LLM reasoning: {result['llm_reasoning'][:200]}...")

    return result


if __name__ == "__main__":
    # main()
    result = example_single_classification()
    print(result)
