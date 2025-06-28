### processing/llm_few_shot.py
"""
Few-shot classification using Gemma 2B model to determine whether a Reddit post discusses bias in AI-generated images.
"""

import logging
import os
import re
import traceback
from functools import lru_cache  # Lazy load template
from multiprocessing import Pool
from typing import Literal, cast

import pandas as pd
import torch
from dotenv import load_dotenv
from jinja2 import Template
from pydantic import ValidationError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

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
    model = torch.compile(model)
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
    if not post_text:
        post_text = ""

    return get_template().render(
        instruction=SYSTEM_INSTRUCTION,
        yes_criteria=YES_CRITERIA,
        no_criteria=NO_CRITERIA,
        post=post_text.strip(),
        include_yes_criteria=True,
        include_no_criteria=True,
    )


def extract_label_and_reasoning(decoded_output):
    """
    Extract both label and reasoning from the model's output.
    Assumes the format:
    Label: bias|non-bias|uncertain
    Reasoning: [One sentence explanation]
    """
    label_match = re.search(
        r"Label:\s*(bias|non-bias|uncertain)", decoded_output, re.IGNORECASE
    )
    reasoning_match = re.search(
        r"Reasoning:\s*(.*)", decoded_output, re.IGNORECASE | re.DOTALL
    )

    label = label_match.group(1).lower() if label_match else "uncertain"
    reasoning = (
        reasoning_match.group(1).strip() if reasoning_match else decoded_output.strip()
    )

    # Optional: clean common token artifacts
    for token in ["<pad>", "<bos>", "<eos>"]:
        reasoning = reasoning.replace(token, "")
    reasoning = reasoning.strip()

    return label, reasoning


def load_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        logging.error(f"Model load error: {e}")
        return None, None


def generate_outputs(batch_texts, tokenizer, model):
    prompts = [build_prompt(text) for text in batch_texts]
    try:
        inputs = batch_tokenize(prompts, tokenizer).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.batch_decode(outputs, skip_special_tokens=False)
    except Exception as e:
        logging.error(f"Model inference error: {e}")
        return [f"Inference error: {e}"] * len(batch_texts)


def postprocess_outputs(decoded_outputs, batch_texts, batch_ids, batch_subreddits):
    rows = []
    for i, decoded in enumerate(decoded_outputs):
        label, reasoning = extract_label_and_reasoning(decoded)
        normalized_label = (
            label if label in ["bias", "non-bias", "uncertain"] else "uncertain"
        )
        try:
            pred_label: Literal["bias", "non-bias", "uncertain"] = normalized_label  # type: ignore
            row = ClassificationResult(
                id=batch_ids[i],
                subreddit=batch_subreddits[i],
                clean_text=batch_texts[i],
                pred_label=pred_label,
                llm_reasoning=reasoning.strip(),
            )
            rows.append(row.model_dump())
        except ValidationError as e:
            logging.error(f"Validation error: {e}")
            rows.append(
                {
                    "id": batch_ids[i],
                    "subreddit": batch_subreddits[i],
                    "clean_text": batch_texts[i],
                    "pred_label": "uncertain",
                    "llm_reasoning": f"Validation Error: {e}",
                }
            )
    return rows


def classify_post_wrapper(batch_input):
    batch_texts, batch_ids, batch_subreddits = batch_input
    tokenizer, model = load_model_and_tokenizer()
    if tokenizer is None or model is None:
        return [{"error": "Model load failed"}] * len(batch_texts)

    decoded_outputs = generate_outputs(batch_texts, tokenizer, model)
    return postprocess_outputs(
        decoded_outputs, batch_texts, batch_ids, batch_subreddits
    )


def main():
    logging.info("🔍 Loading data...")
    df = pd.read_csv(CLEANED_DATA)
    df = df.head(10)
    texts = df["clean_text"].fillna("").astype(str).tolist()
    subreddits = df["subreddit"] if "subreddit" in df.columns else ["unknown"] * len(df)
    ids = df["id"] if "id" in df.columns else [f"unknown_{i}" for i in range(len(df))]

    batch_size = BATCH_SIZE
    batch_input_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = pd.Series(ids[i : i + batch_size]).reset_index(drop=True)
        batch_subreddits = pd.Series(subreddits[i : i + batch_size]).reset_index(
            drop=True
        )
        batch_input_list.append((batch_texts, batch_ids, batch_subreddits))

    logging.info("🚀 Starting multiprocessing classification...")
    all_results = []
    with Pool(processes=4) as pool:  # You can tune this number
        for result_batch in tqdm(
            pool.imap_unordered(classify_post_wrapper, batch_input_list),
            total=len(batch_input_list),
        ):
            all_results.extend(result_batch)

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(FEWSHOT_RESULT, index=False)
    result_df[result_df["pred_label"] == "bias"].to_csv(CLASSIFIED_BIAS, index=False)
    result_df[result_df["pred_label"] == "non-bias"].to_csv(
        CLASSIFIED_NONBIAS, index=False
    )
    result_df[result_df["pred_label"] == "uncertain"].to_csv(
        BIAS_UNCERTAIN, index=False
    )

    logging.info("✅ Few-shot classification complete. Files saved:")
    logging.info(f"→ {FEWSHOT_RESULT}")
    logging.info(f"→ {CLASSIFIED_BIAS}")
    logging.info(f"→ {CLASSIFIED_NONBIAS}")
    logging.info(f"→ {BIAS_UNCERTAIN}")


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
                temperature=0.0,
                top_k=1,
                repetition_penalty=1.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode result
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract label
        label = extract_label_and_reasoning(decoded_output)

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

    print("Running example of single post classification...\n")
    print(f"Input text: {example_text.strip()}")

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
    main()
    # result = example_single_classification()
    # print(result)
