### processing/llm_few_shot.py
"""
Few-shot classification using Gemma 2B model to determine whether a Reddit post discusses bias in AI-generated images.
"""

import json
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
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from config.config import (
    BATCH_SIZE,
    CLASSIFIED_NO,
    CLASSIFIED_YES,
    CLEANED_DATA,
    MODEL_ID,
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
    with open(TEMPLATE_PATH, "r") as f:
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

    return get_template().render(post=(post_text or "").strip())


def extract_label_and_reasoning(decoded_output):
    """
    Extract both label and reasoning from the model's output using regex patterns.
    """
    try:
        # Clean the output
        cleaned = re.sub(r"```json|```", "", decoded_output).strip()

        # Try to extract label first
        label_patterns = [
            r'"label"\s*:\s*"([^"]+)"',  # JSON format
            r'label\s*:\s*"([^"]+)"',  # Without quotes
            r"label\s*:\s*([a-zA-Z-]+)",  # Without quotes, alphanumeric
        ]

        label = None
        for pattern in label_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                label = match.group(1).strip().lower()
                break

        # If no label found, try keyword-based detection
        if not label:
            text_lower = cleaned.lower()
            if "no" in text_lower and "yes" not in text_lower:
                return "no", f"Fallback parsing: {decoded_output.strip()[:100]}"
            elif re.search(
                r"\b(yes|bias|biased|fairness|representation|diversity|stereotype|identity)\b",
                text_lower,
            ):
                return "yes", f"Fallback parsing: {decoded_output.strip()[:100]}"
            else:
                # 최종 fallback은 보수적으로 no로 처리
                return (
                    "no",
                    f"Fallback (no strong signal): {decoded_output.strip()[:100]}",
                )

        # Validate label
        if label not in {"yes", "no"}:
            label = "no"

        # Try to extract reasoning
        reasoning_patterns = [
            r'"reasoning"\s*:\s*"([^"]+)"',  # JSON format
            r'reasoning\s*:\s*"([^"]+)"',  # Without quotes
            r"reasoning\s*:\s*([^,\n]+)",  # Without quotes, until comma or newline
        ]

        reasoning = "No reasoning provided"
        for pattern in reasoning_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                # Remove trailing punctuation
                reasoning = re.sub(r"[.,;]+$", "", reasoning)
                break

        # If no reasoning found, try to extract meaningful text
        if reasoning == "No reasoning provided":
            # Look for sentences that might contain reasoning
            sentences = re.split(r"[.!?]", cleaned)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(
                    word in sentence.lower()
                    for word in [
                        "because",
                        "since",
                        "as",
                        "due",
                        "reason",
                        "concern",
                        "issue",
                        "problem",
                    ]
                ):
                    reasoning = sentence
                    break

        return label, reasoning

    except Exception as e:
        # Log the first few failed cases for debugging
        if not hasattr(extract_label_and_reasoning, "_logged_failures"):
            extract_label_and_reasoning._logged_failures = 0

        if extract_label_and_reasoning._logged_failures < 3:
            logging.warning(
                f"⚠️ Failed to parse output (case {extract_label_and_reasoning._logged_failures + 1}): {e}"
            )
            logging.warning(f"Raw output: {decoded_output[:500]}...")
            extract_label_and_reasoning._logged_failures += 1
        elif extract_label_and_reasoning._logged_failures == 3:
            logging.warning("⚠️ Suppressing further parsing failure logs...")
            extract_label_and_reasoning._logged_failures += 1

        # Fallback: try to extract any meaningful information
        text_lower = decoded_output.lower()
        if "yes" in text_lower and "no" not in text_lower:
            return "yes", f"Fallback parsing: {decoded_output.strip()[:100]}"
        else:
            return "no", f"Fallback parsing: {decoded_output.strip()[:100]}"


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
                max_new_tokens=200,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Remove special tokens and clean up the output
        decoded_outputs = []
        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            # Remove the original prompt from the output
            for prompt in prompts:
                if prompt in decoded:
                    decoded = decoded.replace(prompt, "").strip()
            decoded_outputs.append(decoded)
        return decoded_outputs
    except Exception as e:
        logging.error(f"Model inference error: {e}")
        return [f"Inference error: {e}"] * len(batch_texts)


def postprocess_outputs(decoded_outputs, batch_texts, batch_ids, batch_subreddits):
    rows = []
    for i, decoded in enumerate(decoded_outputs):
        label, reasoning = extract_label_and_reasoning(decoded)
        try:
            pred_label: Literal["yes", "no"] = label  # type: ignore
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
                    "pred_label": "No",
                    "llm_reasoning": f"Validation Error: {e}",
                }
            )
    return rows


def classify_post_wrapper(batch_input):
    batch_texts, batch_ids, batch_subreddits = batch_input
    tokenizer, model = load_model_and_tokenizer()
    if tokenizer is None or model is None:
        return [
            {
                "id": batch_ids[i],
                "subreddit": batch_subreddits[i],
                "clean_text": batch_texts[i],
                "pred_label": "no",
                "llm_reasoning": "Model load failed",
            }
            for i in range(len(batch_texts))
        ]

    decoded_outputs = generate_outputs(batch_texts, tokenizer, model)
    return postprocess_outputs(
        decoded_outputs, batch_texts, batch_ids, batch_subreddits
    )


def main():
    logging.info("🔍 Loading data...")
    df = pd.read_csv(CLEANED_DATA)
    texts = df["clean_text"].fillna("").astype(str).tolist()
    subreddits = df["subreddit"] if "subreddit" in df.columns else ["unknown"] * len(df)
    ids = df["id"] if "id" in df.columns else [f"unknown_{i}" for i in range(len(df))]

    # Adjust batch size based on data size and available memory
    batch_size = min(BATCH_SIZE, max(1, len(texts) // 4))
    logging.info(f"Using batch size: {batch_size}")

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

    # Adjust number of processes based on system resources
    num_processes = min(4, os.cpu_count() or 1)
    logging.info(f"Using {num_processes} processes for classification")

    with Pool(processes=num_processes) as pool:
        for result_batch in tqdm(
            pool.imap_unordered(classify_post_wrapper, batch_input_list),
            total=len(batch_input_list),
            desc="Classifying posts",
        ):
            all_results.extend(result_batch)

    result_df = pd.DataFrame(all_results)

    # Print classification statistics
    logging.info(f"📊 Classification Results:")
    logging.info(f"Total posts processed: {len(result_df)}")
    if len(result_df) > 0:
        label_counts = result_df["pred_label"].value_counts()
        logging.info(f"Label distribution:")
        for label, count in label_counts.items():
            logging.info(f"  {label}: {count} ({count/len(result_df)*100:.1f}%)")

    result_df[result_df["pred_label"] == "yes"].to_csv(CLASSIFIED_YES, index=False)
    result_df[result_df["pred_label"] == "no"].to_csv(CLASSIFIED_NO, index=False)
    logging.info(f"→ {CLASSIFIED_YES}")
    logging.info(f"→ {CLASSIFIED_NO}")


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
                max_new_tokens=200,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode result
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the output
        decoded_output = decoded_output.replace(prompt, "").strip()

        # Extract label
        label, reasoning = extract_label_and_reasoning(decoded_output)

        # Construct result
        result = {
            "id": post_id,
            "subreddit": subreddit,
            "clean_text": post_text,
            "pred_label": label,
            "llm_reasoning": reasoning,
        }

        logging.info(f"✅ Classification complete: {label}")
        return result

    except Exception as e:
        logging.error(f"❌ Error during classification: {e}")
        return {
            "id": post_id,
            "subreddit": subreddit,
            "clean_text": post_text,
            "pred_label": "no",
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
    print(
        json.dumps(
            {"label": result["pred_label"], "reasoning": result["llm_reasoning"]},
            indent=2,
        )
    )

    return result


if __name__ == "__main__":
    main()
    # result = example_single_classification()
    # print(result)
