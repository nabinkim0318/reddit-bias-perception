### processing/llm_few_shot.pyMore actions
"""
Few-shot classification using Gemma 2B model to determine whether a Reddit post discusses bias in AI-generated images.
"""

import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache  # Lazy load template
from multiprocessing import Pool
from typing import List, Literal, cast

import pandas as pd
import torch
from dotenv import load_dotenv
from jinja2 import BaseLoader, Environment, Template
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


# === Utilities ===
@lru_cache
def get_template():
    """
    Load and cache the Jinja2 prompt template from file.

    Returns:
        Template: Jinja2 template object used to render few-shot prompts.
    """
    with open(TEMPLATE_PATH, "r") as f:
        template_text = f.read()
        env = Environment(
            loader=BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return env.from_string(template_text)


def build_prompt(post_text):
    """
    Render the classification prompt using Jinja2 template.
    """
    rendered = get_template().render(post_text=(post_text or "").strip())
    return rendered


def clean_output(decoded_output):
    """Clean the raw model output by removing markdown, system instructions, etc."""
    cleaned = re.sub(r"```json|```", "", decoded_output).strip()
    cleaned = re.sub(
        r"### SYSTEM_INSTRUCTION.*?###",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"system_instruction.*?###",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def parse_label(cleaned):
    """Extract label ('yes'/'no') from cleaned output."""
    label_patterns = [
        r'"label"\s*:\s*"([^"]+)"',
        r'label\s*:\s*"([^"]+)"',
        r"label\s*:\s*([a-zA-Z-]+)",
    ]
    for pattern in label_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            label = match.group(1).strip().lower()
            # Normalize
            if label in {"y", "yes", "true", "1"}:
                return "yes"
            elif label in {"n", "no", "false", "0"}:
                return "no"
            elif label in {"yes", "no"}:
                return label
    return None  # No label found


def parse_reasoning(cleaned):
    """Extract reasoning from cleaned output."""
    reasoning_patterns = [
        r'"reasoning"\s*:\s*"([^"]+)"',
        r'reasoning\s*:\s*"([^"]+)"',
        r"reasoning\s*:\s*([^,\n]+)",
    ]
    for pattern in reasoning_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
            reasoning = re.sub(r"[.,;]+$", "", reasoning)
            return reasoning
    # Try weak fallback: text after label
    label_pos = re.search(r"label\s*:\s*(yes|no)", cleaned, re.IGNORECASE)
    if label_pos:
        after_label = cleaned[label_pos.end() :].strip()
        candidate = re.split(r"\n+", after_label)[0]
        if len(candidate) > 10:
            return candidate
    return "No reasoning provided"


def extract_label_and_reasoning(decoded_output):
    """Main function: clean, extract label + reasoning with fallbacks."""
    try:
        cleaned = clean_output(decoded_output)

        # Check for repeated prompt
        if (
            decoded_output.strip().startswith("You are an AI ethics researcher")
            or "Now classify the following post" in decoded_output
            or "{{ post }}" in decoded_output
            or 'Post: "{{' in decoded_output
            or decoded_output.count("Post:") > 1
        ):
            return (
                "no",
                "⚠️ Model failed — repeated system prompt without generating output",
            )

        label = parse_label(cleaned)
        reasoning = parse_reasoning(cleaned)

        if not label:
            # Fallback keyword match
            text_lower = cleaned.lower()
            if "yes" in text_lower and any(
                k in text_lower
                for k in [
                    "bias",
                    "image",
                    "representation",
                    "diversity",
                    "gender",
                    "race",
                ]
            ):
                return (
                    "yes",
                    f"Fallback parsing (weak keyword match): {decoded_output.strip()[:100]}",
                )
            else:
                return (
                    "no",
                    f"Fallback (no strong signal): {decoded_output.strip()[:100]}",
                )

        return label, reasoning

    except Exception as e:
        # Failure logging (up to 3 cases)
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

        # Final fallback
        text_lower = decoded_output.lower()
        if "yes" in text_lower and "no" not in text_lower:
            return "yes", f"Fallback parsing: {decoded_output.strip()[:100]}"
        else:
            return "no", f"Fallback parsing: {decoded_output.strip()[:100]}"


# === MODEL Loading & Inference ===
def load_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        logging.error(f"Model load error: {e}")
        return None, None


def generate_outputs(batch_texts, tokenizer, model):
    decoded_outputs = []

    try:
        for text in batch_texts:
            prompt = build_prompt(text)
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI ethics researcher analyzing Reddit posts. Follow the task strictly.",
                },
                {"role": "user", "content": prompt},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=600,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=[
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                    ],
                )

            # Decode and strip any special tokens
            response = outputs[0][input_ids.shape[-1] :]  # prompt 이후만 추출
            decoded = tokenizer.decode(response, skip_special_tokens=True).strip()

            decoded_outputs.append(decoded)
        return decoded_outputs

    except Exception as e:
        logging.error(f"Model inference error: {e}")
        return [f"Inference error: {e}"] * len(batch_texts)


# === Postprocessing ===
def postprocess_outputs(decoded_outputs, batch_texts, batch_ids, batch_subreddits):
    rows = []
    for i, decoded in enumerate(decoded_outputs):
        label, reasoning = extract_label_and_reasoning(decoded)
        try:
            # Ensure label is properly typed
            pred_label: Literal["yes", "no"] = cast(Literal["yes", "no"], label)
            row = ClassificationResult(
                id=batch_ids[i],
                subreddit=batch_subreddits[i],
                clean_text=batch_texts[i],
                pred_label=pred_label,
                llm_reasoning=reasoning.strip(),
                raw_output=decoded,
            )
            rows.append(row.model_dump())
        except ValidationError as e:
            logging.error(f"Validation error for ID {batch_ids[i]}: {e}")
            logging.error(f"Raw output: {decoded}")
            rows.append(
                {
                    "id": batch_ids[i],
                    "subreddit": batch_subreddits[i],
                    "clean_text": batch_texts[i],
                    "pred_label": label,
                    "llm_reasoning": f"Validation Error: {e}",
                    "raw_output": decoded,
                }
            )
    return rows


def classify_post_wrapper(batch_input, tokenizer, model):
    batch_texts, batch_ids, batch_subreddits = batch_input
    if tokenizer is None or model is None:
        return [
            {
                "id": batch_ids[i],
                "subreddit": batch_subreddits[i],
                "clean_text": batch_texts[i],
                "pred_label": "no",
                "llm_reasoning": "Model load failed",
                "raw_output": "",
            }
            for i in range(len(batch_texts))
        ]

    decoded_outputs = generate_outputs(batch_texts, tokenizer, model)
    return postprocess_outputs(
        decoded_outputs, batch_texts, batch_ids, batch_subreddits
    )


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
        tokenizer, model = load_model_and_tokenizer()

    # Check if model loading failed
    if tokenizer is None or model is None:
        logging.error("❌ Failed to load model and tokenizer")
        return {
            "id": post_id,
            "subreddit": subreddit,
            "clean_text": post_text,
            "pred_label": "no",
            "llm_reasoning": "Model load failed",
        }

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
        print(f"Decoded output: {decoded_output}")

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


# === Pipeline Entry Point ===
def main():
    tokenizer, model = load_model_and_tokenizer()

    if tokenizer is None or model is None:
        logging.error("❌ Failed to load model and tokenizer. Exiting.")
        return

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

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(classify_post_wrapper, batch_input, tokenizer, model)
            for batch_input in batch_input_list
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Classifying posts"
        ):
            all_results.extend(future.result())

    result_df = pd.DataFrame(all_results)

    if result_df.empty:
        logging.error(
            "❌ No results were generated — check prompt, model, or parser issues."
        )
        return

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


if __name__ == "__main__":
    main()
    # result = example_single_classification()
    # print(result)
