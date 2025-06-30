### processing/llm_few_shot.py
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
    Extract label and reasoning by parsing LLM output.
    Returns:
        label (str): 'yes' or 'no' if found, otherwise 'skip'
        reasoning (str): Reasoning string if found, else empty
        raw_output (str): Full decoded output
    """
    try:
        cleaned = re.sub(r"```json|```", "", decoded_output).strip()

        # Extract label
        label_match = re.search(r"(?i)label\s*[:\-]?\s*(yes|no)", cleaned)
        reasoning_match = re.search(r"(?i)reasoning\s*[:\-]?\s*(.+)", cleaned)

        label = label_match.group(1).strip().lower() if label_match else None
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        # Validate label
        if label not in {"yes", "no"}:
            logging.warning(
                f"⚠️ Label missing or malformed in output: {decoded_output[:150]}"
            )
            return "skip", "", decoded_output

        # Validate reasoning
        if not reasoning or reasoning.lower() in {
            "your reasoning in 1-2 sentences",
            "none",
            "n/a",
            "",
        }:
            logging.warning(
                f"⚠️ Reasoning missing or invalid in output: {decoded_output[:150]}"
            )
            reasoning = ""

        return label, reasoning, decoded_output

    except Exception as e:
        logging.warning(
            f"⚠️ Exception during parsing: {e} | Output: {decoded_output[:100]}"
        )
        return "skip", "", decoded_output


def load_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            offload_buffers=True,
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
                max_new_tokens=600,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Remove special tokens and clean up the output
        decoded_outputs = []
        input_lengths = [len(input_ids) for input_ids in inputs["input_ids"]]

        for i, output in enumerate(outputs):
            # Slice off the prompt tokens from the generated output
            generated_tokens = output[input_lengths[i] :]
            decoded = tokenizer.decode(
                generated_tokens, skip_special_tokens=False
            ).strip()
            decoded_outputs.append(decoded)
        return decoded_outputs
    except Exception as e:
        logging.error(f"Model inference error: {e}")
        return [f"Inference error: {e}"] * len(batch_texts)


def split_multiple_responses(decoded_output: str) -> List[str]:
    """
    Split model output into separate prompt responses based on repeated 'Label:' prefix.
    Useful when LLM returns multiple completions in a single generation.
    """
    blocks = re.split(r"(?i)(?=label\s*:)", decoded_output)
    return [b.strip() for b in blocks if b.strip().startswith("Label")]


def postprocess_outputs(decoded_outputs, batch_texts, batch_ids, batch_subreddits):
    rows = []
    for i, decoded in enumerate(decoded_outputs):
        responses = split_multiple_responses(decoded)
        for response in responses:
            label, reasoning, raw_output = extract_label_and_reasoning(response)
            try:
                pred_label: Literal["yes", "no"] = label  # type: ignore
                row = ClassificationResult(
                    id=batch_ids[i],
                    subreddit=batch_subreddits[i],
                    clean_text=batch_texts[i],
                    pred_label=pred_label,
                    llm_reasoning=reasoning.strip(),
                    raw_output=raw_output,
                )
                rows.append(row.model_dump())
            except ValidationError as e:
                logging.error(f"Validation error: {e}")
                rows.append(
                    {
                        "id": batch_ids[i],
                        "subreddit": batch_subreddits[i],
                        "clean_text": batch_texts[i],
                        "pred_label": "no",
                        "llm_reasoning": f"Validation Error: {e}",
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

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(classify_post_wrapper, batch_input, tokenizer, model)
            for batch_input in batch_input_list
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Classifying posts"
        ):
            all_results.extend(future.result())

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
        tokenizer, model = load_model_and_tokenizer()

    if tokenizer is None or model is None:
        logging.error("❌ Failed to load model and tokenizer")
        return {
            "id": post_id,
            "subreddit": subreddit,
            "clean_text": post_text,
            "pred_label": "no",
            "llm_reasoning": "Model load failed",
            "raw_output": "",
        }

    try:
        # Create prompt
        prompt = build_prompt(post_text)

        inputs = batch_tokenize([prompt], tokenizer).to(model.device)

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
        label, reasoning, raw_output = extract_label_and_reasoning(decoded_output)

        # Construct result
        result = {
            "id": post_id,
            "subreddit": subreddit,
            "clean_text": post_text,
            "pred_label": label,
            "llm_reasoning": reasoning,
            "raw_output": raw_output,
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
            "raw_output": "",
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
