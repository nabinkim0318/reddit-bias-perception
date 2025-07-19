### processing/llm_few_shot.py
"""
Few-shot classification using Gemma 2B model to determine whether a Reddit post discusses bias in AI-generated images.
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Literal, cast, List, Dict, Any, Optional, Tuple
import threading
import time
from multiprocessing import Pool, cpu_count
from contextlib import contextmanager

import pandas as pd
import torch
from dotenv import load_dotenv
from jinja2 import BaseLoader, Environment
from pydantic import ValidationError
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from config.config import (
    BATCH_SIZE,
    CLASSIFIED_NO,
    CLASSIFIED_YES,
    KEYWORDS_FILTERED_DATA,
    MODEL_ID,
    TEMPLATE_PATH,
)
from processing.schema import ClassificationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Global model cache
_model_cache = {}
_model_lock = threading.Lock()

# === Utilities ===
@lru_cache(maxsize=1)
def get_template():
    """
    Load and cache the Jinja2 prompt template from file.
    """
    with open(TEMPLATE_PATH, "r") as f:
        template_text = f.read()
        env = Environment(
            loader=BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return env.from_string(template_text)


def build_prompt(post_text: str) -> str:
    """
    Render the classification prompt using Jinja2 template.
    """
    rendered = get_template().render(post_text=(post_text or "").strip())
    return rendered


def clean_output(decoded_output: str) -> str:
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


def parse_label(cleaned: str) -> Optional[str]:
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
    return None


def parse_reasoning(cleaned: str) -> str:
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


def extract_label_and_reasoning(decoded_output: str) -> Tuple[str, str]:
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
            bias_keywords = ["bias", "image", "representation", "diversity", "gender", "race"]
            if "yes" in text_lower and any(k in text_lower for k in bias_keywords):
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
@contextmanager
def get_model_and_tokenizer():
    """
    Context manager for model and tokenizer with caching.
    
    Yields:
        Tuple[Optional[AutoTokenizer], Optional[AutoModelForCausalLM]]: 
        Tokenizer and model, or (None, None) if loading failed
    """
    global _model_cache
    
    with _model_lock:
        if 'model' not in _model_cache or 'tokenizer' not in _model_cache:
            logging.info("🔍 Loading model and tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    token=HF_TOKEN,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True,  # Memory optimization
                )
                model.eval()
                
                _model_cache['tokenizer'] = tokenizer
                _model_cache['model'] = model
                logging.info("✅ Model loaded successfully")
                
            except Exception as e:
                logging.error(f"❌ Model load error: {e}")
                _model_cache['tokenizer'] = None
                _model_cache['model'] = None
    
    try:
        yield _model_cache['tokenizer'], _model_cache['model']
    except Exception as e:
        logging.error(f"❌ Error during model inference: {e}")
        yield None, None


def generate_outputs(batch_texts: List[str], tokenizer, model) -> List[str]:
    """
    Generate outputs for a batch of texts with improved memory management.
    """
    if tokenizer is None or model is None:
        return [f"Model not available"] * len(batch_texts)
    
    decoded_outputs = []
    
    try:
        # Process in smaller sub-batches to manage memory
        sub_batch_size = min(4, len(batch_texts))  # Smaller sub-batches
        
        for i in range(0, len(batch_texts), sub_batch_size):
            sub_batch = batch_texts[i:i + sub_batch_size]
            sub_outputs = []
            
            for text in sub_batch:
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

                attention_mask = (input_ids != tokenizer.pad_token_id).long()

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=400,  # Reduced for efficiency
                        do_sample=False,
                        eos_token_id=[
                            tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                        ],
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # Decode and strip any special tokens
                response = outputs[0][input_ids.shape[-1]:]
                decoded = tokenizer.decode(response, skip_special_tokens=True).strip()
                sub_outputs.append(decoded)
                
                # Clear GPU memory
                del outputs, input_ids, attention_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            decoded_outputs.extend(sub_outputs)
            
        return decoded_outputs

    except Exception as e:
        logging.error(f"❌ Model inference error: {e}")
        return [f"Inference error: {e}"] * len(batch_texts)


def postprocess_outputs(
    decoded_outputs: List[str], 
    batch_texts: List[str], 
    batch_ids: List[str], 
    batch_subreddits: List[str]
) -> List[Dict[str, Any]]:
    """
    Postprocess model outputs with improved error handling.
    """
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
            rows.append({
                "id": batch_ids[i],
                "subreddit": batch_subreddits[i],
                "clean_text": batch_texts[i],
                "pred_label": label,
                "llm_reasoning": f"Validation Error: {e}",
                "raw_output": decoded,
            })
    return rows


def classify_post_wrapper(batch_input: Tuple[List[str], List[str], List[str]]) -> List[Dict[str, Any]]:
    """
    Wrapper function for multiprocessing with improved error handling.
    """
    try:
        batch_texts, batch_ids, batch_subreddits = batch_input
        
        with get_model_and_tokenizer() as (tokenizer, model):
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
            return postprocess_outputs(decoded_outputs, batch_texts, batch_ids, batch_subreddits)

    except Exception as e:
        logging.exception("❌ classify_post_wrapper failed with exception")
        return [{
            "id": f"unknown_{i}",
            "subreddit": "unknown",
            "clean_text": "",
            "pred_label": "no",
            "llm_reasoning": f"Batch failed: {str(e)}",
            "raw_output": "",
        } for i in range(len(batch_input[0]))]

# === SINGLE POST CLASSIFICATION ===
def classify_single_post(
    post_text: str, 
    subreddit: str = "unknown", 
    post_id: str = "unknown"
) -> Dict[str, Any]:
    """
    Perform bias classification for a single Reddit post with optimized model loading.
    """
    with get_model_and_tokenizer() as (tokenizer, model):
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

            # Tokenize with memory optimization
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Perform inference
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    repetition_penalty=1.1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=[
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                    ],
                )

            # Decode result
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            decoded_output = decoded_output.replace(prompt, "").strip()

            # Extract label
            label, reasoning = extract_label_and_reasoning(decoded_output)

            # Clear memory
            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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


def example_single_classification() -> Dict[str, Any]:
    """
    A function to show an example of single post classification.
    """
    example_text = """
    I tried to generate images of 'doctors' and 'nurses' using AI,
    but all the doctors came out as white men and all the nurses as women.
    This feels really biased and doesn't represent the diversity we see in real healthcare.
    """

    print("Running example of single post classification...\n")
    print(f"Input text: {example_text.strip()}")

    result = classify_single_post(
        post_text=example_text, subreddit="artificial", post_id="example_001"
    )

    print("\nClassification result:")
    print(f"Label: {result['pred_label']}")
    print(f"Subreddit: {result['subreddit']}")
    print(f"ID: {result['id']}")
    print(json.dumps(
        {"label": result["pred_label"], "reasoning": result["llm_reasoning"]},
        indent=2,
    ))

    return result


# === Pipeline Entry Point ===
def main():
    """
    Main pipeline with improved memory management and error handling.
    """
    file_path = "data/filtered/aiwars_full_filtered_posts_cleaned_posts.csv"
    
    # Check if file exists
    if not os.path.exists(file_path):
        logging.error(f"❌ File not found: {file_path}")
        return

    logging.info("🔍 Loading data...")
    try:
        df = pd.read_csv(file_path)
        texts = df["clean_text"].fillna("").astype(str).tolist()
        subreddits = df["subreddit"] if "subreddit" in df.columns else ["unknown"] * len(df)
        ids = df["id"] if "id" in df.columns else [f"unknown_{i}" for i in range(len(df))]
    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        return

    # Optimize batch size based on available memory and CPU cores
    available_cores = os.cpu_count() or 1
    optimal_batch_size = min(BATCH_SIZE, max(2, len(texts) // (available_cores * 2)))
    logging.info(f"Using batch size: {optimal_batch_size}")

    # Prepare batches
    batch_input_list = []
    for i in range(0, len(texts), optimal_batch_size):
        batch_texts = texts[i : i + optimal_batch_size]
        batch_ids = ids[i : i + optimal_batch_size]
        batch_subreddits = subreddits[i : i + optimal_batch_size]
        batch_input_list.append((batch_texts, batch_ids, batch_subreddits))

    logging.info("🚀 Starting multiprocessing classification...")
    all_results = []

    # Use fewer processes to avoid memory issues
    num_processes = min(available_cores, 2)  # Reduced from 4 to 2
    logging.info(f"Using {num_processes} processes for classification")

    try:
        with Pool(processes=num_processes) as pool:
            results = pool.map(classify_post_wrapper, tqdm(
                batch_input_list,
                desc="Classifying posts",
                total=len(batch_input_list)
            ))
            for batch_result in results:
                all_results.extend(batch_result)
    except Exception as e:
        logging.error(f"❌ Error during multiprocessing: {e}")
        return

    if not all_results:
        logging.error("❌ No results were generated")
        return

    result_df = pd.DataFrame(all_results)

    # Print classification statistics
    logging.info(f"📊 Classification Results:")
    logging.info(f"Total posts processed: {len(result_df)}")
    
    if len(result_df) > 0:
        label_counts = result_df["pred_label"].value_counts()
        logging.info(f"Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(result_df)) * 100
            logging.info(f"  {label}: {count} ({percentage:.1f}%)")

    # Save results
    try:
        result_df[result_df["pred_label"] == "yes"].to_csv(CLASSIFIED_YES, index=False)
        result_df[result_df["pred_label"] == "no"].to_csv(CLASSIFIED_NO, index=False)
        logging.info(f"✅ Results saved to:")
        logging.info(f"  → {CLASSIFIED_YES}")
        logging.info(f"  → {CLASSIFIED_NO}")
    except Exception as e:
        logging.error(f"❌ Error saving results: {e}")


if __name__ == "__main__":
    main()
