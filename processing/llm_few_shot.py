### processing/llm_few_shot.py
"""
Few-shot classification using Gemma 2B model to determine whether a Reddit post discusses bias in AI-generated images.
"""

import json
import logging
import multiprocessing
import os
import threading
import warnings
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from config.config import BATCH_SIZE, MODEL_ID
from processing.llm_post_processing import (
    parse_label_and_reasoning,
    postprocess_outputs,
)
from processing.llm_propmts import build_prompt
from utils.llm_utils import get_paths_for_subreddit, log_device_info, log_gpu_memory

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Global model cache
_model_cache = {}
_model_lock = threading.Lock()

multiprocessing.set_start_method("spawn", force=True)


# === MODEL Loading & Inference ===
def get_model_and_tokenizer():
    """
    Loading model and tokenizer with caching.

    Returns:
        Tuple[Optional[AutoTokenizer], Optional[AutoModelForCausalLM]]:
        Tokenizer and model, or (None, None) if loading failed
    """
    global _model_cache

    with _model_lock:
        if "model" not in _model_cache or "tokenizer" not in _model_cache:
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

                _model_cache["tokenizer"] = tokenizer
                _model_cache["model"] = model
                logging.info("✅ Model loaded successfully")

            except Exception as e:
                logging.error(f"❌ Model load error: {e}")
                _model_cache["tokenizer"] = None
                _model_cache["model"] = None

    return _model_cache["tokenizer"], _model_cache["model"]


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
            sub_batch = batch_texts[i : i + sub_batch_size]
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
                response = outputs[0][input_ids.shape[-1] :]
                decoded = tokenizer.decode(response, skip_special_tokens=True).strip()
                sub_outputs.append(decoded)

                # Clear GPU memory
                del outputs, input_ids, attention_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Log memory usage every 100 batches (reduced frequency)
                    if len(decoded_outputs) % 100 == 0:
                        log_gpu_memory()

            decoded_outputs.extend(sub_outputs)

        return decoded_outputs

    except Exception as e:
        logging.error(f"❌ Model inference error: {e}")
        return [f"Inference error: {e}"] * len(batch_texts)


def classify_post_wrapper(
    batch_input: Tuple[List[str], List[str], List[str]]
) -> List[Dict[str, Any]]:
    """
    Wrapper function for multiprocessing with improved error handling.
    """
    try:
        batch_texts, batch_ids, batch_subreddits = batch_input

        tokenizer, model = get_model_and_tokenizer()
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

        try:
            decoded_outputs = generate_outputs(batch_texts, tokenizer, model)
            return postprocess_outputs(
                decoded_outputs, batch_texts, batch_ids, batch_subreddits
            )
        except Exception as e:
            logging.exception(
                "❌ Error during decoding or postprocessing inside classify_post_wrapper"
            )
            return [
                {
                    "id": str(batch_ids[i]) if i < len(batch_ids) else f"unknown_{i}",
                    "subreddit": (
                        str(batch_subreddits[i])
                        if i < len(batch_subreddits)
                        else "unknown"
                    ),
                    "clean_text": batch_texts[i] if i < len(batch_texts) else "",
                    "pred_label": "no",
                    "llm_reasoning": f"Postprocessing failed: {str(e)}",
                    "raw_output": "",
                }
                for i in range(len(batch_texts))
            ]

    except Exception as e:
        logging.exception("❌ classify_post_wrapper failed with exception")
        return [
            {
                "id": f"unknown_{i}",
                "subreddit": "unknown",
                "clean_text": "",
                "pred_label": "no",
                "llm_reasoning": f"Batch failed: {str(e)}",
                "raw_output": "",
            }
            for i in range(len(batch_input[0]))
        ]


# === SINGLE POST CLASSIFICATION ===
def classify_single_post(
    post_text: str, subreddit: str = "unknown", post_id: str = "unknown"
) -> Dict[str, Any]:
    """
    Perform bias classification for a single Reddit post with optimized model loading.
    """
    tokenizer, model = get_model_and_tokenizer()
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
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
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
        label, reasoning = parse_label_and_reasoning(decoded_output)

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
    print(
        json.dumps(
            {"label": result["pred_label"], "reasoning": result["llm_reasoning"]},
            indent=2,
        )
    )

    return result


# === Pipeline Entry Point ===
def main(subreddit: str):
    """
    Main pipeline with improved memory management and error handling.
    """
    # Log device information
    log_device_info()

    file_path = "data/filtered/aiwars_full_filtered_posts_cleaned_posts.csv"

    # Check if file exists
    if not os.path.exists(file_path):
        logging.error(f"❌ File not found: {file_path}")
        return

    logging.info("🔍 Loading data...")
    try:
        df = pd.read_csv(file_path)
        texts = df["clean_text"].fillna("").astype(str).tolist()
        subreddits = (
            df["subreddit"] if "subreddit" in df.columns else ["unknown"] * len(df)
        )
        ids = (
            df["id"] if "id" in df.columns else [f"unknown_{i}" for i in range(len(df))]
        )
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
        batch_ids = list(ids[i : i + optimal_batch_size])
        batch_subreddits = list(subreddits[i : i + optimal_batch_size])
        batch_input_list.append((batch_texts, batch_ids, batch_subreddits))

    logging.info("🚀 Starting multiprocessing classification...")
    all_results = []

    # Use fewer processes to avoid memory issues
    num_processes = min(available_cores, 2)  # Reduced from 4 to 2
    logging.info(f"Using {num_processes} processes for classification")

    try:
        with Pool(processes=num_processes) as pool:
            results = pool.map(
                classify_post_wrapper,
                tqdm(
                    batch_input_list,
                    desc="Classifying posts",
                    total=len(batch_input_list),
                ),
            )
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
        paths = get_paths_for_subreddit(subreddit)
        result_df[result_df["pred_label"] == "yes"].to_csv(
            paths["CLASSIFIED_YES"], index=False
        )
        result_df[result_df["pred_label"] == "no"].to_csv(
            paths["CLASSIFIED_NO"], index=False
        )
        logging.info(f"✅ Results saved to:")
        logging.info(f"  → {paths['CLASSIFIED_YES']}")
        logging.info(f"  → {paths['CLASSIFIED_NO']}")
    except Exception as e:
        logging.error(f"❌ Error saving results: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    subreddit = "aiwars"
    main(subreddit)
