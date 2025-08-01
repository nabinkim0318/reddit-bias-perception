### processing/llm_few_shot_pipeline.py
"""
Few-shot classification using Gemma 2B model to determine whether a Reddit post discusses bias in AI-generated images.
"""

import logging
import os
import sys

print(sys.path)
try:
    # check whether Colab is running
    IS_COLAB = "google.colab" in str(get_ipython())
except NameError:
    IS_COLAB = False

if IS_COLAB:
    # add project root to sys.path
    sys.path.append("/content/reddit-bias-perception")
else:
    # add relative path to project root
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import threading
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

from config.config import MAX_TOKENS, MODEL_ID, TEMPERATURE, TOP_K, TOP_P
from processing.llm_post_processing import postprocess_outputs
from processing.llm_propmts import build_prompt
from utils.llm_utils import (
    get_dynamic_sub_batch_size,
    get_paths_for_subreddit,
    log_device_info,
)

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Global model cache
_model_cache = {}
_model_lock = threading.Lock()


# === MODEL Loading & Inference ===
def get_vllm_engine():
    """
    Loading model and tokenizer with caching.
    """
    global _model_cache

    with _model_lock:
        if "model" not in _model_cache:
            logging.info("🔍 Loading model with vLLM...")
            try:
                model = LLM(
                    model=MODEL_ID,
                    tokenizer=MODEL_ID,
                    trust_remote_code=True,
                    dtype="bfloat16",
                )
                _model_cache["model"] = model
                logging.info("✅ vLLM model loaded successfully")

            except Exception as e:
                logging.error(f"❌ vLLM model load error: {e}")
                _model_cache["model"] = None

    return _model_cache["model"]


def load_data(subreddit: str) -> Optional[pd.DataFrame]:
    file_path = os.path.join(
        "data", "filtered", f"{subreddit}_full_filtered_posts_cleaned_posts.csv"
    )

    if not os.path.exists(file_path):
        logging.error(f"❌ File not found: {file_path}")
        return None

    logging.info("🔍 Loading data...")
    try:
        df = pd.read_csv(file_path)
        df["clean_text"].fillna("").astype(str).tolist()
        df["subreddit"].tolist() if "subreddit" in df.columns else ["unknown"] * len(df)
        (
            df["id"].tolist()
            if "id" in df.columns
            else [f"unknown_{i}" for i in range(len(df))]
        )

    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        return None

    return df


def get_sampling_params():
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stop=["</s>", "<|eot_id|>"],
        top_p=TOP_P,
        top_k=TOP_K,
    )
    return sampling_params


def generate_outputs(batch_texts: List[str], llm: LLM, batch_size: int) -> List[str]:
    """
    Generate outputs for a batch of texts using vLLM.
    Assumes llm is a vllm.LLM object.

    Args:
        batch_texts (List[str]): Raw input texts (e.g., Reddit post text)
        llm (vllm.LLM): Pre-loaded vLLM model

    Returns:
        List[str]: Decoded model outputs
    """
    if llm is None:
        return [f"Model not available"] * len(batch_texts)

    decoded_outputs = []
    sampling_params = get_sampling_params()

    for i in range(0, len(batch_texts), batch_size):
        sub_batch = batch_texts[i : i + batch_size]
        prompts = []

        for text in sub_batch:
            prompt = build_prompt(text)
            full_prompt = (
                "You are an AI ethics researcher analyzing Reddit posts. Follow the task strictly.\n\n"
                f"User: {prompt}"
            )
            prompts.append(full_prompt)

        try:
            outputs = llm.generate(prompts, sampling_params)
            for o in outputs:
                response = o.outputs[0].text.strip()
                decoded_outputs.append(response)

        except Exception as e:
            logging.error(f"❌ vLLM inference error: {e}")
            return ["ERROR: " + str(e)] * len(batch_texts)

    return decoded_outputs


def classify_post_wrapper(
    llm: LLM,
    batch_texts: List[str],
    batch_ids: List[str],
    batch_subreddits: List[str],
    sub_batch_size: int,
) -> List[Dict[str, Any]]:
    """
    Wrapper function for vLLM-based batch classification with robust error handling.
    """
    try:
        decoded_outputs = generate_outputs(batch_texts, llm, sub_batch_size)
        return postprocess_outputs(
            decoded_outputs, batch_texts, batch_ids, batch_subreddits
        )
    except Exception as e:
        logging.exception("❌ Error during decoding or postprocessing")
        return [
            {
                "id": str(batch_ids[i]) if i < len(batch_ids) else f"unknown_{i}",
                "subreddit": (
                    str(batch_subreddits[i]) if i < len(batch_subreddits) else "unknown"
                ),
                "clean_text": batch_texts[i] if i < len(batch_texts) else "",
                "pred_label": "no",
                "llm_reasoning": f"Postprocessing failed: {str(e)}",
                "raw_output": "",
            }
            for i in range(len(batch_texts))
        ]


def classify_batches(df: pd.DataFrame, llm: LLM) -> pd.DataFrame:
    """
    Classify all posts in the DataFrame using vLLM in batches.
    """
    batch_size = get_dynamic_sub_batch_size()
    logging.info(f"🧠 Using dynamic sub_batch_size={batch_size} based on GPU memory")

    all_results = []

    for i in tqdm(range(0, len(df), batch_size), desc="Classifying posts"):
        batch = df.iloc[i : i + batch_size]
        batch_texts = batch["clean_text"].astype(str).tolist()
        batch_ids = batch["id"].astype(str).tolist()
        batch_subreddits = batch["subreddit"].astype(str).tolist()

        try:
            results = classify_post_wrapper(
                llm, batch_texts, batch_ids, batch_subreddits, batch_size
            )
            all_results.extend(results)
        except Exception as e:
            logging.error(f"❌ Error in batch {i // batch_size}: {e}")
            all_results.extend(
                [
                    {
                        "id": batch_ids[j],
                        "subreddit": batch_subreddits[j],
                        "clean_text": batch_texts[j],
                        "pred_label": "no",
                        "llm_reasoning": "Error during classification",
                        "raw_output": str(e),
                    }
                    for j in range(len(batch_texts))
                ]
            )

    if not all_results:
        logging.error("❌ No results were generated")
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def summarizing_filtered_results(result_df: pd.DataFrame, subreddit: str):
    logging.info(f"📊 Classification Results:")
    logging.info(f"Total posts processed: {len(result_df)}")

    if len(result_df) > 0:
        label_counts = result_df["pred_label"].value_counts()
        logging.info("Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(result_df)) * 100
            logging.info(f"  {label}: {count} ({percentage:.1f}%)")


def saving_filtered_results(result_df: pd.DataFrame, subreddit: str):
    try:
        paths = get_paths_for_subreddit(subreddit)
        result_df[result_df["pred_label"] == "yes"].to_csv(
            paths["CLASSIFIED_YES"], index=False
        )
        result_df[result_df["pred_label"] == "no"].to_csv(
            paths["CLASSIFIED_NO"], index=False
        )
        logging.info("✅ Results saved to:")
        logging.info(f"  → {paths['CLASSIFIED_YES']}")
        logging.info(f"  → {paths['CLASSIFIED_NO']}")
    except Exception as e:
        logging.error(f"❌ Error saving results: {e}")


# === Pipeline Entry Point ===
def main(subreddit: str):
    """
    Main pipeline using vLLM for sequential classification (no multiprocessing).
    """
    log_device_info()

    # Load data
    df = load_data(subreddit)
    if df is None:
        logging.error("❌ Failed to load data")
        return
    logging.info(f"🚀 Loaded {len(df)} posts from {subreddit}")

    # Load vLLM model
    llm = get_vllm_engine()
    if llm is None:
        logging.error("❌ vLLM engine not initialized.")
        return
    logging.info("✅ Loaded vLLM model")

    # Classify posts and save results
    results_df = classify_batches(df, llm)
    summarizing_filtered_results(results_df, subreddit)
    saving_filtered_results(results_df, subreddit)


if __name__ == "__main__":
    subreddit = "aiwars"
    main(subreddit)
