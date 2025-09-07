### processing/llm_few_shot_pipeline.py
"""
Few-shot classification using Gemma 2B model to determine whether a Reddit post discusses bias in AI-generated images.
"""

import argparse
import json
import logging
import multiprocessing
import os
import re
import threading
import warnings
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import pandas as pd
import torch
from dotenv import load_dotenv
from jinja2 import BaseLoader, Environment
from pydantic import ValidationError
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils import logging as hf_logging

from config.config import BATCH_SIZE, MODEL_ID, TEMPLATE_PATH
from processing.schema import ClassificationResult

warnings.filterwarnings("ignore", message="The following generation flags")
hf_logging.set_verbosity_error()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    logging.warning("HF_TOKEN not set; private models may fail to load.")


EMPTY_COLS = [
    "id",
    "subreddit",
    "clean_text",
    "pred_label",
    "llm_reasoning",
    "raw_output",
]


def _write_empty_outputs(yes_path: Path, no_path: Path):
    empty = pd.DataFrame(columns=EMPTY_COLS)
    empty.to_csv(yes_path, index=False)
    empty.to_csv(no_path, index=False)
    logging.info(f"✅ Empty outputs written to:")
    logging.info(f"  → {yes_path}")
    logging.info(f"  → {no_path}")


# Global model cache
_model_cache = {}
_model_lock = threading.Lock()


def log_device_info():
    """Log current device information."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_total = (
            torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        )
        logging.info(f"🚀 Using GPU: {device_name} ({memory_total:.1f}GB)")
        logging.info(f"   Available GPUs: {device_count}")
    else:
        logging.info("💻 Using CPU for inference")


def log_gpu_memory():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        logging.info(
            f"📊 GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
        )


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
            bias_keywords = [
                "bias",
                "image",
                "representation",
                "diversity",
                "gender",
                "race",
            ]
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
def get_model_and_tokenizer():
    global _model_cache
    with _model_lock:
        if "model" not in _model_cache or "tokenizer" not in _model_cache:
            logging.info("🔍 Loading model and tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_ID,
                    token=HF_TOKEN,
                    use_fast=True,
                    trust_remote_code=True,
                )
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                use_gpu = torch.cuda.is_available()
                USE_4BIT = os.getenv("USE_4BIT", "1") not in {"0", "false", "False"}
                
                load_kwargs = dict(
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                    
                if use_gpu and USE_4BIT:
                    load_kwargs.update(dict(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    ))
                else:
                    load_kwargs.update(dict(
                        torch_dtype=torch.float16 if use_gpu else torch.float32
                    ))

                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    token=HF_TOKEN,
                    trust_remote_code=True,
                    **load_kwargs,
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
    if tokenizer is None or model is None:
        return ["Model not available"] * len(batch_texts)

    try:
        SUB_BATCH_SIZE = int(os.getenv("LLM_SUB_BATCH", "8"))
        sub_batch_size = max(1, min(SUB_BATCH_SIZE, len(batch_texts)))

        # Safe EOS
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_list = [tok for tok in [tokenizer.eos_token_id, eot_id] if tok is not None]

        decoded_outputs: List[str] = []
        sub_batch_idx = 0  # ← Sub-batch counter

        for i in range(0, len(batch_texts), sub_batch_size):
            sub_texts = batch_texts[i:i+sub_batch_size]

            # (1) batched messages
            messages_batch = []
            for text in sub_texts:
                prompt = build_prompt(text)
                messages_batch.append([
                    {"role": "system", "content": "You are an AI ethics researcher analyzing Reddit posts. Follow the task strictly."},
                    {"role": "user", "content": prompt},
                ])

            # (2) apply template & tokenize (batched)
            prompt_texts = tokenizer.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                tokenize=False,          # ← take strings
            )
            
            if isinstance(prompt_texts, str):
                prompt_texts = [prompt_texts]

            # Now encode with normal tokenizer (dict return)
            enc_max_len = min(2048, getattr(tokenizer, "model_max_length", 2048))
            enc = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=enc_max_len,
            )

            # move to device
            inputs = {k: v.to(model.device) for k, v in enc.items()}

            # (3) generate once per sub-batch
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    eos_token_id=eos_list if eos_list else None,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=False,
                )

            # (4) cut off the prompt per-sample using attention_mask lengths
            attn_mask = inputs["attention_mask"]
            in_lens = attn_mask.sum(dim=1).tolist()  # [B]

            for j, out_ids in enumerate(outputs):   # ← enumerate index is j
                gen_only = out_ids[in_lens[j]:]
                text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
                decoded_outputs.append(text or "[EMPTY_OUTPUT]")

            # cleanup
            del outputs, inputs, enc, attn_mask, in_lens

            # Sometimes clear cache (to avoid fragmentation)
            if torch.cuda.is_available():
                every = int(os.getenv("CUDA_EMPTY_CACHE_EVERY", "4"))
                if every > 0 and (sub_batch_idx % every == 0):
                    torch.cuda.empty_cache()
                    # Option: torch.cuda.synchronize()

            sub_batch_idx += 1  # ← Sub-batch counter increase
        return decoded_outputs

    except Exception as e:
        logging.error(f"❌ Model inference error: {e}")
        return [f"Inference error: {e}"] * len(batch_texts)



def postprocess_outputs(
    decoded_outputs: List[str],
    batch_texts: List[str],
    batch_ids: List[str],
    batch_subreddits: List[str],
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
def classify_single_post(post_text: str, subreddit: str = "unknown", post_id: str = "unknown") -> Dict[str, Any]:
    tokenizer, model = get_model_and_tokenizer()
    if tokenizer is None or model is None:
        logging.error("❌ Failed to load model and tokenizer")
        return {
            "id": post_id, "subreddit": subreddit, "clean_text": post_text,
            "pred_label": "no", "llm_reasoning": "Model load failed",
        }

    try:
        prompt = build_prompt(post_text)
        messages = [
            {"role": "system", "content": "You are an AI ethics researcher analyzing Reddit posts. Follow the task strictly."},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_list = [tokenizer.eos_token_id] + ([eot_id] if eot_id is not None else [])

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=200,
                do_sample=True, temperature=0.6, top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=eos_list,
            )

        gen_only = outputs[0][input_ids.shape[-1]:]
        decoded_output = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
        if not decoded_output:
            decoded_output = "[EMPTY_OUTPUT]"

        label, reasoning = extract_label_and_reasoning(decoded_output)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = {
            "id": post_id, "subreddit": subreddit, "clean_text": post_text,
            "pred_label": label, "llm_reasoning": reasoning, "raw_output": decoded_output,
        }
        logging.info(f"✅ Classification complete: {label}")
        return result

    except Exception as e:
        logging.error(f"❌ Error during classification: {e}")
        return {
            "id": post_id, "subreddit": subreddit, "clean_text": post_text,
            "pred_label": "no", "llm_reasoning": f"Error: {str(e)}",
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

    file_path = f"data/filtered/{subreddit}_keyword_filtered.csv"

    out_dir = Path("data/filtered")
    out_dir.mkdir(parents=True, exist_ok=True)
    yes_path = out_dir / f"{subreddit}_filtered_ai_bias.csv"
    no_path = out_dir / f"{subreddit}_filtered_ai_non_bias.csv"

    # Check if file exists
    if not os.path.exists(file_path):
        logging.error(f"❌ File not found: {file_path}")
        _write_empty_outputs(yes_path, no_path)
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
        _write_empty_outputs(yes_path, no_path)
        return

    # Optimize batch size based on available memory and CPU cores
    num_processes = 1
    optimal_batch_size = int(os.getenv("LLM_BATCH", "16"))
    logging.info(f"Using batch size: {optimal_batch_size}")

    # Prepare batches
    batch_input_list = []
    for i in range(0, len(texts), optimal_batch_size):
        batch_texts = texts[i : i + optimal_batch_size]
        batch_ids = list(ids[i : i + optimal_batch_size])
        batch_subreddits = list(subreddits[i : i + optimal_batch_size])
        batch_input_list.append((batch_texts, batch_ids, batch_subreddits))

    logging.info("🚀 Starting classification (single-process, batched generate)...")
    all_results = []
    try:
        for batch_input in tqdm(batch_input_list, desc="Classifying", unit="batch", total=len(batch_input_list)):
            all_results.extend(classify_post_wrapper(batch_input))
    except Exception as e:
        logging.error(f"❌ Error during classification loop: {e}")
        _write_empty_outputs(yes_path, no_path)
        return

    if not all_results:
        logging.error("❌ No results were generated")
        _write_empty_outputs(yes_path, no_path)
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
        result_df[result_df["pred_label"] == "yes"].to_csv(yes_path, index=False)
        result_df[result_df["pred_label"] == "no"].to_csv(no_path, index=False)
        logging.info(f"✅ Results saved to:")
        logging.info(f"  → {yes_path}")
        logging.info(f"  → {no_path}")
    except Exception as e:
        logging.error(f"❌ Error saving results: {e}")
        _write_empty_outputs(yes_path, no_path)
        return


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddit", type=str, required=True, help="Subreddit name")
    args = parser.parse_args()

    main(args.subreddit)
