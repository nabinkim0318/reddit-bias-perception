### processing/llm_few_shot_pipeline.py
"""
Few-shot classification: whether a post discusses visual-identity bias
in AI-generated images (discourse/perception, not objective system bias).

Scientific labels are yes/no only after a successful parse. Model, generation,
and parse failures are recorded as status + null pred_label, never as "no".
Automated labels are model predictions, not human-validated ground truth.
"""

import argparse
import json
import logging
import multiprocessing
import os
import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from dotenv import load_dotenv
from jinja2 import BaseLoader, Environment
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils import logging as hf_logging

from config.config import BATCH_SIZE, MODEL_ID, TEMPLATE_PATH
from processing.llm_annotation import (
    ERROR_GENERATION_EXCEPTION,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_OUTPUT_CARDINALITY_MISMATCH,
    AnnotationModelError,
    annotation_output_paths,
    classify_batch,
    empty_annotation_frame,
    ensure_result_cardinality,
    write_annotation_outputs,
)

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


GenerateFn = Callable[[List[str]], List[str]]


def _write_empty_outputs(paths: Dict[str, Path]) -> None:
    write_annotation_outputs(empty_annotation_frame(), paths)


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
                    load_kwargs.update(
                        dict(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.float16,
                        )
                    )
                else:
                    load_kwargs.update(
                        dict(torch_dtype=torch.float16 if use_gpu else torch.float32)
                    )

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
    """Generate one decoded string per input. Failures raise AnnotationModelError."""
    if tokenizer is None or model is None:
        raise AnnotationModelError(ERROR_MODEL_UNAVAILABLE)

    n = len(batch_texts)
    try:
        SUB_BATCH_SIZE = int(os.getenv("LLM_SUB_BATCH", "8"))
        sub_batch_size = max(1, min(SUB_BATCH_SIZE, n))

        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_list = [tok for tok in [tokenizer.eos_token_id, eot_id] if tok is not None]

        decoded_outputs: List[str] = []
        sub_batch_idx = 0

        for i in range(0, n, sub_batch_size):
            sub_texts = batch_texts[i : i + sub_batch_size]

            messages_batch = []
            for text in sub_texts:
                prompt = build_prompt(text)
                messages_batch.append(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are an AI ethics researcher analyzing Reddit "
                                "posts. Follow the task strictly."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                )

            prompt_texts = tokenizer.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                tokenize=False,
            )

            if isinstance(prompt_texts, str):
                prompt_texts = [prompt_texts]

            enc_max_len = min(2048, getattr(tokenizer, "model_max_length", 2048))
            enc = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=enc_max_len,
            )

            inputs = {k: v.to(model.device) for k, v in enc.items()}

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

            if len(outputs) != len(sub_texts):
                raise AnnotationModelError(ERROR_OUTPUT_CARDINALITY_MISMATCH)

            attn_mask = inputs["attention_mask"]
            in_lens = attn_mask.sum(dim=1).tolist()

            for j, out_ids in enumerate(outputs):
                gen_only = out_ids[in_lens[j] :]
                text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
                decoded_outputs.append(text)

            del outputs, inputs, enc, attn_mask, in_lens

            if torch.cuda.is_available():
                every = int(os.getenv("CUDA_EMPTY_CACHE_EVERY", "4"))
                if every > 0 and (sub_batch_idx % every == 0):
                    torch.cuda.empty_cache()

            sub_batch_idx += 1

        if len(decoded_outputs) != n:
            raise AnnotationModelError(ERROR_OUTPUT_CARDINALITY_MISMATCH)
        return decoded_outputs

    except AnnotationModelError:
        raise
    except Exception:
        logging.error("Model inference failed (%s)", ERROR_GENERATION_EXCEPTION)
        raise AnnotationModelError(ERROR_GENERATION_EXCEPTION) from None


def _unavailable_generate(_texts: List[str]) -> List[str]:
    raise AnnotationModelError(ERROR_MODEL_UNAVAILABLE)


def classify_post_wrapper(
    batch_input: Tuple[List[str], List[str], List[str]],
    generate_fn: Optional[GenerateFn] = None,
    tokenizer=None,
    model=None,
) -> List[Dict[str, Any]]:
    """Classify one batch. Generation can be injected for tests."""
    try:
        batch_texts, batch_ids, batch_subreddits = batch_input
    except Exception:
        logging.error("Invalid batch payload")
        return []

    if generate_fn is None:
        if tokenizer is None or model is None:
            tokenizer, model = get_model_and_tokenizer()
        if tokenizer is None or model is None:
            generate_fn = _unavailable_generate
        else:
            tok, mdl = tokenizer, model

            def _bound_generate(texts: List[str], _tok=tok, _mdl=mdl) -> List[str]:
                return generate_outputs(texts, _tok, _mdl)

            generate_fn = _bound_generate

    rows = classify_batch(batch_texts, batch_ids, batch_subreddits, generate_fn)
    return ensure_result_cardinality(rows, batch_ids, batch_texts, batch_subreddits)


def classify_single_post(
    post_text: str,
    subreddit: str = "unknown",
    post_id: str = "unknown",
    generate_fn: Optional[GenerateFn] = None,
    tokenizer=None,
    model=None,
) -> Dict[str, Any]:
    rows = classify_post_wrapper(
        ([post_text], [post_id], [subreddit]),
        generate_fn=generate_fn,
        tokenizer=tokenizer,
        model=model,
    )
    if rows:
        return rows[0]
    return classify_batch([post_text], [post_id], [subreddit], _unavailable_generate)[0]


def example_single_classification() -> Dict[str, Any]:
    """Show an example of single-post classification."""
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
    print(f"Status: {result.get('status')}")
    print(f"Label: {result['pred_label']}")
    print(f"Subreddit: {result['subreddit']}")
    print(f"ID: {result['id']}")
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "label": result["pred_label"],
                "error_type": result.get("error_type"),
            },
            indent=2,
        )
    )
    return result


def main(subreddit: str):
    """Run annotation and write yes / no / unclassified artifacts."""
    log_device_info()

    file_path = f"data/filtered/{subreddit}_keyword_filtered.csv"

    out_dir = Path("data/filtered")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = annotation_output_paths(out_dir, subreddit)

    if not os.path.exists(file_path):
        logging.error("Input file not found")
        _write_empty_outputs(paths)
        return

    logging.info("Loading data...")
    try:
        df = pd.read_csv(file_path)
        texts = df["clean_text"].fillna("").astype(str).tolist()
        subreddits = (
            df["subreddit"] if "subreddit" in df.columns else ["unknown"] * len(df)
        )
        ids = (
            df["id"] if "id" in df.columns else [f"unknown_{i}" for i in range(len(df))]
        )
        ids = [str(x) for x in ids]
        subreddits = [str(x) for x in subreddits]
    except Exception:
        logging.error("Failed to load annotation input")
        _write_empty_outputs(paths)
        return

    _ = BATCH_SIZE
    optimal_batch_size = int(os.getenv("LLM_BATCH", "16"))
    logging.info(f"Using batch size: {optimal_batch_size}")

    batch_input_list = []
    for i in range(0, len(texts), optimal_batch_size):
        batch_texts = texts[i : i + optimal_batch_size]
        batch_ids = list(ids[i : i + optimal_batch_size])
        batch_subreddits = list(subreddits[i : i + optimal_batch_size])
        batch_input_list.append((batch_texts, batch_ids, batch_subreddits))

    logging.info("Starting classification (single-process, batched generate)...")
    all_results: List[Dict[str, Any]] = []
    try:
        for batch_input in tqdm(
            batch_input_list,
            desc="Classifying",
            unit="batch",
            total=len(batch_input_list),
        ):
            all_results.extend(classify_post_wrapper(batch_input))
    except Exception:
        logging.error("Classification loop failed (%s)", ERROR_GENERATION_EXCEPTION)
        if not all_results:
            _write_empty_outputs(paths)
            return

    all_results = ensure_result_cardinality(all_results, ids, texts, subreddits)

    if not all_results:
        logging.error("No annotation rows were produced")
        _write_empty_outputs(paths)
        return

    result_df = pd.DataFrame(all_results)
    try:
        write_annotation_outputs(result_df, paths)
    except Exception:
        logging.error("Failed to write annotation artifacts")
        _write_empty_outputs(paths)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddit", type=str, required=True, help="Subreddit name")
    args = parser.parse_args()

    main(args.subreddit)
