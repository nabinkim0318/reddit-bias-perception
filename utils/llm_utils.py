import logging
import os

import torch

from config.config import OUTPUT_DIR


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
        logging.debug(
            f"📊 GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
        )


def get_paths_for_subreddit(subreddit: str) -> dict:
    return {
        "CLASSIFIED_YES": os.path.join(OUTPUT_DIR, f"{subreddit}_filtered_ai_bias.csv"),
        "CLASSIFIED_NO": os.path.join(
            OUTPUT_DIR, f"{subreddit}_filtered_ai_non_bias.csv"
        ),
        "FEWSHOT_RESULT": os.path.join(
            OUTPUT_DIR, f"{subreddit}_llm_classification_results.csv"
        ),
    }


def get_dynamic_sub_batch_size(max_target_memory_mb=1000, max_cap=64):
    """
    Dynamically estimate batch size based on free GPU memory.
    """
    if not torch.cuda.is_available():
        return 2

    torch.cuda.empty_cache()
    gpu_id = torch.cuda.current_device()
    free_mem_mb = torch.cuda.mem_get_info(gpu_id)[0] // (1024**2)

    est_batch = max(1, free_mem_mb // max_target_memory_mb)
    capped_batch = min(est_batch, max_cap)

    logging.info(
        f"🧮 Free GPU memory: {free_mem_mb}MB | Est. batch: {est_batch} | Using: {capped_batch}"
    )
    return capped_batch
