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


def get_dynamic_sub_batch_size(max_target_memory_mb=3000):
    """
    Estimate sub-batch size dynamically based on available GPU memory.
    :param max_target_memory_mb: approximate memory you want to allocate per sub-batch (MB)
    :return: sub_batch_size (int)
    """
    if not torch.cuda.is_available():
        return 2  # Safe default for CPU or Colab Basic

    torch.cuda.empty_cache()
    gpu_id = torch.cuda.current_device()
    mem_info = torch.cuda.mem_get_info(gpu_id)
    free_mem_mb = mem_info[0] // (1024**2)  # Convert bytes to MB

    # Estimate sub_batch_size
    est_batch = max(1, free_mem_mb // max_target_memory_mb)
    return min(est_batch, 16)  # Cap max for safety
