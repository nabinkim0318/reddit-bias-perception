import torch
import logging
import os

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
