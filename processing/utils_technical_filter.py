### processing/utils_technical_filter.py

from pathlib import Path

# Prepare technical blacklist terms to be injected into subreddit_fetcher.py
TECHNICAL_BLACKLIST = [
    "how to install",
    "error",
    "model",
    "ckpt",
    "yaml",
    "diffusers",
    "setup",
    "api",
    "automatic1111",
    "torch",
    "cuda",
    "rtx",
    "comfyui",
    "training",
    "checkpoint",
    "tutorial",
    "paper",
    "gpu",
]

# General quesitons, error report etc
GENERAL_BLACKLIST = [
    "help",
    "issue",
    "installing",
    "question",
    "debug",
    "problem",
    "how do i",
    "crash",
    "fix",
    "code snippet",
    "traceback",
    "not working",
    "where can i",
    "looking for",
]

# Full blacklist = the combination of technical and general blacklist
FULL_BLACKLIST = TECHNICAL_BLACKLIST + GENERAL_BLACKLIST


def is_blacklist_post(text):
    """
    Detect whether a Reddit post is technical or off-topic
    based on predefined blacklist terms.
    """
    text = str(text).lower()
    return any(term in text for term in FULL_BLACKLIST)
