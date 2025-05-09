# vectorizer_config.py
"""
Reusable CountVectorizer configuration for topic modeling.
Includes custom and default stopwords for AI bias Reddit dataset.
"""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# Custom stopwords relevant to AI art/model discussions
CUSTOM_STOPWORDS = set(
    [
        "use",
        "know",
        "thing",
        "like",
        "just",
        "really",
        "even",
        "think",
        "get",
        "used",
        "dont",
        "make",
        "want",
        "need",
        "see",
        "im",
        "people",
        "post",
        "prompt",
        "prompts",
        "image",
        "images",
        "mj",
        "midjourney",
        "dalle",
        "openai",
        "transformer",
        "resnets",
        "layer",
        "weight",
        "param",
        "train",
        "loss",
        "epoch",
        "tokenizer",
        "attention",
        "model",
        "error",
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
        "ai",
        "models",
        "sd",
        "sdxl",
        "movie",
        "movies",
        "human",
        "new",
        "good",
        "time",
        "feel",
        "using",
        "seen",
        "character",
    ]
)

# Merge with sklearn's English stopwords
STOPWORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS)

# Define vectorizer with consistent config
vectorizer_model = CountVectorizer(stop_words=STOPWORDS, min_df=2)
