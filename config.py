import os

# === PATHS ===
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# === INPUT FILES ===
RAW_REDDIT_DATA = os.getenv(
    "RAW_REDDIT_DATA", os.path.join(RAW_DIR, "reddit_bias_data.csv")
)
FILTERED_DATA = os.getenv(
    "FILTERED_DATA", os.path.join(PROCESSED_DIR, "filtered_ai_bias.csv")
)
FINAL_ANALYSIS_INPUT = os.getenv(
    "FINAL_ANALYSIS_INPUT", os.path.join(PROCESSED_DIR, "ai_bias_final.csv")
)

# === OUTPUT FILES ===
SENTIMENT_OUTPUT = os.getenv(
    "SENTIMENT_OUTPUT", os.path.join(RESULTS_DIR, "sentiment_labeled.csv")
)
TOPIC_OUTPUT = os.getenv(
    "TOPIC_OUTPUT", os.path.join(RESULTS_DIR, "bertopic_labeled.csv")
)

# === MODEL PARAMETERS ===
EMOTION_MODEL = "j-hartmann/emotion-english-roberta-base"
EMOTION_TOP_K = 3
MAX_TEXT_LENGTH = 512

# === VADER ===
VADER_THRESHOLD = 0.05

# === ETC ===
RANDOM_SEED = 42
