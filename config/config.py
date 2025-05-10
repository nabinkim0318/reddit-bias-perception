import json
import os

# === COLAB ===
try:
    IS_COLAB = "google.colab" in str(get_ipython())
except NameError:
    IS_COLAB = False

if IS_COLAB:
    BASE_DIR = "/content/drive/MyDrive/reddit_bias_data"
else:
    BASE_DIR = "data"

# === PATHS ===
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# === INPUT FILES ===
RAW_REDDIT_DATA = os.getenv("RAW_REDDIT_DATA", os.path.join(RAW_DIR, "reddit_raw.json"))
CLEANED_DATA = os.getenv(
    "CLEANED_DATA", os.path.join(PROCESSED_DIR, "reddit_bias_data_clean.csv")
)
# CLASSIFIED_BIAS is the input for the keyword-based filtering
FILTERED_DATA = os.getenv(
    "FILTERED_DATA", os.path.join(PROCESSED_DIR, "filtered_ai_bias.csv")
)
FINAL_ANALYSIS_INPUT = os.getenv("FINAL_ANALYSIS_INPUT", FILTERED_DATA)


# === KEYWORDS ===
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


BIAS_KEYWORDS = load_json(os.path.join("config", "bias_keywords.json"))
AI_KEYWORDS = load_json(os.path.join("config", "ai_keywords.json"))

# === LLM Few-Shot OUTPUTS ===
OUTPUT_DIR = os.getenv("FEWSHOT_OUTPUT_DIR", PROCESSED_DIR)
CLASSIFIED_BIAS = os.path.join(OUTPUT_DIR, "classified_bias.csv")
CLASSIFIED_NONBIAS = os.path.join(OUTPUT_DIR, "classified_nonbias.csv")
BIAS_UNCERTAIN = os.path.join(OUTPUT_DIR, "bias_uncertain.csv")
FEWSHOT_RESULT = os.path.join(OUTPUT_DIR, "fewshot_classification_results.csv")

# === TEMPLATE & MODEL ===
TEMPLATE_PATH = "config/fewshot_prompt_template.j2"
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-2b-it")

# === TOPIC MODELING OUTPUT ===
TOPIC_OUTPUT = os.getenv(
    "TOPIC_OUTPUT", os.path.join(RESULTS_DIR, "bertopic_topic_info.csv")
)
TOPIC_ASSIGNMENT_PATH = os.getenv(
    "TOPIC_ASSIGNMENT_PATH", os.path.join(RESULTS_DIR, "bertopic_post_topics.csv")
)

# === SENTIMENT ANALYSIS ===
SENTIMENT_OUTPUT = os.getenv(
    "SENTIMENT_OUTPUT", os.path.join(RESULTS_DIR, "sentiment_labeled.csv")
)

# === MODEL PARAMETERS ===
EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
EMOTION_TOP_K = 3
MAX_TEXT_LENGTH = 512

# === VADER ===
VADER_THRESHOLD = 0.05
VADER_PLOT_PATH = os.path.join(RESULTS_DIR, "vader_dist.png")

# === ETC ===
RANDOM_SEED = 42
