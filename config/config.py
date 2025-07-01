import json
import os

# === COLAB DETECTION ===
try:
    IS_COLAB = "google.colab" in str(get_ipython())
except NameError:
    IS_COLAB = False

# === BASE PATH SETTING ===
BASE_DIR = "/content/drive/MyDrive/reddit_bias_data" if IS_COLAB else "data"
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# === INPUT FILES ===
RAW_REDDIT_DATA = os.path.join(RAW_DIR, "reddit_raw.json")
CLEANED_DATA = os.path.join(PROCESSED_DIR, "reddit_bias_data_clean.csv")
FILTERED_DATA = os.path.join(PROCESSED_DIR, "filtered_ai_bias.csv")
FINAL_ANALYSIS_INPUT = FILTERED_DATA


# === KEYWORDS ===
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


BIAS_KEYWORDS = load_json(os.path.join(CONFIG_DIR, "bias_keywords.json"))
AI_KEYWORDS = load_json(os.path.join(CONFIG_DIR, "ai_keywords.json"))


# === LLM Few-Shot OUTPUTS ===
OUTPUT_DIR = PROCESSED_DIR
CLASSIFIED_YES = os.path.join(OUTPUT_DIR, "classified_yes.csv")
CLASSIFIED_NO = os.path.join(OUTPUT_DIR, "classified_no.csv")
FEWSHOT_RESULT = os.path.join(OUTPUT_DIR, "fewshot_classification_results.csv")

# === TEMPLATE & MODEL ===
TEMPLATE_PATH = os.path.join(CONFIG_DIR, "fewshot_prompt_template.j2")
MODEL_ID = "NousResearch/Nous-Hermes-2-Mistral-7B"

# === TOPIC MODELING OUTPUT ===
TOPIC_OUTPUT = os.path.join(RESULTS_DIR, "bertopic_topic_info.csv")
TOPIC_ASSIGNMENT_PATH = os.path.join(RESULTS_DIR, "bertopic_post_topics.csv")
TOPIC_MODEL_PATH = os.path.join("models", "bertopic_model")

# === SENTIMENT ANALYSIS ===
SENTIMENT_OUTPUT = os.path.join(RESULTS_DIR, "sentiment_labeled.csv")

# === MODEL PARAMETERS ===
EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
EMOTION_TOP_K = 3
MAX_TEXT_LENGTH = 512

# === VADER ===
VADER_THRESHOLD = 0.05
VADER_PLOT_PATH = os.path.join(RESULTS_DIR, "vader_dist.png")

# === ETC ===
RANDOM_SEED = 42
SUBREDDIT_GROUPS_PATH = os.path.join(CONFIG_DIR, "subreddit_groups.json")
TOP_COMMENT_NUM = 10
BATCH_SIZE = 4
