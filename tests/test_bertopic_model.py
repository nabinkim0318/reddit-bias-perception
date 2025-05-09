### analysis/sentiment_analysis.py
"""
Sentiment analysis on AI bias Reddit posts using:
1. GoEmotions RoBERTa (PyTorch, multi-label)
2. VADER (rule-based)

Also prepares output for MANOVA-based emotion distribution analysis.
"""

import ast
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Setup paths and model
INPUT_PATH = os.getenv("SENTIMENT_INPUT", "data/processed/ai_bias_final.csv")
OUTPUT_PATH = os.getenv("SENTIMENT_OUTPUT", "data/results/sentiment_labeled.csv")
MODEL_ID = "SamLowe/roberta-base-go_emotions"
BIAS_MAPPING_PATH = "config/topic_to_bias.json"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔍 Loading GoEmotions model (PyTorch)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
id2label = model.config.id2label

# Load VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Load topic → bias mapping (for MANOVA)
if os.path.exists(BIAS_MAPPING_PATH):
    with open(BIAS_MAPPING_PATH) as f:
        TOPIC_TO_BIAS = json.load(f)
else:
    TOPIC_TO_BIAS = {}


def batch_tokenize(texts, tokenizer, max_length=512):
    return tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )


def run_goemotions(texts, batch_size=32):
    model.eval()
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = batch_tokenize(batch, tokenizer)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()

        for i_probs in probs:
            all_probs = {id2label[i]: float(score) for i, score in enumerate(i_probs)}
            results.append(all_probs)
    print("DEBUG result:", results)
    return results


def run_vader(text):
    return vader_analyzer.polarity_scores(text)


def plot_goemotion_distribution(df):
    print("📊 Plotting GoEmotions distribution...")
    df_plot = pd.DataFrame(df["goemotions_probs"].tolist())
    melted = df_plot.melt(var_name="emotion", value_name="prob")
    melted = melted[melted["prob"] > 0.4]
    sns.countplot(y=melted["emotion"])
    plt.title("Top Emotions (prob > 0.4)")
    plt.tight_layout()
    plt.show()


def plot_vader_distribution(df):
    print("📊 Plotting VADER compound score distribution...")
    compound_scores = df["vader"].apply(
        lambda x: (
            ast.literal_eval(x)["compound"] if isinstance(x, str) else x["compound"]
        )
    )
    sns.histplot(compound_scores, bins=30, kde=True)
    plt.title("VADER Compound Score Distribution")
    plt.xlabel("Compound Score")
    plt.tight_layout()
    plt.show()


def analyze_comments(comments):
    comment_emotions = []
    for c in comments:
        if not isinstance(c, str) or not c.strip():
            comment_emotions.append({"goemotions": {}, "vader": {}})
            continue
        ge = run_goemotions([c])[0]
        vd = run_vader(c)
        comment_emotions.append({"goemotions": ge, "vader": vd})
    return comment_emotions


def main():
    df = pd.read_csv(INPUT_PATH)
    df["text"] = df["text"].fillna("")

    texts = [t for t in df["text"].tolist() if t.strip()]
    print(f"🧠 Running GoEmotions on {len(texts)} texts...")
    ge_outputs = run_goemotions(texts)
    vader_outputs = [run_vader(t) for t in texts]

    df = df.loc[df["text"].str.strip() != ""].copy()
    df["goemotions_probs"] = ge_outputs
    df["vader"] = vader_outputs

    if "topic" in df.columns and TOPIC_TO_BIAS:
        df["bias_category"] = df["topic"].map(
            lambda x: TOPIC_TO_BIAS.get(str(x), "Other")
        )

    if "comments" in df.columns:
        print("🧠 Analyzing comments...")
        df["comments"] = df["comments"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df["comment_sentiment"] = df["comments"].apply(analyze_comments)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Sentiment analysis complete. Output saved to: {OUTPUT_PATH}")

    plot_goemotion_distribution(df)
    plot_vader_distribution(df)


if __name__ == "__main__":
    main()
