import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# === Configuration ===
# Specify model and file paths
MODEL_ID = "SamLowe/roberta-base-go_emotions"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_PATH = "sample_sentiment_input.csv"
OUTPUT_PATH = "sentiment_labeled_output.csv"

# === Load model and tokenizer ===
print(f"🔍 Loading model on {DEVICE} ...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
vader = SentimentIntensityAnalyzer()
id2label = model.config.id2label  # Map class indices to emotion labels


# === Run GoEmotions classification ===
def run_goemotions(texts, threshold=0.4):
    model.eval()
    results = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()

        for row in probs:
            emotions = [
                (id2label[i], float(score))
                for i, score in enumerate(row)
                if score > threshold
            ]
            sorted_emotions = sorted(emotions, key=lambda x: x[1], reverse=True)
            results.append(sorted_emotions)

    return results


# === Run VADER sentiment analysis ===
def run_vader(texts):
    return [vader.polarity_scores(t) for t in texts]


# === Sample input data ===
sample_texts = [
    "I feel a strange mix of admiration and sadness.",
    "Why is the AI so biased?",
    "I'm really impressed with this result.",
    "This is disappointing, honestly.",
    "The image gave me chills... in a good way!",
]
df = pd.DataFrame({"text": sample_texts})
df.to_csv(INPUT_PATH, index=False)

# === Perform sentiment analysis ===
all_goemotions, all_vader = [], []
for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch = df["text"].iloc[i : i + BATCH_SIZE].tolist()
    all_goemotions.extend(run_goemotions(batch))
    all_vader.extend(run_vader(batch))

# === Save results to CSV ===
df["goemotions_top"] = all_goemotions
df["vader"] = all_vader
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Done! Output saved to {OUTPUT_PATH}")
