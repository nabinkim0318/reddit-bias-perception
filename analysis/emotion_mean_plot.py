### analysis/emotion_mean_plot.py
"""
Plots mean emotion probabilities per bias category using barplots.
"""

import ast
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load data
SENTIMENT_CSV = "data/results/sentiment_labeled.csv"
TOPIC_MAP_JSON = "config/topic_to_bias.json"

print("📥 Loading data...")
df = pd.read_csv(SENTIMENT_CSV)
with open(TOPIC_MAP_JSON, "r") as f:
    topic_map = json.load(f)

df["bias_category"] = df["topic"].map(topic_map)
df = df.dropna(subset=["bias_category", "goemotions_probs"])

# Expand emotion probs
df_emotions = pd.DataFrame(df["goemotions_probs"].apply(ast.literal_eval).tolist())
df_emotions["bias_category"] = df["bias_category"].values

# Compute means
mean_df = df_emotions.groupby("bias_category").mean().T

# Plot
plt.figure(figsize=(12, 6))
sns.heatmap(mean_df, cmap="YlGnBu", annot=True, fmt=".2f")
plt.title("Mean Emotion Probabilities per Bias Category")
plt.xlabel("Bias Category")
plt.ylabel("Emotions")
plt.tight_layout()
plt.savefig("data/results/emotion_heatmap.png", dpi=300)
plt.show()
