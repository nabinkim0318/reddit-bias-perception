### analysis/emotion_clustering.py
"""
Clusters bias categories based on mean emotion vectors using hierarchical clustering.
"""

import ast
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

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

# Compute group means
group_means = df_emotions.groupby("bias_category").mean()

# Clustering
linkage_matrix = linkage(group_means, method="ward")

# Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=group_means.index.tolist(), leaf_rotation=45)
plt.title("Bias Category Clustering based on Emotion Profiles")
plt.tight_layout()
plt.savefig("data/results/emotion_cluster_dendrogram.png")
plt.show()
