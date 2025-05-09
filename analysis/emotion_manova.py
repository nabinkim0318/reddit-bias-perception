### analysis/emotion_manova.py
"""
MANOVA + Post-hoc ANOVA analysis on GoEmotions probabilities
Grouped by bias category derived from topic model
"""

import ast
import json

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import anova_lm

# Load sentiment data
SENTIMENT_CSV = "data/results/sentiment_labeled.csv"
TOPIC_MAP_JSON = "config/topic_to_bias.json"
ANOVA_RESULTS_CSV = "data/results/emotion_anova_results.csv"

print("📥 Loading data...")
df = pd.read_csv(SENTIMENT_CSV)

with open(TOPIC_MAP_JSON, "r") as f:
    topic_map = json.load(f)

# Map topic ID to bias category
df["bias_category"] = df["topic"].map(topic_map)
df = df.dropna(subset=["bias_category", "goemotions_probs"])

# Expand emotion probabilities into columns
print("🔍 Extracting emotion vectors...")
example_probs = ast.literal_eval(df["goemotions_probs"].iloc[0])
emotion_labels = [f"emotion_{i}" for i in range(len(example_probs))]

df_emotions = pd.DataFrame(
    df["goemotions_probs"].apply(ast.literal_eval).to_list(), columns=emotion_labels
)
df = pd.concat([df.reset_index(drop=True), df_emotions], axis=1)

# MANOVA
manova_formula = " + ".join(emotion_labels)
print("🧠 Running MANOVA...")
manova = MANOVA.from_formula(f"{manova_formula} ~ bias_category", data=df)
print(manova.mv_test())

# Post-hoc ANOVA
print("\n📊 Post-hoc ANOVA per emotion:")
results = []
for emotion in emotion_labels:
    model = ols(f"{emotion} ~ C(bias_category)", data=df).fit()
    aov = anova_lm(model, typ=2)
    f_val = round(aov["F"][0], 3)
    p_val = round(aov["PR(>F)"][0], 4)
    results.append((emotion, f_val, p_val))
    if p_val < 0.05:
        print(f"✔️ {emotion}: F = {f_val}, p = {p_val}")

# Save results to CSV
results_df = pd.DataFrame(results, columns=["emotion", "F", "p"])
results_df.to_csv(ANOVA_RESULTS_CSV, index=False)
print(f"\n📁 Saved ANOVA results to {ANOVA_RESULTS_CSV}")
print("\n✅ Analysis complete. Use these results for plotting or reporting.")
