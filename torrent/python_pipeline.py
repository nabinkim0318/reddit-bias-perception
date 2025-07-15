from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import torrent

# ✅ Example: stopwords removal only (Will add more postprocessing later)
stopwords = set(ENGLISH_STOP_WORDS)


def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stopwords])


# DuckDB → Python: remove stopwords
df = torrent.query("SELECT id, clean_text FROM posts").to_df()
df["clean_text"] = df["clean_text"].apply(remove_stopwords)

# Add postprocessing
# df["matched_categories"] = df["matched_categories"].apply(lambda x: ", ".join(x) if x else "")
# df["matched_keywords"] = df["matched_keywords"].apply(lambda x: ", ".join(x) if x else "")


# Add expected columns
for col in expected_columns:
    if col not in df.columns:
        df[col] = None

# Save
df.to_csv("cleaned_output.csv", index=False)
