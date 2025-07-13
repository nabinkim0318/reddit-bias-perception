from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import duckdb

# ✅ 예: stopwords 제거만 Python으로 처리 추가
stopwords = set(ENGLISH_STOP_WORDS)


def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stopwords])


# DuckDB로 정제된 텍스트 → Python으로 stopwords 제거
df = duckdb.query("SELECT id, clean_text FROM posts").to_df()
df["clean_text"] = df["clean_text"].apply(remove_stopwords)

# 후처리 추가
# df["matched_categories"] = df["matched_categories"].apply(lambda x: ", ".join(x) if x else "")
# df["matched_keywords"] = df["matched_keywords"].apply(lambda x: ", ".join(x) if x else "")


# expected columns 보완
for col in expected_columns:
    if col not in df.columns:
        df[col] = None

# 저장
df.to_csv("cleaned_output.csv", index=False)
