import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stopwords = set(ENGLISH_STOP_WORDS)


def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stopwords])


# DuckDB → Python: remove stopwords
def main():
    subreddit = "aiwars"
    df = pd.read_csv(f"data/filtered/{subreddit}_full_filtered_posts.csv")
    df["clean_text"] = df["clean_text"].apply(remove_stopwords)
    df["matched_categories"] = df["matched_categories"].apply(
        lambda x: ", ".join(x) if x else ""
    )
    df["matched_keywords"] = df["matched_keywords"].apply(
        lambda x: ", ".join(x) if x else ""
    )
    df.to_csv(
        f"data/filtered/{subreddit}_full_filtered_posts_cleaned_posts.csv", index=False
    )


if __name__ == "__main__":
    main()
