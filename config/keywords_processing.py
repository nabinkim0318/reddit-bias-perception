# config/keywords_processing.py
import csv
import json
import logging
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


# file paths settings
bias_json_path = Path("config/bias_keywords.json")
subreddit_json_path = Path("config/subreddit_groups.json")

bias_csv_path = Path("config/bias_keywords.csv")
subreddit_csv_path = Path("config/subreddit_groups.csv")


def process_bias_keywords():
    with open(bias_json_path, "r", encoding="utf-8") as f:
        bias_keywords_dict = json.load(f)

    with open(bias_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "keyword"])

        for category, keywords in bias_keywords_dict.items():
            for keyword in tqdm(
                keywords, desc="Processing bias keywords", total=len(keywords)
            ):
                cleaned = keyword.strip().lower()
                if cleaned:
                    writer.writerow([category, cleaned])


def process_subreddit_groups():
    with open(subreddit_json_path, "r", encoding="utf-8") as f:
        group_data = json.load(f)

    with open(subreddit_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subreddit", "group"])
        for group, subreddits in tqdm(
            group_data.items(),
            desc="Processing subreddit groups",
            total=len(group_data),
        ):
            for sub in tqdm(
                subreddits, desc="Processing subreddit groups", total=len(subreddits)
            ):
                writer.writerow([sub.lower(), group])


def main():
    logging.info("📦 Processing bias keywords and subreddit groups...")
    process_bias_keywords()
    process_subreddit_groups()
    logging.info("✅ Files saved to:")
    logging.info(f"  - {bias_csv_path}")
    logging.info(f"  - {subreddit_csv_path}")


if __name__ == "__main__":
    main()
