import csv
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)


# file paths settings
bias_json_path = Path("config/bias_keywords.json")
subreddit_json_path = Path("config/subreddit_groups.json")

bias_csv_path = Path("torrent/bias_keywords.csv")
subreddit_csv_path = Path("torrent/subreddit_groups.csv")


def process_bias_keywords():
    with open(bias_json_path, "r", encoding="utf-8") as f:
        bias_keywords_dict = json.load(f)

    with open(bias_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "keyword"])

        for category, keywords in bias_keywords_dict.items():
            for keyword in keywords:
                cleaned = keyword.strip().lower()
                if cleaned:
                    writer.writerow([category, cleaned])


def process_subreddit_groups():
    with open(subreddit_json_path, "r", encoding="utf-8") as f:
        group_data = json.load(f)

    with open(subreddit_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subreddit", "group"])
        for group, subreddits in group_data.items():
            for sub in subreddits:
                writer.writerow([sub.lower(), group])


def main():
    print("📦 Processing bias keywords and subreddit groups...")
    process_bias_keywords()
    process_subreddit_groups()
    print("✅ Files saved to:")
    print(f"  - {bias_csv_path}")
    print(f"  - {subreddit_csv_path}")


if __name__ == "__main__":
    main()
