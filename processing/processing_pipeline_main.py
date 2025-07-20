import asyncio
import time

from processing.clean_text import main as clean_text
from processing.keyword_filter import main as keyword_filter
from processing.llm_few_shot_pipeline import main as llm_filter
from processing.duckdb_data_processing import main as duckdb_data_processing
import argparse


def timed_step(label, func, subreddit: str):
    print(f"\n🚩 [{label}] started...")
    start = time.time()
    func(subreddit)
    end = time.time()
    print(f"✅ [{label}] completed in {end - start:.2f} seconds")


def main(subreddit: str):
    timed_step("1. DuckDB Data Processing", duckdb_data_processing, subreddit)
    # timed_step("2. Text Cleaning", clean_text)
    # timed_step("3. Keyword Filtering", keyword_filter)
    # timed_step("4. LLM Filtering", llm_filter)

    print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddit", type=str, required=True, help="Subreddit name (e.g., 'midjourney')")
    args = parser.parse_args()
    main(args.subreddit)
