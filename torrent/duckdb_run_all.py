from torrent.duckdb_processing import load_and_preview_jsonl
from torrent.duckdb_pipeline import main as duckdb_pipeline_main
from torrent.python_pipeline import run as python_pipeline_main
import logging
import argparse

logging.basicConfig(level=logging.INFO)


def main(subreddit: str):
    logging.info(f"🚀 Starting pipeline for subreddit: {subreddit}")

    # Step 1: Decompress + Preview
    df = load_and_preview_jsonl(subreddit)

    # Step 2: DuckDB Filtering
    duckdb_pipeline_main(subreddit)

    # Step 3: Python-side LLM Classification
    python_pipeline_main(subreddit)

    logging.info(f"✅ Completed full pipeline for subreddit: {subreddit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full Reddit pipeline")
    parser.add_argument("--subreddit", type=str, required=True, help="Subreddit name (e.g., aiwars)")

    args = parser.parse_args()
    main(args.subreddit)