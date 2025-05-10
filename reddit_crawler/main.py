### reddit_crawler/main.py
"""
Reddit crawler CLI entry point.
Fetches posts + comments from subreddits and saves to JSON.
"""

import argparse
import asyncio
import logging

from reddit_crawler.reddit_client import get_reddit_client
from reddit_crawler.subreddit_fetcher import fetch_all
from reddit_crawler.utils import load_subreddit_groups, save_json

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reddit data crawler for AI bias project"
    )
    parser.add_argument(
        "--limit", type=int, default=250, help="Max posts per subreddit"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/reddit_raw.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--sleep", type=int, default=2, help="Seconds to sleep between subreddits"
    )
    return parser.parse_args()


async def run_crawler(limit=250, output="data/raw/reddit_raw.json", sleep=2):
    expert_subs, casual_subs = load_subreddit_groups()
    all_subs = expert_subs + casual_subs

    print(f"📡 Starting crawl for {len(all_subs)} subreddits...")

    reddit = await get_reddit_client()
    try:
        all_data = await fetch_all(reddit, all_subs, limit=limit, sleep_sec=sleep)
        save_json(output, all_data)
        print(f"✅ Finished. Saved {len(all_data)} posts to {output}")
    finally:
        await reddit.close()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_crawler(args.limit, args.output, args.sleep))
