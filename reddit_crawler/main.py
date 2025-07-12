### reddit_crawler/main.py
"""
Reddit crawler CLI entry point.
Fetches posts + comments from subreddits and saves to JSON.
"""

import asyncio
import logging

from config.config import RAW_REDDIT_DATA
from reddit_crawler.reddit_client import get_reddit_client
from reddit_crawler.subreddit_fetcher import (
    fetch_all,
    fetch_reddit_posts,
    fetch_reddit_posts_v2,
)
from reddit_crawler.utils import load_subreddit_groups, save_json

logging.basicConfig(level=logging.INFO)


async def run_crawler(output=RAW_REDDIT_DATA):
    expert_subs, casual_subs = load_subreddit_groups()
    all_subs = expert_subs + casual_subs

    print(f"📡 Starting crawl for {len(all_subs)} subreddits...")

    reddit = await get_reddit_client()
    try:
        all_data = await fetch_all(reddit, all_subs, use_pagination=False)
        save_json(output, all_data)
        print(
            f"✅ Finished. Saved {len(all_subs)} subreddits processed, {len(all_data)} posts saved to {output}"
        )
    finally:
        await reddit.close()


def main():
    df = fetch_reddit_posts_v2("AIGenArt", max_posts=3000)
    df.to_csv("AIGenArt_posts_v2.csv", index=False)


if __name__ == "__main__":
    # asyncio.run(run_crawler())
    main()
