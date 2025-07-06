# reddit_crawler/subreddit_fetcher.py

import asyncio
import logging

import pandas as pd
from tqdm import tqdm

from config.config import SUBREDDIT_CSV_DIR
from processing.utils_technical_filter import is_blacklist_post


async def safe_fetch(fetch_func, *args, retries=3, sleep_sec=10, **kwargs):
    """
    Wrapper for safe fetching with retry on rate limits (HTTP 429).
    """
    for attempt in range(retries):
        try:
            return await fetch_func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                print(
                    f"⚠️ Rate limited. Sleeping for {sleep_sec} seconds before retry..."
                )
                await asyncio.sleep(sleep_sec)
            else:
                raise e


async def fetch_posts_general(
    reddit, subreddit_name, limit=2000, use_pagination=False, page_size=100
):
    if use_pagination:
        return await safe_fetch(
            fetch_posts_with_pagination,
            reddit,
            subreddit_name,
            total_limit=limit,
            page_size=page_size,
        )
    else:
        return await safe_fetch(fetch_posts, reddit, subreddit_name, limit)


async def fetch_posts_with_pagination(
    reddit, subreddit_name: str, total_limit: int = 1000, page_size: int = 100
):
    """
    Fetch posts with pagination from a subreddit.
    total_limit: total number of posts to fetch
    page_size: number of posts to fetch per page (usually 100~200)
    """
    subreddit = await reddit.subreddit(subreddit_name)
    posts = []
    after = None
    fetched = 0

    while fetched < total_limit:
        submissions = subreddit.new(limit=page_size, params={"after": after})
        async for post in submissions:
            posts.append(post)
            fetched += 1
            after = post.fullname  # Reddit pagination key
            if fetched >= total_limit:
                break

        if fetched == len(posts):
            break

    print(f"✅ Fetched {len(posts)} posts from r/{subreddit_name}")
    return posts


# Fallback method - fetch all posts at once without pagination
async def fetch_posts(reddit, subreddit_name: str, limit: int = 200) -> list:
    """
    Fetch recent posts and top comments from a subreddit.
    """
    subreddit = await reddit.subreddit(subreddit_name)
    submissions = subreddit.new(limit=limit)
    posts_list = [post async for post in submissions]  # Async generator to list

    results = []
    # Process posts with progress bar
    for post in tqdm(posts_list, desc=f"📥 Fetching r/{subreddit_name}", unit="post"):
        await post.load()
        post.comment_sort = "top"  # Explicit sort method

        # Filtering: Exclude technical posts
        combined_text = f"{post.title} {post.selftext}"
        if is_blacklist_post(combined_text):
            continue  # Skip posts with blacklist keywords

        comments, top_comments = await extract_comments(post)
        results.append(serialize_post(post, subreddit_name, comments, top_comments))

    return results


async def extract_comments(post):
    """
    Extracts comments and top-level comments from a post.
    """
    comments = []
    top_comments = []
    if hasattr(post, "comments") and post.comments is not None:
        try:
            await post.comments.replace_more(limit=0)
            comments = [c.body for c in post.comments[:10]]
            top_level = [c for c in post.comments if c.is_root]
            top_comments = [
                c.body
                for c in sorted(top_level, key=lambda x: x.score, reverse=True)[:5]
            ]
        except Exception as e:
            logging.warning(f"⚠️ Failed to process comments for post {post.id}: {e}")
    return comments, top_comments


def serialize_post(post, subreddit_name, comments, top_comments):
    """
    Serializes a post object to a dictionary.
    """
    return {
        "id": post.id,
        "subreddit": subreddit_name,
        "title": post.title,
        "selftext": post.selftext,
        "comments": comments,
        "top_comments": top_comments,
        "score": post.score,
        "num_comments": post.num_comments,
        "upvote_ratio": post.upvote_ratio,
        "flair": post.link_flair_text,
        "created_utc": post.created_utc,
    }


async def fetch_all(
    reddit,
    subreddits: list,
    limit: int = 250,
    sleep_sec: int = 2,
    use_pagination: bool = True,
    output_dir: str = SUBREDDIT_CSV_DIR,
) -> list:
    """
    Fetch posts from multiple subreddits and save per-subreddit CSV.
    """
    all_data = []
    for sub in tqdm(subreddits, desc="🌐 Crawling subreddits", unit="subreddit"):
        try:
            posts = await fetch_posts_general(
                reddit, sub, limit=limit, use_pagination=use_pagination
            )
            all_data.extend(posts)

            # Save per-subreddit CSV
            df = pd.DataFrame(posts)
            df.drop_duplicates(subset=["id"], inplace=True)  # duplicates removal
            sanitized_sub = sub.replace("/", "_")  # safe file name
            csv_path = f"{output_dir}/{sanitized_sub}.csv"
            df.to_csv(csv_path, index=False)
            print(f"📄 Saved {len(posts)} posts to {csv_path}")

        except Exception as e:
            logging.warning(f"⚠️ Failed to fetch r/{sub}: {e}")
        await asyncio.sleep(sleep_sec)  # Sleep between subreddits
    return all_data
