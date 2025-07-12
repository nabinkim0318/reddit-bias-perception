# reddit_crawler/subreddit_fetcher.py

import asyncio
import logging
import time

import pandas as pd
import requests
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


async def fetch_posts_general(reddit, subreddit_name, limit=2000, use_pagination=False):
    if use_pagination:
        return await safe_fetch(
            fetch_posts_with_pagination, reddit, subreddit_name, total_limit=limit
        )
    else:
        return await safe_fetch(fetch_posts, reddit, subreddit_name, limit)


async def fetch_posts_with_pagination(
    reddit, subreddit_name: str, total_limit: int = 1000
):
    subreddit = await reddit.subreddit(subreddit_name)
    posts = []
    fetched = 0

    async for post in subreddit.new(limit=total_limit):
        post.comment_sort = "top"
        await post.load()

        combined_text = f"{post.title} {post.selftext}"
        if is_blacklist_post(combined_text):
            continue

        comments, top_comments = await extract_comments(post)
        post_data = serialize_post(post, subreddit_name, comments, top_comments)
        posts.append(post_data)
        fetched += 1

        if fetched >= total_limit:
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

            # 🔍 Check first before deduplication
            if "id" in df.columns:
                df.drop_duplicates(subset=["id"], inplace=True)
            else:
                print("⚠️ Skipped deduplication: 'id' column not found in DataFrame")

            sanitized_sub = sub.replace("/", "_")  # safe file name
            csv_path = f"{output_dir}/{sanitized_sub}.csv"
            df.to_csv(csv_path, index=False)
            print(f"📄 Saved {len(posts)} posts to {csv_path}")

        except Exception as e:
            logging.warning(f"⚠️ Failed to fetch r/{sub}: {e}")
        await asyncio.sleep(sleep_sec)  # Sleep between subreddits
    return all_data


def fetch_reddit_posts(subreddit, max_posts=500, sleep_sec=2):
    url = f"https://www.reddit.com/r/{subreddit}/new.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    posts = []
    after = None

    while len(posts) < max_posts:
        params = {"limit": 100}
        if after:
            params["after"] = after

        try:
            res = requests.get(url, headers=headers, params=params)
            if res.status_code != 200:
                print(f"⚠️ Failed with status code {res.status_code}")
                break

            data = res.json()
            children = data["data"]["children"]

            for item in children:
                post = item["data"]
                posts.append(
                    {
                        "id": post["id"],
                        "title": post["title"],
                        "selftext": post["selftext"],
                        "created_utc": post["created_utc"],
                        "score": post["score"],
                        "num_comments": post["num_comments"],
                    }
                )

            after = data["data"]["after"]
            if not after:
                break  # 더 이상 가져올 게 없음
            print(f"✅ Retrieved {len(posts)} posts so far...")

            time.sleep(sleep_sec)  # Reddit 서버에 부담 주지 않도록 쉬어줌

        except Exception as e:
            print(f"❌ Error: {e}")
            break

    return pd.DataFrame(posts)


def fetch_reddit_posts_v2(subreddit, max_posts=3000, sleep_sec=10):
    url = f"https://www.reddit.com/r/{subreddit}/new.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    posts = []
    after = None
    fetched = 0

    while fetched < max_posts:
        limit = min(100, max_posts - fetched)  # 한 번에 최대 100개
        params = {"limit": limit}
        if after:
            params["after"] = after

        try:
            res = requests.get(url, headers=headers, params=params)
            if res.status_code != 200:
                print(f"⚠️ Status code {res.status_code} — stopping.")
                break

            data = res.json()
            children = data["data"]["children"]
            if not children:
                print("⚠️ No more posts to fetch.")
                break

            for item in children:
                post = item["data"]
                posts.append(
                    {
                        "id": post["id"],
                        "title": post.get("title", ""),
                        "selftext": post.get("selftext", ""),
                        "created_utc": post.get("created_utc", 0),
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "upvote_ratio": post.get("upvote_ratio", 0),
                        "subreddit": post.get("subreddit", subreddit),
                    }
                )

            fetched += len(children)
            after = data["data"].get("after")
            print(f"✅ Fetched {fetched} posts from r/{subreddit}...")

            if not after:
                break  # 더 이상 없을 때

            time.sleep(sleep_sec)  # 서버 보호

        except Exception as e:
            print(f"❌ Error: {e}")
            break

    return pd.DataFrame(posts)
