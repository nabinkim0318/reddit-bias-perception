# reddit_crawler/subreddit_fetcher.py

import asyncio
import logging


async def fetch_posts(reddit, subreddit_name: str, limit: int = 200) -> list:
    """
    Fetch recent posts and top comments from a subreddit.
    """
    subreddit = await reddit.subreddit(subreddit_name)
    submissions = subreddit.new(limit=limit)
    results = []

    async for post in submissions:
        await post.load()

        if not hasattr(post, "comments") or post.comments is None:
            comments = []
        else:
            try:
                await post.comments.replace_more(limit=0)
                comments = [c.body for c in post.comments[:3]]
            except Exception as e:
                logging.warning(
                    f"⚠️ Skipped comments in r/{subreddit_name} due to error: {e}"
                )
                comments = []

        results.append(
            {
                "id": post.id,
                "subreddit": subreddit_name,
                "title": post.title,
                "selftext": post.selftext,
                "comments": comments,
                "score": post.score,
                "created_utc": post.created_utc,
            }
        )

    return results


async def fetch_all(
    reddit, subreddits: list, limit: int = 250, sleep_sec: int = 2
) -> list:
    """
    Fetch posts from multiple subreddits in parallel with robust error handling.
    """

    all_data = []
    for sub in subreddits:
        logging.info(f"📥 Fetching from r/{sub}...")
        try:
            posts = await fetch_posts(reddit, sub, limit)
            all_data.extend(posts)
        except Exception as e:
            logging.warning(f"⚠️ Failed to fetch r/{sub}: {e}")
        await asyncio.sleep(sleep_sec)  # <-- sleep between requests
    return all_data
