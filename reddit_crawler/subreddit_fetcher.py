# reddit_crawler/subreddit_fetcher.py

import asyncio
import logging

from processing.utils_technical_filter import is_blacklist_post


async def fetch_posts(reddit, subreddit_name: str, limit: int = 200) -> list:
    """
    Fetch recent posts and top comments from a subreddit.
    """
    subreddit = await reddit.subreddit(subreddit_name)
    submissions = subreddit.new(limit=limit)
    results = []

    async for post in submissions:
        await post.load()

        # Filtering: Exclude technical posts
        combined_text = f"{post.title} {post.selftext}"
        if is_blacklist_post(combined_text):
            continue  # Skip posts with blacklist keywords

        comments = []
        top_comments = []

        if hasattr(post, "comments") and post.comments is not None:
            try:
                await post.comments.replace_more(limit=0)
                # comments: first 10 comments (any depth, sequential order) for random sampling
                comments = [c.body for c in post.comments[:10]]

                # top_comments: top 5 highest-scoring top-level comments
                top_level = [c for c in post.comments if c.is_root]
                top_comments = [
                    c.body
                    for c in sorted(top_level, key=lambda x: x.score, reverse=True)[:5]
                ]

            except Exception as e:
                logging.warning(
                    f"⚠️ Skipped comments in r/{subreddit_name} due to error: {e}"
                )

        results.append(
            {
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
