import pytest

from reddit_crawler.reddit_client import get_reddit_client
from reddit_crawler.subreddit_fetcher import fetch_posts


@pytest.mark.asyncio
async def test_fetch_posts_small():
    reddit = await get_reddit_client()  # ✅ 코루틴을 실행해서 실제 객체 받기
    try:
        posts = await fetch_posts(reddit, "midjourney", limit=3)
        assert isinstance(posts, list)
        assert len(posts) <= 3
        assert all("id" in post for post in posts)
    finally:
        await reddit.close()
