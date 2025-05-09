# reddit_crawler/reddit_client.py

# reddit_crawler/reddit_client.py

import asyncio
import os

import asyncpraw
from dotenv import load_dotenv

load_dotenv()


async def get_reddit_client() -> asyncpraw.Reddit:
    """Return a properly authenticated Reddit client or raise error."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    if not all([client_id, client_secret, user_agent]):
        raise EnvironmentError(
            "❌ Reddit API credentials are missing from environment variables."
        )

    print("🔑 Reddit client authenticated successfully.")
    return asyncpraw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


async def main():
    reddit = await get_reddit_client()
    subreddit = await reddit.subreddit("MachineLearning")
    async for post in subreddit.hot(limit=3):
        print(f"📌 {post.title}")
    await reddit.close()


if __name__ == "__main__":
    asyncio.run(main())
