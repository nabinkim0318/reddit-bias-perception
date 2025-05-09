# reddit_crawler/reddit_client.py

import os

import asyncpraw
from dotenv import load_dotenv

# Load .env variables (if not already loaded in main)
load_dotenv()


async def get_reddit_client():
    """
    Returns an authenticated asyncpraw.Reddit client using environment variables.

    Environment variables expected:
        - REDDIT_CLIENT_ID
        - REDDIT_CLIENT_SECRET
        - REDDIT_USER_AGENT
    """
    return asyncpraw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )
