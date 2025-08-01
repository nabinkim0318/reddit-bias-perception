# import os
# from unittest import mock

# import pytest


# # ✅ Fixture to inject environment variables for testing
# @pytest.fixture
# def mock_env():
#     with mock.patch.dict(
#         os.environ,
#         {
#             "REDDIT_CLIENT_ID": "dummy_id",
#             "REDDIT_CLIENT_SECRET": "dummy_secret",
#             "REDDIT_USER_AGENT": "dummy_agent",
#         },
#     ):
#         yield


# # ✅ Test successful Reddit client creation with mock environment
# @pytest.mark.asyncio
# async def test_get_reddit_client_success(mock_env):
#     from reddit_crawler.reddit_client import get_reddit_client

#     reddit = await get_reddit_client()
#     try:
#         # Verify the returned object has expected attributes
#         assert hasattr(reddit, "subreddit")
#         assert callable(reddit.subreddit)
#     finally:
#         # Ensure aiohttp session is properly closed
#         await reddit.close()


# # ✅ Test that missing environment variables raises error
# @pytest.mark.asyncio
# async def test_get_reddit_client_missing_env():
#     from reddit_crawler.reddit_client import get_reddit_client

#     with mock.patch.dict(os.environ, {}, clear=True):
#         with pytest.raises(EnvironmentError, match="credentials are missing"):
#             await get_reddit_client()
