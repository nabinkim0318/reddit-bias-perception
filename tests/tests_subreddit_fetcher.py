# tests/test_subreddit_fetcher.py

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reddit_crawler.subreddit_fetcher import fetch_all


@pytest.mark.asyncio
async def test_fetch_all_success():
    # Mock Reddit and subreddit response
    mock_reddit = AsyncMock()
    mock_subreddit = AsyncMock()
    mock_submission = AsyncMock()
    mock_submission.title = "Test Title"
    mock_submission.selftext = "Test Body"
    mock_submission.id = "abc123"
    mock_submission.comments = [MagicMock(body="Comment 1")]
    mock_submission.score = 123
    mock_submission.created_utc = 1715000000

    # simulate async iteration
    mock_subreddit.new.return_value = [mock_submission]
    mock_reddit.subreddit.return_value = mock_subreddit

    result = await fetch_all(mock_reddit, ["mocksub"], limit=1, sleep_sec=0)
    print("DEBUG result:", result)

    assert len(result) == 1
    assert result[0]["title"] == "Test Title"
    assert result[0]["comments"][0] == "Comment 1"


@pytest.mark.asyncio
async def test_fetch_all_failure_handled_gracefully():
    mock_reddit = AsyncMock()
    mock_reddit.subreddit.side_effect = Exception("API failure")

    result = await fetch_all(mock_reddit, ["fail_sub"], limit=1, sleep_sec=0)
    print("DEBUG result:", result)
    assert result == []  # Expect graceful fallback
