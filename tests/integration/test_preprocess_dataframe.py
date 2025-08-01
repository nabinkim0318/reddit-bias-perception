# import os
# import sys

# import pandas as pd
# import pytest

# # Ensure module path is correct
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from processing.clean_text import preprocess_dataframe


# def test_preprocess_dataframe_schema_and_cleaning():
#     data = [
#         {
#             "id": "001",
#             "subreddit": "example",
#             "title": "Test Title",
#             "selftext": "Some selftext content here.",
#             "comments": ["Interesting...", "Thanks!"],
#             "score": 5,
#             "num_comments": 2,
#             "upvote_ratio": 0.9,
#             "flair": "News",
#             "created_utc": 1610000000.0,
#         }
#     ]
#     df = pd.DataFrame(data)
#     result_df = preprocess_dataframe(df)

#     expected_columns = [
#         "id",
#         "subreddit",
#         "title",
#         "selftext",
#         "comments",
#         "top_comments",
#         "full_text",
#         "clean_text",
#         "score",
#         "num_comments",
#         "upvote_ratio",
#         "flair",
#         "created_utc",
#     ]

#     assert list(result_df.columns) == expected_columns
#     assert len(result_df) == 1
#     assert isinstance(result_df["clean_text"].iloc[0], str)
#     assert isinstance(result_df["top_comments"].iloc[0], list)
#     assert len(result_df["top_comments"].iloc[0]) == 2


# def test_preprocess_dataframe_filters_invalid_rows():
#     data = [
#         {
#             "id": "002",
#             "subreddit": "test",
#             "title": "[deleted]",
#             "selftext": "[removed]",
#             "comments": [],
#             "score": 0,
#             "num_comments": 0,
#             "upvote_ratio": 0.0,
#             "flair": None,
#             "created_utc": 1610000001.0,
#         }
#     ]
#     df = pd.DataFrame(data)
#     result_df = preprocess_dataframe(df)

#     assert result_df.empty
