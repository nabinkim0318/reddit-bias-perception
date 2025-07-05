### processing/utils_technical_filter.py

import json


def is_blacklist_post(text):
    """
    Detect whether a Reddit post is technical or off-topic
    based on predefined blacklist terms.
    """
    with open("config/blacklist_words.json") as f:
        blacklist = json.load(f)

    technical_blacklist = blacklist["technical_blacklist"]

    text = str(text).lower()
    return any(term in text for term in technical_blacklist)
