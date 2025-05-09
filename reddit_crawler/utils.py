# reddit_crawler/utils.py
import json


def load_subreddit_groups(path="config/subreddit_groups.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("expert", []), data.get("casual", [])


def save_json(path: str, data, indent: int = 2):
    """
    Save a list or dictionary to a JSON file.

    Args:
        data: Data to save (list or dict)
        path: File path to write JSON
        indent: Indentation level for pretty-printing
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    print(f"✅ Saved {len(data)} records to {path}")
