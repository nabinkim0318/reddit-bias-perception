### utils/text_utils.py
"""
Utility functions for text preprocessing, cleaning, and parsing used across modules.
"""
import ast
import re
from typing import List, Union


def clean_text(text: str) -> str:
    """Basic Reddit text cleaning."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "[URL]", text)
    text = text.replace("&amp;", "&")
    text = re.sub(r"\[deleted\]|\[removed\]", "", text)
    return " ".join(text.split())


def parse_emotion_field(x: Union[str, list]) -> Union[str, list]:
    """Convert string representation of emotion list to Python list."""
    try:
        if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
            parsed = ast.literal_eval(x)
            return parsed if isinstance(parsed, list) else x
        return x
    except:
        return x


def get_primary_emotion(emotion_list: Union[list, str]) -> str:
    """Extract the top emotion from list (or return as-is)."""
    if isinstance(emotion_list, list) and len(emotion_list) > 0:
        return emotion_list[0]
    return str(emotion_list)


def flatten_nested_list(nested_list: Union[list, str]) -> List[str]:
    """Flatten a list of nested lists or strings into a single flat list."""
    if not isinstance(nested_list, list):
        try:
            nested_list = ast.literal_eval(nested_list)
        except:
            return []
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(item)
        elif isinstance(item, str):
            flat.append(item)
    return flat
