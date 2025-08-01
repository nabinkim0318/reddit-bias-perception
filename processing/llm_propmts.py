import logging
from functools import lru_cache

from jinja2 import BaseLoader, Environment

from config.config import TEMPLATE_PATH


# === Utilities ===
@lru_cache(maxsize=1)
def get_template():
    """
    Load and cache the Jinja2 prompt template from file.
    """
    try:
        with open(TEMPLATE_PATH, "r") as f:
            template_text = f.read()
            env = Environment(
                loader=BaseLoader(),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            return env.from_string(template_text)
    except FileNotFoundError as e:
        logging.error(f"❌ Template file not found at {TEMPLATE_PATH}: {e}")
        raise e


def build_prompt(post_text: str) -> str:
    """
    Render the classification prompt using Jinja2 template.
    """
    try:
        return get_template().render(post_text=(post_text or "").strip())
    except Exception as e:
        logging.warning(f"⚠️ Failed to render prompt: {e}")
        return f"Post: {post_text}"


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("🔍 Starting LLM prompt generation...")

    # Test with a sample post
    sample_post = "This is a test post about AI bias."
    prompt = build_prompt(sample_post)
    logging.info(f"Generated prompt:\n{prompt}")
