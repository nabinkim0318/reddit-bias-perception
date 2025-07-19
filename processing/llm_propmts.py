from functools import lru_cache
from jinja2 import Environment, BaseLoader
import logging

from config.config import TEMPLATE_PATH



# === Utilities ===
@lru_cache(maxsize=1)
def get_template():
    """
    Load and cache the Jinja2 prompt template from file.
    """
    with open(TEMPLATE_PATH, "r") as f:
        template_text = f.read()
        env = Environment(
            loader=BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return env.from_string(template_text)


def build_prompt(post_text: str) -> str:
    """
    Render the classification prompt using Jinja2 template.
    """
    rendered = get_template().render(post_text=(post_text or "").strip())
    return rendered


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("🔍 Starting LLM prompt generation...")

    # Test with a sample post
    sample_post = "This is a test post about AI bias."
    prompt = build_prompt(sample_post)
    logging.info(f"Generated prompt:\n{prompt}")