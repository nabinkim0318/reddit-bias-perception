"""Prompt template still asks for the strict Label: yes|no contract."""

from pathlib import Path


def test_production_template_requests_yes_no_label():
    root = Path(__file__).resolve().parents[2]
    template = (root / "config" / "fewshot_prompt_template.j2").read_text(
        encoding="utf-8"
    )
    assert "Label: yes or no" in template
    assert "{{ post_text }}" in template
    assert "bias" in template.lower()
    assert "Label: <bias|non-bias|uncertain>" not in template
