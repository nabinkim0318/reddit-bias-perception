import re
from typing import Optional, List, Dict, Any, Literal, cast, Tuple
from pydantic import ValidationError
import logging

from processing.schema import ClassificationResult


def parse_label_and_reasoning(decoded_output: str) -> Tuple[str, str]:
    """Main function: clean, extract label + reasoning with fallbacks."""
    try:
        cleaned = clean_output(decoded_output)

        # Check for repeated prompt
        if (
            decoded_output.strip().startswith("You are an AI ethics researcher")
            or "Now classify the following post" in decoded_output
            or "{{ post }}" in decoded_output
            or 'Post: "{{' in decoded_output
            or decoded_output.count("Post:") > 1
        ):
            return (
                "no",
                "⚠️ Model failed — repeated system prompt without generating output",
            )

        label = parse_label(cleaned)
        reasoning = parse_reasoning(cleaned)

        if not label:
            # Fallback keyword match
            text_lower = cleaned.lower()
            bias_keywords = [
                "bias",
                "image",
                "representation",
                "diversity",
                "gender",
                "race",
            ]
            if "yes" in text_lower and any(k in text_lower for k in bias_keywords):
                return (
                    "yes",
                    f"Fallback parsing (weak keyword match): {decoded_output.strip()[:100]}",
                )
            else:
                return (
                    "no",
                    f"Fallback (no strong signal): {decoded_output.strip()[:100]}",
                )

        return label, reasoning

    except Exception as e:
        # Failure logging (up to 3 cases)
        if not hasattr(parse_label_and_reasoning, "_logged_failures"):
            parse_label_and_reasoning._logged_failures = 0

        if parse_label_and_reasoning._logged_failures < 3:
            logging.warning(
                f"⚠️ Failed to parse output (case {parse_label_and_reasoning._logged_failures + 1}): {e}"
            )
            logging.warning(f"Raw output: {decoded_output[:500]}...")
            parse_label_and_reasoning._logged_failures += 1
        elif parse_label_and_reasoning._logged_failures == 3:
            logging.warning("⚠️ Suppressing further parsing failure logs...")
            parse_label_and_reasoning._logged_failures += 1

        # Final fallback
        text_lower = decoded_output.lower()
        if "yes" in text_lower and "no" not in text_lower:
            return "yes", f"Fallback parsing: {decoded_output.strip()[:100]}"
        else:
            return "no", f"Fallback parsing: {decoded_output.strip()[:100]}"



def clean_output(decoded_output: str) -> str:
    """Clean the raw model output by removing markdown, system instructions, etc."""
    cleaned = re.sub(r"```json|```", "", decoded_output).strip()
    cleaned = re.sub(
        r"### SYSTEM_INSTRUCTION.*?###",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"system_instruction.*?###",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def parse_label(cleaned: str) -> Optional[str]:
    """Extract label ('yes'/'no') from cleaned output."""
    label_patterns = [
        r'"label"\s*:\s*"([^"]+)"',
        r'label\s*:\s*"([^"]+)"',
        r"label\s*:\s*([a-zA-Z-]+)",
    ]
    for pattern in label_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            label = match.group(1).strip().lower()
            # Normalize
            if label in {"y", "yes", "true", "1"}:
                return "yes"
            elif label in {"n", "no", "false", "0"}:
                return "no"
            elif label in {"yes", "no"}:
                return label
    return None


def parse_reasoning(cleaned: str) -> str:
    """Extract reasoning from cleaned output."""
    reasoning_patterns = [
        r'"reasoning"\s*:\s*"([^"]+)"',
        r'reasoning\s*:\s*"([^"]+)"',
        r"reasoning\s*:\s*([^,\n]+)",
    ]
    for pattern in reasoning_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
            reasoning = re.sub(r"[.,;]+$", "", reasoning)
            return reasoning

    # Try weak fallback: text after label
    label_pos = re.search(r"label\s*:\s*(yes|no)", cleaned, re.IGNORECASE)
    if label_pos:
        after_label = cleaned[label_pos.end() :].strip()
        candidate = re.split(r"\n+", after_label)[0]
        if len(candidate) > 10:
            return candidate
    return "No reasoning provided"


def postprocess_outputs(
    decoded_outputs: List[str],
    batch_texts: List[str],
    batch_ids: List[str],
    batch_subreddits: List[str],
) -> List[Dict[str, Any]]:
    """
    Postprocess model outputs with improved error handling.
    """
    rows = []
    for i, decoded in enumerate(decoded_outputs):
        label, reasoning = parse_label(decoded), parse_reasoning(decoded)
        try:
            # Ensure label is properly typed
            pred_label: Literal["yes", "no"] = cast(Literal["yes", "no"], label)
            row = ClassificationResult(
                id=batch_ids[i],
                subreddit=batch_subreddits[i],
                clean_text=batch_texts[i],
                pred_label=pred_label,
                llm_reasoning=reasoning.strip(),
                raw_output=decoded,
            )
            rows.append(row.model_dump())
        except ValidationError as e:
            logging.error(f"Validation error for ID {batch_ids[i]}: {e}")
            rows.append(
                {
                    "id": batch_ids[i],
                    "subreddit": batch_subreddits[i],
                    "clean_text": batch_texts[i],
                    "pred_label": label,
                    "llm_reasoning": f"Validation Error: {e}",
                    "raw_output": decoded,
                }
            )
    return rows


def main():
    logging.basicConfig(level=logging.INFO)

    # Test with a sample output
    sample_output = """
    """