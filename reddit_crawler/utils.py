# reddit_crawler/utils.py
import json
import logging

import pandas as pd

from config.config import SUBREDDIT_GROUPS_PATH
from processing.llm_few_shot import extract_label_and_reasoning, generate_outputs

test_cases = [
    # Definite yes cases (MUST pass)
    (
        "I noticed that whenever I generate AI images of scientists, they almost always appear as white men. It's troubling how biased these models are toward certain demographics.",
        "yes",
    ),
    (
        "Why do all the AI-generated images of athletes show only muscular, young people? There's no diversity in body types at all.",
        "yes",
    ),
    (
        "When I tried generating images of weddings, the couples were always heterosexual and from the same race. This feels really exclusionary.",
        "yes",
    ),
    (
        "It's strange that AI art tools rarely create characters with disabilities. It makes me wonder whether such representation was ever included in the training data.",
        "yes",
    ),
    # Borderline cases (for manual check)
    (
        "I keep seeing AI images where people all look pretty much the same. Does anyone else think this is weird?",
        "borderline",
    ),
    (
        "AI-generated images of festivals often look very generic. They don’t seem to capture cultural diversity at all.",
        "borderline",
    ),
    (
        "I asked the AI to create a street scene, but the people looked weird and unrealistic. It's like they can't capture natural human appearances.",
        "borderline",
    ),
    (
        "Is it just me, or do AI images of CEOs always look the same? I feel like they're stuck on some stereotype.",
        "borderline",
    ),
]


def load_subreddit_groups(path=SUBREDDIT_GROUPS_PATH):
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


def run_tests_with_existing_pipeline(
    test_cases, tokenizer, model, save_csv=False, csv_path="llm_test_results.csv"
):
    """
    Run test cases through the existing LLM pipeline and optionally save results to CSV.

    Args:
        test_cases (list): List of tuples (post_text, expected_label)
        tokenizer: Hugging Face tokenizer
        model: Hugging Face model
        save_csv (bool): Whether to save the test results to CSV.
        csv_path (str): Path for saving the CSV file.
    """
    batch_texts = [case[0] for case in test_cases]
    expected_labels = [case[1] for case in test_cases]

    decoded_outputs = generate_outputs(batch_texts, tokenizer, model)

    passed = 0
    failed = 0
    rows = []

    for idx, (decoded, expected, post_text) in enumerate(
        zip(decoded_outputs, expected_labels, batch_texts), 1
    ):
        label, reasoning = extract_label_and_reasoning(decoded)

        if expected == "yes":
            if label == "yes":
                passed += 1
                logging.info(
                    f"[PASSED ✅] Test Case {idx} - Correctly classified as 'yes'"
                )
            else:
                failed += 1
                logging.error(
                    f"[FAILED ❌] Test Case {idx} - Expected 'yes' but got '{label}'"
                )
                logging.error(f"Post:\n{post_text}\nModel Reasoning:\n{reasoning}\n")
        else:
            logging.info(f"[BORDERLINE] Test Case {idx}")
            logging.info(
                f"Post:\n{post_text}\nPredicted Label: {label}\nReasoning: {reasoning}\n"
            )

        # ✅ Record results
        rows.append(
            {
                "case_id": idx,
                "post_text": post_text,
                "expected_label": expected,
                "predicted_label": label,
                "reasoning": reasoning,
                "raw_output": decoded,
            }
        )

    logging.info(
        f"\n=== TEST SUMMARY ===\nPassed: {passed} / {passed + failed} definite 'yes' cases"
    )
    if failed > 0:
        logging.warning("⚠️ Some 'yes' cases failed. Review model or prompt.")

    # ✅ Save CSV
    if save_csv:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logging.info(f"✅ Test results saved to {csv_path}")
