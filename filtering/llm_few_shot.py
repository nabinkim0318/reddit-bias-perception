### filtering/llm_few_shot.py
"""
Few-shot classification using Gemma 2B model to determine whether a Reddit post discusses bias in AI-generated images.
"""

import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer

# Config
MODEL_ID = "google/gemma-2b-it"
INPUT_PATH = os.getenv("FEWSHOT_INPUT_PATH", "data/processed/ai_bias_posts_clean.csv")
OUTPUT_DIR = os.getenv("FEWSHOT_OUTPUT_DIR", "data/processed/")

# Classification instruction (system prompt)
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Your job is to classify whether the following Reddit post "
    "reflects a **concern, observation, or critique about bias** in AI-generated images."
)

YES_CRITERIA = (
    "- A complaint or observation about misrepresentation related to race, gender, body type, culture, disability, or religion\n"
    "- Frustration about the lack of diversity or overuse of stereotypical visuals in AI output\n"
    "- Emotional, social, or ethical concerns about fairness, exclusion, or how certain identities are depicted or erased\n"
    "- Even if subtle, the post reflects discomfort or surprise with how identity is visualized or how bias manifests in image generation"
)

NO_CRITERIA = (
    "- Focuses on technical details, creative tools, styles, rendering techniques, or model features\n"
    "- Shares AI-generated art, music, or fictional work without discussing fairness, identity, or social representation\n"
    "- Describes how the AI responded to prompts with no identity-related concern\n"
    "- Reflects on the creative process or philosophical/artistic value of AI output without reference to social bias or exclusion"
)

# Load model and tokenizer
print("🔍 Loading model...")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto", trust_remote_code=True
)


def build_prompt(post_text):
    return f"""<s> [INST] {SYSTEM_INSTRUCTION}

A post should be classified as \"1\" (Yes) if it includes:
{YES_CRITERIA}

A post should be classified as \"0\" (No) if it:
{NO_CRITERIA}

Now decide:
Post: \"{post_text.strip()}\"
[/INST]
Output:"""


def classify_post(post_text):
    prompt = build_prompt(post_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    match = re.search(r"\[/INST\]\s*Output:\s*(\d)", decoded)
    if match:
        label = "Yes" if match.group(1) == "1" else "No"
    else:
        label = "Uncertain"
    return label, decoded


def main():
    df = pd.read_csv(INPUT_PATH)
    texts = df["text"].fillna("").astype(str).tolist()
    subreddits = df["subreddit"] if "subreddit" in df.columns else ["unknown"] * len(df)

    results = []
    for i, text in tqdm(enumerate(texts), total=len(texts)):
        try:
            label, output = classify_post(text)
        except Exception as e:
            label, output = "Error", str(e)
        results.append(
            {
                "index": i,
                "text": text,
                "subreddit": subreddits[i],
                "pred_label": label,
                "full_output": output,
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        os.path.join(OUTPUT_DIR, "classified_fewshot_all.csv"), index=False
    )

    result_df[result_df["pred_label"] == "Yes"].to_csv(
        os.path.join(OUTPUT_DIR, "included_BCD_fewshot.csv"), index=False
    )
    result_df[result_df["pred_label"] == "No"].to_csv(
        os.path.join(OUTPUT_DIR, "excluded_AE_fewshot.csv"), index=False
    )
    result_df[~result_df["pred_label"].isin(["Yes", "No"])].to_csv(
        os.path.join(OUTPUT_DIR, "uncertain_fewshot.csv"), index=False
    )

    print("✅ Few-shot classification complete. Saved:")
    print("- included_BCD_fewshot.csv")
    print("- excluded_AE_fewshot.csv")
    print("- uncertain_fewshot.csv")


if __name__ == "__main__":
    main()
