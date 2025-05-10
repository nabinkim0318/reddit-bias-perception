### processing/llm_few_shot.py
"""
Few-shot classification using Gemma 2B model to determine whether a Reddit post discusses bias in AI-generated images.
"""

import os
import traceback

import pandas as pd
import torch
from dotenv import load_dotenv
from jinja2 import Template
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.config import (
    BIAS_UNCERTAIN,
    CLASSIFIED_BIAS,
    CLASSIFIED_NONBIAS,
    CLEANED_DATA,
    FEWSHOT_RESULT,
    MODEL_ID,
    OUTPUT_DIR,
    TEMPLATE_PATH,
)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

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

print("🔍 Loading model...")


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", torch_dtype="auto", trust_remote_code=True
    )
    return tokenizer, model


# Lazy load template
from functools import lru_cache


@lru_cache
def get_template():
    with open(TEMPLATE_PATH) as f:
        return Template(f.read())


def build_prompt(post_text):
    return get_template().render(
        instruction=SYSTEM_INSTRUCTION,
        yes_criteria=YES_CRITERIA,
        no_criteria=NO_CRITERIA,
        post=post_text.strip(),
        include_yes_criteria=True,
        include_no_criteria=True,
    )


def classify_post(post_text, tokenizer, model):
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
    if "Output: 1" in decoded:
        return "Yes", decoded
    elif "Output: 0" in decoded:
        return "No", decoded
    else:
        return "Uncertain", decoded


def main():
    tokenizer, model = load_model(MODEL_ID)
    df = pd.read_csv(CLEANED_DATA)
    texts = df["text"].fillna("").astype(str).tolist()
    subreddits = df["subreddit"] if "subreddit" in df.columns else ["unknown"] * len(df)

    results = []
    for i, text in tqdm(enumerate(texts), total=len(texts)):
        try:
            label, output = classify_post(text, tokenizer, model)
        except Exception as e:
            label, output = "Error", traceback.format_exc()
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
    result_df.to_csv(FEWSHOT_RESULT, index=False)

    result_df[result_df["pred_label"] == "Yes"].to_csv(CLASSIFIED_BIAS, index=False)
    result_df[result_df["pred_label"] == "No"].to_csv(CLASSIFIED_NONBIAS, index=False)
    result_df[~result_df["pred_label"].isin(["Yes", "No"])].to_csv(
        BIAS_UNCERTAIN, index=False
    )

    print("✅ Few-shot classification complete. Saved:")
    print("- classified_bias.csv")
    print("- classified_nonbias.csv")
    print("- bias_uncertain.csv")


if __name__ == "__main__":
    main()
