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

# Load environment variables
load_dotenv()

# Config
MODEL_ID = "google/gemma-2b-it"
INPUT_PATH = os.getenv("FEWSHOT_INPUT_PATH", "data/processed/ai_bias_posts_clean.csv")
OUTPUT_DIR = os.getenv("FEWSHOT_OUTPUT_DIR", "data/processed/")
TEMPLATE_PATH = "templates/fewshot_prompt_template.j2"
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


# Load Jinja2 template once
with open(TEMPLATE_PATH) as f:
    template = Template(f.read())


def build_prompt(post_text):
    return template.render(
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
    # Model loading
    tokenizer, model = load_model(MODEL_ID)

    # Data loading
    df = pd.read_csv(INPUT_PATH)
    texts = df["text"].fillna("").astype(str).tolist()
    subreddits = df["subreddit"] if "subreddit" in df.columns else ["unknown"] * len(df)

    results = []

    # Classify each text
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

    # Saving results
    result_df = pd.DataFrame(results)
    result_df.to_csv(
        os.path.join(OUTPUT_DIR, "fewshot_classification_results.csv"), index=False
    )

    result_df[result_df["pred_label"] == "Yes"].to_csv(
        os.path.join(OUTPUT_DIR, "classified_bias.csv"), index=False
    )
    result_df[result_df["pred_label"] == "No"].to_csv(
        os.path.join(OUTPUT_DIR, "classified_nonbias.csv"), index=False
    )
    result_df[~result_df["pred_label"].isin(["Yes", "No"])].to_csv(
        os.path.join(OUTPUT_DIR, "bias_uncertain.csv"), index=False
    )

    print("✅ Few-shot classification complete. Saved:")
    print("- classified_bias.csv")
    print("- classified_nonbias.csv")
    print("- bias_uncertain.csv")


if __name__ == "__main__":
    main()
