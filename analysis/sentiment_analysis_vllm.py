import json
import os
import re
import pandas as pd
import torch
from vllm import LLM, SamplingParams

# Hugging Face gated model note:
# You'll need access to meta-llama/Llama-3.2-8B-Instruct on your HF account.
# If not, swap to an accessible instruct model (e.g., llama-3.1-8b-instruct).
from config.config import (
    EMOTION_MODEL,
    FINAL_ANALYSIS_INPUT,
    SENTIMENT_OUTPUT,
    VADER_PLOT_PATH,
    SENTIMENT_PLOT_PATH
)

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

llm = LLM(
    model=MODEL_ID,
    dtype="bfloat16",              
    max_model_len=8192,
    trust_remote_code=True,
    gpu_memory_utilization=0.90,
)

sampler = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=320,
    stop=["<|eot_id|>","</json>","</s>","<|end|>"]  # harmless extra stops
)


SYSTEM_INSTRUCTIONS = """You are an emotion analysis engine.

Task
- From the input text, infer emotions from Plutchik’s primary set:
  anger, anticipation, joy, trust, fear, surprise, sadness, disgust.
- Return ONLY the top-5 emotions by strength (multi-label), each with a score in [0,1].
- Output STRICT JSON: a list of EXACTLY 5 objects with keys:
  "emotion" (one of the 8 above) and "score" (float in [0,1]).
- Sort by descending "score". No prose, no markdown, no extra keys.

Output format
- Return ONLY JSON (no explanation).
- EXACTLY 5 items; if fewer than five emotions are apparent, fill the remaining with the weakest plausible ones (very low scores) to keep list length 5.
- Sort by descending score; no ties.
"""
# Scoring policy
# - Start with lexical/semantic cues; adjust with:
#   • Intensifiers: (“very”, “incredibly”) → increase score by ~0.1–0.25
#   • Downtoners: (“slightly”, “a bit”) → decrease score by ~0.05–0.15
#   • Negation: (“not happy”, “no fear”) → invert/attenuate expected signal
#   • Modality/uncertainty: (“might”, “maybe”, “I guess”) → lower scores
#   • Sarcasm/irony indicators: exaggeration + negative context → raise anger/disgust or sadness as appropriate; reduce joy unless sincere
#   • Temporal distance: past resolved events usually reduce arousal unless text shows lingering affect
# - Typical ranges:
#   • Dominant emotion: 0.55–0.90 (rarely >0.90)
#   • Secondary: 0.25–0.60
#   • Weak/background: 0.05–0.30
# - If evidence is extremely weak or ambiguous, keep all scores low (≤0.25).

# Disambiguation tips
# - anger vs disgust: anger targets an agent/action (“they lied”); disgust targets contamination/morality (“gross”, “vile”)
# - fear vs anticipation: fear = threat expectation; anticipation = forward-looking expectation (can be positive or worried)
# - joy vs trust: joy = felt positivity/pleasure; trust = safety/reliability/assurance cues
# - sadness vs anger: sadness = loss/helplessness; anger = blame/agency

# Edge cases
# - Neutral/informational text: return low scores across categories (e.g., 0.05–0.15).
# - Mixed sentiment (“bittersweet”): include both positive (joy/trust/anticipation) and negative (sadness/fear) with moderate values.
# - Emoji/exclamations:
#   • “!!!”, “😂”, “😭” → boost relevant emotions slightly
# - Profanity alone does NOT guarantee anger—use context.

# Language & domain
# - Be domain- and culture-agnostic; do not import world knowledge beyond the text.
# - Multilingual: if non-English, infer using common affective cues; still return English emotion labels.

def build_prompt(text: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_INSTRUCTIONS}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Text:\n{text}\n"
        "Respond with ONLY the JSON list of 5 objects as specified.\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

_json_start_re = re.compile(r'\[', re.DOTALL)
_json_end_re = re.compile(r'\](?!.*\])', re.DOTALL)

def parse_top5(json_like: str):
    """
    Parse model output into a Python list of 5 dicts.
    Tries strict JSON first; falls back to bracket slicing + json.loads.
    """
    try:
        data = json.loads(json_like)
        if isinstance(data, list) and len(data) == 5:
            return data
    except Exception:
        pass

    # Fallback: extract outermost JSON array
    try:
        s = _json_start_re.search(json_like)
        e = _json_end_re.search(json_like)
        if s and e:
            chunk = json_like[s.start():e.end()]
            data = json.loads(chunk)
            if isinstance(data, list) and len(data) == 5:
                return data
    except Exception:
        pass

    # Final fallback: return empty schema if irrecoverable
    return [
        {"emotion": "", "score": 0.0},
        {"emotion": "", "score": 0.0},
        {"emotion": "", "score": 0.0},
        {"emotion": "", "score": 0.0},
        {"emotion": "", "score": 0.0},
    ]

def llm_emotions_for_texts(texts, batch_size=16):
    prompts = [build_prompt(t) for t in texts]
    outputs = []

    # vLLM can take the whole list at once; batching keeps memory predictable on T4
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        results = llm.generate(batch_prompts, sampler)
        # results is a list aligned with batch_prompts
        for res in results:
            text_out = res.outputs[0].text.strip()
            #outputs.append(parse_top5(text_out))
            outputs.append(text_out)
    return outputs

def main():
    # paths from your config; or set directly
    # input_csv = FINAL_ANALYSIS_INPUT
    input_csv = '/home/tsutar3/reddit-bias-perception/data/processed/final_filtered_df_old.csv'
    # output_csv = SENTIMENT_OUTPUT
    output_csv = "/home/tsutar3/reddit-bias-perception/results/sentiments_labelled_old_file_input.csv"

    df = pd.read_csv(input_csv)
    df = df[~df['flair'].str.startswith('Question', na=False)]
    df["combined_text"] = df["combined_text"].fillna("").astype(str)

    # Filter non-empty rows for inference, but keep original indexing
    mask = df["combined_text"].str.strip() != ""
    texts = df.loc[mask, "combined_text"].tolist()

    print(f"🧠 Running LLM emotion classification on {len(texts)} texts via vLLM...")
    top5_lists = llm_emotions_for_texts(texts, batch_size=8)

    # Store JSON strings to be CSV-safe
    df.loc[mask, "llm_classification"] = [json.dumps(x, ensure_ascii=False) for x in top5_lists]
    df.loc[~mask, "llm_classification"] = json.dumps(
        [
            {"emotion": "", "score": 0.0},
            {"emotion": "", "score": 0.0},
            {"emotion": "", "score": 0.0},
            {"emotion": "", "score": 0.0},
            {"emotion": "", "score": 0.0},
        ]
    )

    df.to_csv(output_csv, index=False)
    print(f"✅ LLM classification complete. Output saved to: {output_csv}")

if __name__ == "__main__":
    main()
