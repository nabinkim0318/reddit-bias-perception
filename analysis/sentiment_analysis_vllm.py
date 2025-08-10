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

MODEL_ID = os.getenv("LLM_MODEL_ID", "meta-llama/Llama-3.2-8B-Instruct")

llm = LLM(
    model=MODEL_ID,
    dtype="half",              # T4: use FP16
    max_model_len=2048,
    trust_remote_code=True,
    gpu_memory_utilization=0.92,
)
sampler = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=160,            # plenty for 5 emotions JSON
)

SYSTEM_INSTRUCTIONS = (
    "You are an emotion analysis engine. "
    "Classify the input text into the TOP-5 emotions with strengths in [0,1]. "
    "Output STRICT JSON only: a list of exactly 5 objects with fields "
    '\"emotion\" (string) and \"score\" (number between 0 and 1). '
    "No prose, no markdown, no extra keys."
)

def build_prompt(text: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_INSTRUCTIONS}\n</|system|>\n"
        f"<|user|>\nText: {text}\n</|user|>\n"
        "<|assistant|>"
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
            outputs.append(parse_top5(text_out))
    return outputs

def main():
    # paths from your config; or set directly
    input_csv = FINAL_ANALYSIS_INPUT
    output_csv = SENTIMENT_OUTPUT

    df = pd.read_csv(input_csv)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)

    # Filter non-empty rows for inference, but keep original indexing
    mask = df["clean_text"].str.strip() != ""
    texts = df.loc[mask, "clean_text"].tolist()

    print(f"🧠 Running LLM emotion classification on {len(texts)} texts via vLLM...")
    top5_lists = llm_emotions_for_texts(texts, batch_size=16)

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