# scripts/make_test_fixture.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import zstandard as zstd

# ---- 기본 설정 ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 루트/ from scripts/
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED = DATA_DIR / "extracted"
CONFIG = PROJECT_ROOT / "config"
PROCESSED = DATA_DIR / "processed" / "filtered"

SUBREDDIT = "TestAI"  # 테스트용 서브레딧 이름

EXTRACTED.mkdir(parents=True, exist_ok=True)
CONFIG.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

jsonl_path = EXTRACTED / f"{SUBREDDIT}_posts.jsonl"
zst_path = EXTRACTED / f"{SUBREDDIT}_submissions.zst"
kw_csv = CONFIG / "bias_keywords.csv"
group_csv = CONFIG / "subreddit_groups.csv"

now = int(time.time())

# ---- 1) 예시 포스트(JSONL) 생성 (필드: id, subreddit, title, selftext, created_utc) ----
#  - 5개 문서: 매칭되는 것/안되는 것/에지 케이스 포함
posts = [
    {
        "id": "t1",
        "subreddit": SUBREDDIT,
        "title": "Stable Diffusion shows gender bias in doctor vs nurse images 😕",
        "selftext": "I prompted 'doctor' and they were mostly men, 'nurse' mostly women. Is SD biased?",
        "created_utc": now - 86400,
    },
    {
        "id": "t2",
        "subreddit": SUBREDDIT,
        "title": "Midjourney prompt tips",
        "selftext": "General tips for composition, not about bias.",
        "created_utc": now - 3600,
    },
    {
        "id": "t3",
        "subreddit": SUBREDDIT,
        "title": "AI image generation under-represents darker skin tones",
        "selftext": "Tried multiple prompts; results lacked diversity across races.",
        "created_utc": now - 7200,
    },
    {
        "id": "t4",
        "subreddit": SUBREDDIT,
        "title": "[removed]",
        "selftext": "placeholder",
        "created_utc": now - 10000,
    },
    {
        "id": "t5",
        "subreddit": SUBREDDIT,
        "title": "Face swap deepfake ethics",
        "selftext": "We need better safeguards for diffusion models to avoid harmful stereotypes.",
        "created_utc": now - 20000,
    },
]

with jsonl_path.open("w", encoding="utf-8") as f:
    for p in posts:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"✅ Wrote JSONL: {jsonl_path} ({len(posts)} lines)")

# ---- 2) 같은 내용 .zst로도 압축 (duckdb_data_processing의 decompressor 테스트용) ----
#     JSONL가 존재하면 너의 코드가 압축해제 생략하므로,
#     '압축해제 경로 테스트'를 원하면 jsonl 삭제 후 돌려봐도 됨.
dctx = zstd.ZstdCompressor(level=3, threads=0)
with zst_path.open("wb") as out, jsonl_path.open("rb") as src:
    out.write(dctx.compress(src.read()))
print(f"✅ Wrote ZST:    {zst_path}")

# ---- 3) bias_keywords.csv (최소 스키마: category,keyword 또는 bias_type,keyword) ----
#     카테고리/바이어스 타입은 자유롭게 쓰되, 파이프라인에서 word boundary 매칭이 잘 되도록 단어형태로 넣음.
kw_rows = [
    # category,bias_type,keyword 중 최소 'keyword'는 필수. 여기선 category,keyword 사용
    ("representation", "gender bias"),
    ("representation", "race"),
    ("representation", "diversity"),
    ("stereotype", "stereotype"),
    ("ethics", "deepfake"),
]
with kw_csv.open("w", encoding="utf-8") as f:
    f.write("category,keyword\n")
    for c, k in kw_rows:
        f.write(f"{c},{k}\n")
print(f"✅ Wrote keywords CSV: {kw_csv}")

# ---- 4) subreddit_groups.csv (스키마: subreddit,group) ----
#     파이프라인의 그룹별 정책을 타게 하려면 group을 아래 중 하나로 줘:
#     'technical' | 'creative_ai_communities' | 'critical_discussion' | 'general_reddit'
with group_csv.open("w", encoding="utf-8") as f:
    f.write("subreddit,group\n")
    f.write(f"{SUBREDDIT},general_reddit\n")  # 테스트 용으로 general_reddit로 둠
print(f"✅ Wrote groups CSV:    {group_csv}")

print("\n🎯 Test fixture is ready!")
print(f"   Subreddit: {SUBREDDIT}")
print(f"   Edit config.BASE_DIR to point to: {PROJECT_ROOT}")
print("   You can now run the pipeline like:")
print(
    f"   python pipeline.py --subreddit {SUBREDDIT} --skip 'LLM'  # (먼저 1~3단계 확인)"
)
print(f"   # LLM까지 돌릴 땐 모델/토큰 설정 후:")
print(f"   # python pipeline.py --subreddit {SUBREDDIT}")
