# torrent/duckdb_processing.py
import json
import os
import subprocess
import logging
from tqdm import tqdm

import pandas as pd
import zstandard as zstd
from config.config import ZSTD_DECOMPRESS_THREADS
from config.config import BASE_DIR

logging.basicConfig(level=logging.INFO)

def ensure_parent_dir_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def decompress_zstd(file_zst, extracted_path, prefer_cli=False, threads=ZSTD_DECOMPRESS_THREADS):
    logging.info(f"📦 Decompressing {file_zst} to {extracted_path}...")
    try:
        if prefer_cli:
            logging.info("⚡ Using zstd CLI for decompression...")
            subprocess.run(
                [
                    "zstd",
                    "-d",
                    "--long=31",
                    f"--threads={threads}",
                    file_zst,
                    "-o",
                    extracted_path,
                ],
                check=True,
            )
        else:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with open(file_zst, "rb") as compressed, open(
                extracted_path, "wb"
            ) as out_file:
                dctx.copy_stream(compressed, out_file)
        logging.info(f"✅ Decompression complete")
    except Exception as e:
        logging.error(f"❌ Decompression failed: {e}")
        raise RuntimeError(f"Decompression failed for {file_zst}") from e


def extract_only(paths):
    extracted_path = paths["extracted_path"]
    raw_path = paths["raw_path"]

    # 🔧 Ensure parent directory exists
    ensure_parent_dir_exists(extracted_path)

    # Decompress
    try:
        decompress_zstd(raw_path, extracted_path, prefer_cli=True)
    except Exception as e:
        logging.error(f"❌ Decompression failed: {e}")
        return

    # Delete compressed file
    try:
        os.remove(raw_path)
        logging.info(f"🧹 Deleted compressed file: {raw_path}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to delete {raw_path}: {e}")

    # Preview
    try:
        with open(extracted_path, "r") as f:
            lines = [next(f) for _ in range(10)]
        logging.info(f"✅ Previewed {len(lines)} lines")
    except Exception as e:
        logging.warning(f"⚠️ Failed to preview file: {e}")

    
def load_and_preview_jsonl(subreddit: str, num_lines: int = 10):
    file_path = f"{BASE_DIR}/extracted/{subreddit}.jsonl.zst"
    extracted_path = f"{BASE_DIR}/extracted/{subreddit}.jsonl"

    decompress_zstd(file_path, extracted_path, prefer_cli=True)

    all_lines = []
    with open(file_path, "r") as f:
        for line in tqdm(f, desc="Loading JSONL file", total=num_lines):
            all_lines.append(json.loads(line))

    df = pd.DataFrame(all_lines)

    # created_utc → datetime
    if "created_utc" in df.columns:
        df["created_dt"] = pd.to_datetime(
            pd.to_numeric(df["created_utc"], errors="coerce"), unit="s"
        )

    # Add columns for missing values
    df["title_missing"] = df["title"].isin(["[deleted]", "[removed]", "", None, "null"])
    df["selftext_missing"] = df["selftext"].isin(
        ["[deleted]", "[removed]", "", None, "null"]
    )

    # Add columns for length
    df["title_len"] = df["title"].fillna("").apply(len)
    df["selftext_len"] = df["selftext"].fillna("").apply(len)

    # Print summary information
    logging.info("📄 Preview of JSONL data:")
    logging.info(df[["id", "subreddit", "created_dt", "title", "selftext"]].head())

    logging.info("\n🧾 Missing content stats:")
    logging.info("Missing titles:", df["title_missing"].sum())
    logging.info("Missing selftexts:", df["selftext_missing"].sum())

    logging.info("\n📏 Length stats (non-missing only):")
    logging.info(df[["title_len", "selftext_len"]].describe())

    logging.info("\n🧩 Columns:")
    logging.info(df.columns.tolist())

    return df


def main(subreddit):
    df_preview = load_and_preview_jsonl(subreddit)
    logging.info(df_preview.head())


if __name__ == "__main__":
    main("technology")
