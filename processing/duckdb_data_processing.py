# processing/duckdb_data_processing.py
import json
import logging
import os
import subprocess

import pandas as pd
import zstandard as zstd
from tqdm import tqdm

from config.config import BASE_DIR, ZSTD_DECOMPRESS_THREADS

logging.basicConfig(level=logging.INFO)



def get_data_paths(subreddit: str):
    return {
        "compressed_path": f"{BASE_DIR}/extracted/{subreddit}_submissions.zst",
        "extracted_path": f"{BASE_DIR}/extracted/{subreddit}_posts.jsonl",
    }

def ensure_parent_dir_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def decompress_zstd(
    paths: dict[str, str], prefer_cli=False, threads=ZSTD_DECOMPRESS_THREADS
):
    logging.info(f"📦 Decompressing {paths['compressed_path']} to {paths['extracted_path']}...")
    try:
        if prefer_cli:
            logging.info("⚡ Using zstd CLI for decompression...")
            subprocess.run(
                [
                    "zstd",
                    "-d",
                    "--long=31",
                    f"--threads={threads}",
                    paths["compressed_path"],
                    "-o",
                    paths["extracted_path"],
                ],
                check=True,
            )
        else:
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with open(paths["compressed_path"], "rb") as compressed, open(
                paths["extracted_path"], "wb"
            ) as out_file:
                dctx.copy_stream(compressed, out_file)
        logging.info(f"✅ Decompression complete")
    except Exception as e:
        logging.error(f"❌ Decompression failed: {e}")
        raise RuntimeError(f"Decompression failed for {paths['compressed_path']}") from e


def extract_only(paths):
    extracted_path = paths["extracted_path"]
    raw_path = paths["raw_path"]

    # 🔧 Ensure parent directory exists
    ensure_parent_dir_exists(extracted_path)

    # Decompress
    try:
        decompress_zstd(paths, prefer_cli=True)
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
        with open(paths["extracted_path"], "r") as f:
            lines = [next(f) for _ in range(10)]
        logging.info(f"✅ Previewed {len(lines)} lines")
    except Exception as e:
        logging.warning(f"⚠️ Failed to preview file: {e}")


def load_and_preview_jsonl(paths: dict[str, str], num_lines: int = 10):
    decompress_zstd(paths, prefer_cli=True)

    all_lines = []
    with open(paths["extracted_path"], "r") as f:
        for line in tqdm(f, desc="Loading JSONL file", total=num_lines):
            all_lines.append(json.loads(line))

    df = pd.DataFrame(all_lines)

    # created_utc → datetime
    if "created_utc" in df.columns:
        df["created_dt"] = pd.to_datetime(
            pd.to_numeric(df["created_utc"], errors="coerce"), unit="s"
        )

    # Print summary information
    logging.info("📄 Preview of JSONL data:")
    logging.info(df[["id", "subreddit", "created_dt", "title", "selftext"]].head())

    logging.info("\n🧩 Columns:")
    logging.info(df.columns.tolist())

    return df


def main(subreddit: str):
    paths = get_data_paths(subreddit)
    df_preview = load_and_preview_jsonl(paths)
    logging.info(df_preview.head())


if __name__ == "__main__":
    dummy_subreddit = "technology"
    main(dummy_subreddit)
