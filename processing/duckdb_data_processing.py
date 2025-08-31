# processing/duckdb_data_processing.py
import json
import logging
import os
import shutil
import subprocess
from itertools import islice
from pathlib import Path

import pandas as pd
import zstandard as zstd

from config.config import BASE_DIR, ZSTD_DECOMPRESS_THREADS

BASE_DIR = Path(BASE_DIR)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_data_paths(subreddit: str) -> dict[str, Path]:
    extracted_dir = Path(BASE_DIR) / "extracted"
    return {
        "compressed_path": extracted_dir / f"{subreddit}_submissions.zst",
        "extracted_path": extracted_dir / f"{subreddit}_posts.jsonl",
    }


def ensure_parent_dir_exists(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _zstd_cli_available() -> bool:
    return shutil.which("zstd") is not None


def decompress_zstd(
    paths: dict[str, Path],
    prefer_cli: bool = False,
    threads: int = ZSTD_DECOMPRESS_THREADS,
) -> None:
    src, dst = paths["compressed_path"], paths["extracted_path"]
    ensure_parent_dir_exists(dst)
    logger.info(f"📦 Decompressing {src} → {dst} ...")

    try:
        use_cli = prefer_cli and _zstd_cli_available()
        if use_cli:
            logger.info(f"⚡ Using zstd CLI (threads={threads})...")
            subprocess.run(
                [
                    "zstd",
                    "-d",
                    "--long=31",
                    f"--threads={threads}",
                    str(src),
                    "-o",
                    str(dst),
                ],
                check=True,
            )
        else:
            if prefer_cli and not use_cli:
                logger.warning(
                    "⚠️ zstd CLI not found. Falling back to Python zstandard."
                )
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with src.open("rb") as compressed, dst.open("wb") as out_file:
                dctx.copy_stream(compressed, out_file)

        # simple integrity check: avoid empty file
        if not dst.exists() or dst.stat().st_size == 0:
            raise RuntimeError("Output file is empty after decompression")

        logger.info(
            f"✅ Decompression complete. Exists={dst.exists()} Size={dst.stat().st_size:,} bytes"
        )
    except Exception as e:
        logger.error(f"❌ Decompression failed: {e}")
        raise RuntimeError(f"Decompression failed for {src}") from e


def extract_only(paths):
    extracted_path = paths["extracted_path"]
    raw_path = paths["compressed_path"]

    # 🔧 Ensure parent directory exists
    ensure_parent_dir_exists(extracted_path)

    # Decompress
    try:
        decompress_zstd(paths, prefer_cli=True)
    except Exception as e:
        logger.error(f"❌ Decompression failed: {e}")
        return

    # Delete compressed file
    try:
        os.remove(raw_path)
        logger.info(f"🧹 Deleted compressed file: {raw_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to delete {raw_path}: {e}")

    # Preview
    try:
        with open(
            paths["extracted_path"], "r", encoding="utf-8", errors="replace"
        ) as f:
            lines = list(islice(f, 10))
        logger.info(f"✅ Previewed {len(lines)} lines")
    except Exception as e:
        logger.warning(f"⚠️ Failed to preview file: {e}")


def load_and_preview_jsonl(paths: dict[str, Path], num_lines: int = 10) -> pd.DataFrame:
    dst = paths["extracted_path"]
    if not dst.exists():
        decompress_zstd(paths, prefer_cli=True)

    records = []
    with dst.open("r", encoding="utf-8", errors="replace") as f:
        for line in islice(f, num_lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # avoid broken lines
                continue

    df = pd.DataFrame(records)

    # created_utc → datetime
    if "created_utc" in df.columns:
        df["created_dt"] = pd.to_datetime(
            pd.to_numeric(df["created_utc"], errors="coerce"), unit="s"
        )

    # safe column preview
    preferred_cols = ["id", "subreddit", "created_dt", "title", "selftext"]
    show_cols = [c for c in preferred_cols if c in df.columns]
    if show_cols:
        logger.info("📄 Preview of JSONL data:")
        logger.info(df[show_cols].head().to_string())
    else:
        logger.info(
            "📄 Preview available but preferred columns not present. Columns: %s",
            list(df.columns),
        )

    logger.info("🧩 Columns: %s", list(df.columns))
    return df


def list_extracted_files():
    """return the list of files in data/extracted directory"""
    extracted_dir = Path(BASE_DIR) / "extracted"

    if not extracted_dir.exists():
        logger.error("❌ data/extracted directory does not exist")
        return []

    return [p.name for p in extracted_dir.iterdir() if p.is_file()]


def cli_main(process_all: bool = False, subreddit: str | None = None) -> None:
    if process_all:
        files = list_extracted_files()
        for file in files:
            if not file.endswith(".zst"):
                logger.info(f"Skipping {file} because it is not a zst file")
                continue

            sub = file.replace("_submissions.zst", "")
            logger.info(f"\n🚩 Processing {sub} ...")
            paths = get_data_paths(sub)
            logger.info(
                f"✅ {paths['extracted_path']} exists: {paths['extracted_path'].exists()}"
            )

            # decompress if not exists
            if not paths["extracted_path"].exists():
                decompress_zstd(paths, prefer_cli=True)

            # load and preview
            df = load_and_preview_jsonl(paths, num_lines=10)
            logger.info(f"✅ Previewed {len(df)} rows for {sub}")
    else:
        if subreddit is None:
            raise ValueError("subreddit must be provided if process_all=False")
        paths = get_data_paths(subreddit)
        if not paths["extracted_path"].exists():
            decompress_zstd(paths, prefer_cli=True)
        df = load_and_preview_jsonl(paths, num_lines=10)
        logger.info(f"✅ Previewed {len(df)} rows for {subreddit}")


def jsonl_to_csv_full(
    jsonl_path: Path, out_csv: Path, chunksize: int = 100_000
) -> Path:
    """
    Memory-efficient JSONL to CSV conversion.
    Use pandas read_json(lines=True, chunksize=...)
    pandas read_json(lines=True, chunksize=...) use.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    wrote_header = False
    for chunk in pd.read_json(jsonl_path, lines=True, chunksize=chunksize):
        if "created_utc" in chunk.columns and "created_dt" not in chunk.columns:
            chunk["created_dt"] = pd.to_datetime(
                pd.to_numeric(chunk["created_utc"], errors="coerce"), unit="s"
            )
        chunk.to_csv(
            tmp,
            index=False,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
        )
        wrote_header = True
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"CSV write failed or empty: {tmp}")
    tmp.replace(out_csv)
    return out_csv


def main(subreddit: str):
    paths = get_data_paths(subreddit)
    # 1) JSONL keep original (reuse if exists)
    if not paths["extracted_path"].exists():
        decompress_zstd(paths, prefer_cli=True)

    # 2) save sample preview (optional)
    out_dir = Path(BASE_DIR) / "filtered"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = out_dir / f"{subreddit}_sample_preview.csv"
    preview_df = load_and_preview_jsonl(paths, num_lines=10)
    preview_df.to_csv(sample_csv, index=False)

    # 3) save full CSV
    full_csv = out_dir / f"{subreddit}_duckdb_processed.csv"
    jsonl_to_csv_full(paths["extracted_path"], full_csv, chunksize=100_000)

    logger.info(f"✅ JSONL kept: {paths['extracted_path']}")
    logger.info(f"✅ Preview saved: {sample_csv}")
    logger.info(f"✅ Full CSV saved: {full_csv}")
    return full_csv


if __name__ == "__main__":
    dummy_subreddit = "StableDiffusion"
    logging.basicConfig(level=logging.INFO)
    # CLI action if needed
    cli_main(process_all=False, subreddit=dummy_subreddit)
