"""Fail if designated public artifacts contain prohibited Reddit identifiers.

This check is scoped to directories that are explicitly intended to hold
public fixtures or (later) governed aggregate outputs. It does not grep the
whole repository, so schema field names in source code and documentation are
not flagged.

Extend PUBLIC_AGGREGATE_DIRS when a sanitized public-results directory is
added. Aggregate dirs should not contain record-level source text.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Public synthetic fixtures (source text is allowed only because it is fictional).
SYNTHETIC_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic"

# Governed aggregate outputs — optional; skipped when the directory is absent.
# When added, records here must not include Reddit source-text fields.
PUBLIC_AGGREGATE_DIRS = [
    REPO_ROOT / "data" / "public",
    REPO_ROOT / "docs" / "public_results",
    REPO_ROOT / "artifacts" / "synthetic_demo",
]

PROHIBITED_FIELD_NAMES = {
    "permalink",
    "author",
    "username",
    "author_fullname",
    "author_flair_text",
}

# Reddit post/comment IDs are typically short base36 tokens. Synthetic IDs
# must use an unmistakable prefix instead.
SYNTHETIC_ID_RE = re.compile(r"^SYNTH-[A-Z]+-\d+$")
REDDIT_LIKE_ID_RE = re.compile(r"^[0-9a-z]{5,8}$")
REDDIT_URL_RE = re.compile(
    r"https?://(?:www\.)?reddit\.com/|https?://(?:www\.)?redd\.it/",
    re.IGNORECASE,
)
AGGREGATE_TEXT_FIELDS = {
    "clean_text",
    "clean_text_lc",
    "selftext",
    "title",
    "full_text",
    "body",
    "comments",
    "top_comments",
}

STRUCTURED_SUFFIXES = {".json", ".jsonl", ".csv"}


def _iter_structured_files(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in STRUCTURED_SUFFIXES:
            yield path


def _walk_records(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_records(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_records(item)


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                records.append(parsed)
            else:
                raise AssertionError(f"{path} line {line_no} is not a JSON object")
        return records
    parsed = json.loads(text)
    return list(_walk_records(parsed))


def _load_csv_records(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _field_names(record: dict[str, Any]) -> set[str]:
    return {str(key).strip().lower() for key in record.keys()}


def _id_values(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in record.items():
        if str(key).lower() in {"id", "post_id", "comment_id"} and value is not None:
            values.append(str(value).strip())
    return values


def _string_values(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for value in record.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
    return values


def test_synthetic_fixture_dir_exists():
    assert SYNTHETIC_FIXTURE_DIR.is_dir(), (
        "Synthetic fixture directory is missing; public examples must stay "
        "under tests/fixtures/synthetic/"
    )
    readme = SYNTHETIC_FIXTURE_DIR / "README.md"
    assert readme.is_file(), "Synthetic fixture README.md is required"


@pytest.mark.parametrize("path", list(_iter_structured_files(SYNTHETIC_FIXTURE_DIR)))
def test_synthetic_fixtures_have_no_prohibited_reddit_fields(path: Path):
    if path.suffix.lower() == ".csv":
        fieldnames, records = _load_csv_records(path)
        names = {name.strip().lower() for name in fieldnames}
    else:
        records = _load_json_records(path)
        names = set()
        for record in records:
            names |= _field_names(record)

    prohibited = names & PROHIBITED_FIELD_NAMES
    assert not prohibited, (
        f"{path.relative_to(REPO_ROOT)} contains prohibited field(s): "
        f"{sorted(prohibited)}"
    )

    for record in records:
        for identifier in _id_values(record):
            assert SYNTHETIC_ID_RE.match(identifier), (
                f"{path.relative_to(REPO_ROOT)} has non-synthetic id "
                f"(expected SYNTH-<LABEL>-<digits>)"
            )
            assert not REDDIT_LIKE_ID_RE.match(
                identifier
            ), f"{path.relative_to(REPO_ROOT)} id looks like a Reddit base36 id"
        for text in _string_values(record):
            assert not REDDIT_URL_RE.search(
                text
            ), f"{path.relative_to(REPO_ROOT)} contains a Reddit permalink-like URL"


def test_public_aggregate_dirs_have_no_record_level_source_text():
    present = [directory for directory in PUBLIC_AGGREGATE_DIRS if directory.is_dir()]
    if not present:
        pytest.skip(
            "No public aggregate directory yet; when adding governed outputs, "
            "place them under data/public/ or docs/public_results/ so this "
            "check can forbid permalink/author/username and source-text fields."
        )

    for directory in present:
        files = list(_iter_structured_files(directory))
        assert files, f"{directory} exists but contains no structured public files"
        for path in files:
            if path.suffix.lower() == ".csv":
                fieldnames, records = _load_csv_records(path)
                names = {name.strip().lower() for name in fieldnames}
            else:
                records = _load_json_records(path)
                names = set()
                for record in records:
                    names |= _field_names(record)

            prohibited = names & (PROHIBITED_FIELD_NAMES | AGGREGATE_TEXT_FIELDS)
            assert not prohibited, (
                f"{path.relative_to(REPO_ROOT)} looks like record-level Reddit "
                f"content in a public aggregate directory: {sorted(prohibited)}"
            )
            for record in records:
                for identifier in _id_values(record):
                    assert not REDDIT_LIKE_ID_RE.match(
                        identifier
                    ), f"{path.relative_to(REPO_ROOT)} contains a Reddit-like id"
                for text in _string_values(record):
                    assert not REDDIT_URL_RE.search(text)
