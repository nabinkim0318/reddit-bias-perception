"""End-to-end and failure tests for the canonical synthetic offline pipeline."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from processing.manifest import (
    AGGREGATE_FILENAME,
    MANIFEST_FILENAME,
    OVERALL_FAILURE,
    OVERALL_REUSED,
    OVERALL_SUCCESS,
    read_json,
)
from processing.synthetic_annotator import SYNTHETIC_ANNOTATOR_ID
from processing.synthetic_pipeline import SyntheticPipelineError, run_synthetic_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_INPUT = REPO_ROOT / "tests" / "fixtures" / "synthetic" / "posts.json"
FIXTURE_GROUPS = REPO_ROOT / "tests" / "fixtures" / "synthetic" / "subreddit_groups.csv"
CANONICAL_CMD = [
    sys.executable,
    "-m",
    "processing.run_pipeline",
    "--synthetic",
]

PROHIBITED_FIELDS = {
    "permalink",
    "author",
    "username",
    "author_fullname",
    "author_flair_text",
    "clean_text",
    "clean_text_lc",
    "selftext",
    "title",
    "full_text",
    "body",
    "comments",
    "top_comments",
}


def _env_with_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def _run_pipeline(tmp_path: Path, input_path: Path = FIXTURE_INPUT, **kwargs):
    return run_synthetic_pipeline(
        input_path,
        tmp_path / "run",
        groups_path=FIXTURE_GROUPS,
        **kwargs,
    )


def test_end_to_end_synthetic_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("network disabled in synthetic pipeline test")
        ),
    )
    result = _run_pipeline(tmp_path)
    output_dir = tmp_path / "run"
    aggregate = result["aggregate"]
    manifest = result["manifest"]

    assert result["reused"] is False
    assert (output_dir / AGGREGATE_FILENAME).is_file()
    assert (output_dir / MANIFEST_FILENAME).is_file()
    assert aggregate["artifact_type"] == "synthetic_demo_aggregate"
    assert aggregate["classification"] == "synthetic"
    assert "research finding" in aggregate["disclaimer"].lower()
    assert aggregate["input_records"] == 9
    assert aggregate["yes"] == 1
    assert aggregate["no"] == 1
    assert aggregate["parse_error"] == 1
    assert aggregate["model_error"] == 0
    assert aggregate["successful_annotations"] == 2
    assert aggregate["total_records"] == 3
    assert aggregate["yes"] + aggregate["no"] + aggregate["parse_error"] == 3

    assert manifest["overall_status"] == OVERALL_SUCCESS
    assert manifest["input"]["checksum_sha256"]
    assert manifest["input"]["classification"] == "synthetic"
    assert manifest["input"]["record_count"] == 9
    assert manifest["config_hash"]
    assert manifest["code_sha"]
    assert manifest["annotation_mode"] == "synthetic_demo"
    assert manifest["annotator_id"] == SYNTHETIC_ANNOTATOR_ID
    assert manifest["config"]["annotator_id"] == SYNTHETIC_ANNOTATOR_ID
    assert manifest["config"]["keyword_policy_hash"]
    assert manifest["config"]["prompt_hash"]
    assert manifest["counts"]["deduplicated"] == 1
    assert manifest["counts"]["invalid_content"] == 1
    assert manifest["counts"]["keyword_kept"] == 3
    assert manifest["annotation_status_counts"]["success_yes"] == 1
    assert manifest["annotation_status_counts"]["success_no"] == 1
    assert manifest["annotation_status_counts"]["parse_error"] == 1
    assert manifest["artifacts"]["aggregate"]["sha256"]
    stage_names = [stage["name"] for stage in manifest["stages"]]
    assert stage_names == [
        "load_input",
        "validate_schema",
        "preprocess",
        "deduplicate",
        "keyword_filter",
        "annotate",
        "aggregate",
        "manifest",
    ]
    assert all(stage["status"] == "success" for stage in manifest["stages"])

    public_files = [path for path in output_dir.rglob("*") if path.is_file()]
    assert {path.name for path in public_files} == {
        AGGREGATE_FILENAME,
        MANIFEST_FILENAME,
    }
    for path in public_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _assert_no_prohibited_fields(payload)


def test_cli_canonical_command_succeeds(tmp_path):
    output_dir = tmp_path / "cli-out"
    completed = subprocess.run(
        [
            *CANONICAL_CMD,
            "--input",
            str(FIXTURE_INPUT),
            "--output-dir",
            str(output_dir),
            "--groups",
            str(FIXTURE_GROUPS),
        ],
        cwd=REPO_ROOT,
        env=_env_with_pythonpath(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / AGGREGATE_FILENAME).is_file()
    assert (output_dir / MANIFEST_FILENAME).is_file()


def test_cli_missing_input(tmp_path):
    output_dir = tmp_path / "missing-out"
    completed = subprocess.run(
        [
            *CANONICAL_CMD,
            "--input",
            str(tmp_path / "does-not-exist.json"),
            "--output-dir",
            str(output_dir),
            "--groups",
            str(FIXTURE_GROUPS),
        ],
        cwd=REPO_ROOT,
        env=_env_with_pythonpath(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        assert read_json(manifest_path)["overall_status"] != OVERALL_SUCCESS


def test_missing_input_raises_without_success_manifest(tmp_path):
    with pytest.raises(SyntheticPipelineError) as exc_info:
        run_synthetic_pipeline(
            tmp_path / "missing.json",
            tmp_path / "out",
            groups_path=FIXTURE_GROUPS,
        )
    assert exc_info.value.code == "missing_input"
    assert exc_info.value.stage == "load_input"
    assert not (tmp_path / "out" / MANIFEST_FILENAME).exists()


def test_invalid_schema_fails_without_silent_drop(tmp_path):
    bad_input = tmp_path / "bad.json"
    bad_input.write_text(
        json.dumps(
            [
                {
                    "id": "SYNTH-POST-9999",
                    "title": "missing required fields",
                    "selftext": "still missing subreddit and created_utc",
                }
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "bad-out"
    with pytest.raises(SyntheticPipelineError) as exc_info:
        run_synthetic_pipeline(
            bad_input, output_dir, groups_path=FIXTURE_GROUPS, force=True
        )
    assert exc_info.value.code == "invalid_schema"
    assert exc_info.value.stage == "validate_schema"
    manifest = read_json(output_dir / MANIFEST_FILENAME)
    assert manifest["overall_status"] == OVERALL_FAILURE
    assert manifest["error"]["code"] == "invalid_schema"
    failed_stages = [s for s in manifest["stages"] if s["status"] == "failure"]
    assert failed_stages
    assert not (output_dir / AGGREGATE_FILENAME).exists()


def test_stale_output_is_not_reused(tmp_path):
    input_path = tmp_path / "posts.json"
    input_path.write_text(FIXTURE_INPUT.read_text(encoding="utf-8"), encoding="utf-8")
    output_dir = tmp_path / "stale"
    first = run_synthetic_pipeline(
        input_path, output_dir, groups_path=FIXTURE_GROUPS, force=True
    )
    first_checksum = first["manifest"]["input"]["checksum_sha256"]
    first_total = first["aggregate"]["total_records"]

    records = json.loads(input_path.read_text(encoding="utf-8"))
    for record in records:
        if record["id"] == "SYNTH-POST-0005":
            record["selftext"] = (
                "The lantern festival poster printer uses artificial intelligence "
                "and draws every dancer with a gender stereotype."
            )
    input_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    second = run_synthetic_pipeline(
        input_path, output_dir, groups_path=FIXTURE_GROUPS, force=False
    )
    assert second["reused"] is False
    assert second["manifest"]["cache_reused"] is False
    assert second["manifest"]["overall_status"] == OVERALL_SUCCESS
    assert second["manifest"]["input"]["checksum_sha256"] != first_checksum
    assert second["aggregate"]["total_records"] > first_total


def test_tampered_aggregate_is_not_reused(tmp_path):
    first = _run_pipeline(tmp_path, force=True)
    output_dir = tmp_path / "run"
    aggregate_path = output_dir / AGGREGATE_FILENAME
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    payload["yes"] = 999
    aggregate_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    second = _run_pipeline(tmp_path, force=False)
    assert second["reused"] is False
    assert second["manifest"]["cache_reused"] is False
    assert second["aggregate"]["yes"] == first["aggregate"]["yes"]
    assert second["aggregate"]["yes"] != 999


def test_matching_cache_is_reused(tmp_path):
    first = _run_pipeline(tmp_path, force=True)
    second = _run_pipeline(tmp_path, force=False)
    assert first["reused"] is False
    assert second["reused"] is True
    assert second["manifest"]["cache_reused"] is True
    assert second["manifest"]["overall_status"] == OVERALL_REUSED
    assert (
        second["manifest"]["input"]["checksum_sha256"]
        == first["manifest"]["input"]["checksum_sha256"]
    )
    assert second["aggregate"]["total_records"] == first["aggregate"]["total_records"]


def test_annotation_failure_is_not_a_negative_label(tmp_path):
    result = _run_pipeline(tmp_path, force=True)
    counts = result["manifest"]["annotation_status_counts"]
    aggregate = result["aggregate"]
    assert counts["parse_error"] >= 1
    assert counts["success_no"] == aggregate["no"] == 1
    assert counts["success_yes"] == aggregate["yes"] == 1
    assert aggregate["unclassified"] == counts["parse_error"]
    assert (
        counts["success_yes"] + counts["success_no"]
        == counts["total"] - counts["unclassified"]
    )


def test_synthetic_pipeline_import_does_not_load_heavy_nlp():
    script = (
        "import sys; "
        "import processing.synthetic_pipeline; "
        "import processing.run_pipeline; "
        "banned = ('transformers', 'bertopic', 'sentence_transformers', 'optimum'); "
        "loaded = [name for name in banned if name in sys.modules]; "
        "assert not loaded, loaded"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=_env_with_pythonpath(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _assert_no_prohibited_fields(obj) -> None:
    if isinstance(obj, dict):
        names = {str(key).strip().lower() for key in obj}
        overlap = names & PROHIBITED_FIELDS
        assert not overlap, overlap
        for value in obj.values():
            _assert_no_prohibited_fields(value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_prohibited_fields(item)
