"""Synthetic end-to-end human-validation workflow. Fully offline."""

from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic" / "validation"
MODEL_CSV = FIXTURE_DIR / "model_annotations.csv"
LABELS_JSON = FIXTURE_DIR / "human_labels.json"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _labels_by_original_id() -> dict[str, dict[str, str]]:
    return json.loads(LABELS_JSON.read_text(encoding="utf-8"))


def _build_human_files(index_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    labels = _labels_by_original_id()
    with index_path.open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    rows_a: list[dict[str, str]] = []
    rows_b: list[dict[str, str]] = []
    adjudication: list[dict[str, str]] = []
    for row in index_rows:
        original = row["original_record_id"]
        mapped = labels[original]
        rows_a.append(
            {
                "task_id": row["task_id"],
                "human_label": mapped["a"],
                "notes": "",
            }
        )
        rows_b.append(
            {
                "task_id": row["task_id"],
                "human_label": mapped["b"],
                "notes": "",
            }
        )
        if mapped.get("adjudication_status"):
            adjudication.append(
                {
                    "task_id": row["task_id"],
                    "annotator_a_label": mapped["a"],
                    "annotator_b_label": mapped["b"],
                    "adjudicated_label": mapped.get("adjudicated_label", ""),
                    "adjudication_status": mapped["adjudication_status"],
                }
            )
    path_a = output_dir / "annotator_a.csv"
    path_b = output_dir / "annotator_b.csv"
    path_adj = output_dir / "adjudication.csv"
    _write_csv(path_a, rows_a, ["task_id", "human_label", "notes"])
    _write_csv(path_b, rows_b, ["task_id", "human_label", "notes"])
    _write_csv(
        path_adj,
        adjudication,
        [
            "task_id",
            "annotator_a_label",
            "annotator_b_label",
            "adjudicated_label",
            "adjudication_status",
        ],
    )
    return path_a, path_b, path_adj


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=_env(),
        check=False,
        capture_output=True,
        text=True,
    )


def test_synthetic_validation_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("network disabled in validation e2e test")
        ),
    )
    work = tmp_path / "validation_run"
    work.mkdir()
    sample = _run_cli(
        [
            "-m",
            "validation.sample",
            "--input",
            str(MODEL_CSV),
            "--output-dir",
            str(work),
            "--sample-size",
            "20",
            "--seed",
            "42",
            "--failure-sample-size",
            "5",
        ]
    )
    assert sample.returncode == 0, sample.stderr
    index_path = work / "sampling_index.csv"
    tasks_path = work / "annotation_tasks.csv"
    assert index_path.is_file()
    with tasks_path.open(newline="", encoding="utf-8") as handle:
        task_fields = list(csv.DictReader(handle).fieldnames or [])
    assert task_fields == [
        "task_id",
        "text_to_annotate",
        "human_label",
        "notes",
    ]
    path_a, path_b, path_adj = _build_human_files(index_path, work)
    report_path = work / "validation_report.json"
    evaluate = _run_cli(
        [
            "-m",
            "validation.evaluate",
            "--sampling-index",
            str(index_path),
            "--annotations-a",
            str(path_a),
            "--annotations-b",
            str(path_b),
            "--adjudication",
            str(path_adj),
            "--sampling-config",
            str(work / "sampling_config.json"),
            "--source-artifact",
            str(MODEL_CSV),
            "--output",
            str(report_path),
            "--bootstrap-iterations",
            "50",
            "--bootstrap-seed",
            "42",
        ]
    )
    assert evaluate.returncode == 0, evaluate.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))

    counts = report["counts"]
    assert counts["sampled"] == 11
    assert counts["double_annotated"] == 11
    assert counts["agreed_yes"] == 3
    assert counts["agreed_no"] == 4
    assert counts["human_disagreements"] == 1
    assert counts["uncertain"] == 1
    assert counts["insufficient_context"] == 2
    assert counts["adjudicated"] == 1
    assert counts["unresolved"] == 3
    assert counts["model_success"] == 9
    assert counts["model_parse_error"] == 1
    assert counts["model_model_error"] == 1
    assert counts["binary_evaluable"] == 7
    assert (
        counts["binary_evaluable"]
        == counts["sampled"]
        - report["exclusions"]["model_execution_failure"]
        - report["exclusions"]["unresolved_human_reference"]
    )

    matrix = report["confusion_matrix"]
    assert matrix == {
        "true_positive": 3,
        "true_negative": 2,
        "false_positive": 1,
        "false_negative": 1,
    }
    metrics = report["metrics"]
    assert metrics["precision_yes"] == pytest.approx(0.75)
    assert metrics["recall_yes"] == pytest.approx(0.75)
    assert metrics["f1_yes"] == pytest.approx(0.75)
    assert metrics["accuracy"] == pytest.approx(5 / 7)
    assert report["exclusions"]["model_execution_failure"] == 2
    assert matrix["true_negative"] == 2
    assert matrix["false_negative"] == 1

    agreement = report["inter_annotator_agreement"]
    assert agreement["disagreement_count"] == 1
    assert agreement["percent_agreement"] == pytest.approx(10 / 11)
    assert agreement["cohens_kappa"] is not None
    assert report["claims"]["human_validation_study_completed"] is False
    assert report["sampling_config_hash"]
    assert report["provenance"]["source_artifact_sha256"]
    assert report["provenance"]["annotations_a_sha256"]
    assert "code_sha" in report
    assert "clean_text" not in report
    assert "text_to_annotate" not in report
    blob = json.dumps(report)
    assert "SYNTH-VAL-" not in blob
    assert "Pixora" not in blob
    assert "reddit.com" not in blob.lower()

    sample_again = _run_cli(
        [
            "-m",
            "validation.sample",
            "--input",
            str(MODEL_CSV),
            "--output-dir",
            str(tmp_path / "second"),
            "--sample-size",
            "20",
            "--seed",
            "42",
            "--failure-sample-size",
            "5",
        ]
    )
    assert sample_again.returncode == 0, sample_again.stderr
    first_index = index_path.read_text(encoding="utf-8")
    second_index = (tmp_path / "second" / "sampling_index.csv").read_text(
        encoding="utf-8"
    )
    assert first_index == second_index
