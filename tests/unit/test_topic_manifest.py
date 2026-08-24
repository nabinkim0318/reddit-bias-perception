"""Topic-run provenance records seed and configuration, not source text."""

from analysis.topic_manifest import build_topic_run_manifest


def test_topic_run_manifest_records_seed_and_excludes_document_rows():
    manifest = build_topic_run_manifest(
        run_id="abc123",
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        code_sha="deadbeef",
        input_filename="local_posts.csv",
        input_sha256="a" * 64,
        input_record_count=10,
        seed=23,
        bertopic_version="0.17.0",
        embedding_model="all-MiniLM-L6-v2",
        embedding_model_revision=None,
        umap_configuration={"random_state": 23, "n_neighbors": 15},
        clustering_configuration={"algorithm": "hdbscan"},
        vectorizer_configuration_hash="b" * 64,
        nr_topics="auto",
        nr_topics_mode="exploratory_auto",
        discovered_inlier_topic_count=4,
        outlier_count=2,
        outlier_rate=0.2,
        topic_assignment_checksum="c" * 64,
        topic_summary_checksum="d" * 64,
        overall_status="success",
        calculate_probabilities=True,
        language="english",
    )
    assert manifest["seed"] == 23
    assert manifest["nr_topics"] == "auto"
    assert manifest["umap_configuration"]["random_state"] == 23
    assert manifest["embedding_model_revision"] is None
    blob = str(manifest)
    assert "clean_text" not in blob
    assert "permalink" not in blob
    assert "assignments" not in manifest
