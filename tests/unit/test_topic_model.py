# tests/unit/test_topic_model.py

"""import pandas as pd

from analysis.bertopic_model import run_bertopic_model


def test_run_bertopic_model_basic():
    # Minimal input
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "clean_text": ["AI bias exists.", "CUDA error", None],
            "full_text": [
                "AI bias exists in portraits.",
                "CUDA out of memory",
                "Empty",
            ],
        }
    )
    model, topic_info, doc_info = run_bertopic_model(df)

    assert not topic_info.empty
    assert len(doc_info) == 3
    assert "post_id" in doc_info.columns
    assert "raw_text" in doc_info.columns
"""
