"""Deterministic fake annotator for the offline synthetic demo.

This module is test / pipeline-validation infrastructure. It is not a
scientific classifier and must not be treated as a research model.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from processing.llm_annotation import classify_batch

SYNTHETIC_ANNOTATOR_ID = "synthetic-demo-annotator/v1"
ANNOTATION_MODE = "synthetic_demo"

# Explicit raw strings for the public synthetic fixture IDs.
# Values are fake model replies, not research labels.
FIXTURE_RAW_OUTPUTS: Dict[str, str] = {
    "SYNTH-POST-0001": "yes",
    "SYNTH-POST-0007": "no",
    "SYNTH-POST-0008": "this is not a yes-or-no contract label",
}

_FALLBACK_RAW_OUTPUTS = (
    "yes",
    "no",
    "not a contract label",
)


class SyntheticDemoAnnotator:
    """Injected stand-in for a real LLM. Never loads a model."""

    identifier = SYNTHETIC_ANNOTATOR_ID

    def __init__(self, script: Optional[Mapping[str, str]] = None) -> None:
        self._script = dict(script) if script is not None else dict(FIXTURE_RAW_OUTPUTS)
        self._bound_ids: List[str] = []

    def bind_ids(self, ids: Sequence[str]) -> None:
        self._bound_ids = [str(item) for item in ids]

    def raw_output_for(self, post_id: str, index: int) -> str:
        if post_id in self._script:
            return self._script[post_id]
        return _FALLBACK_RAW_OUTPUTS[index % len(_FALLBACK_RAW_OUTPUTS)]

    def generate(self, batch_texts: List[str]) -> List[str]:
        if len(batch_texts) != len(self._bound_ids):
            raise RuntimeError("synthetic annotator IDs are not bound to this batch")
        return [
            self.raw_output_for(post_id, index)
            for index, post_id in enumerate(self._bound_ids)
        ]

    def annotate_records(self, records: Sequence[Mapping[str, object]]) -> list[dict]:
        if not records:
            return []
        texts = [str(row.get("clean_text") or "") for row in records]
        ids = [str(row.get("id") or "") for row in records]
        subreddits = [str(row.get("subreddit") or "") for row in records]
        self.bind_ids(ids)
        return classify_batch(texts, ids, subreddits, self.generate)
