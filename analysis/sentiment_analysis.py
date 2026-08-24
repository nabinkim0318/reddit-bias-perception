"""Dual sentiment analysis: GoEmotions (multi-label) and VADER (rule-based).

Tokenizer, transformer, and VADER objects are created on first use, not at
import time. The GoEmotions model ID and optional revision are explicit
configuration. A missing revision is recorded as unset and is not an
immutable pin.

This module does not validate GoEmotions for Reddit, calibrate scores, or
change 512-token truncation. Predictions remain exploratory local tooling.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from config.config import (
    EMOTION_MODEL,
    FINAL_ANALYSIS_INPUT,
    SENTIMENT_OUTPUT,
    VADER_PLOT_PATH,
)

logger = logging.getLogger(__name__)

DEFAULT_GOEMOTIONS_MODEL_ID = EMOTION_MODEL
DEFAULT_GOEMOTIONS_MODEL_REVISION: Optional[str] = None
DEFAULT_GOEMOTIONS_LOCAL_FILES_ONLY = True
DEFAULT_GOEMOTIONS_MAX_LENGTH = 512
# Backward-compatible alias for the explicit default model ID.
MODEL_ID = DEFAULT_GOEMOTIONS_MODEL_ID

_goemotions_runtime: Optional["GoEmotionsRuntime"] = None
_vader_analyzer: Any = None


@dataclass(frozen=True)
class GoEmotionsRuntime:
    model_id: str
    model_revision: Optional[str]
    tokenizer: Any
    model: Any
    device: Any
    id2label: Any
    local_files_only: bool

    def provenance(self) -> dict[str, Optional[str]]:
        return goemotions_provenance(
            model_id=self.model_id,
            model_revision=self.model_revision,
        )


def goemotions_provenance(
    model_id: str = DEFAULT_GOEMOTIONS_MODEL_ID,
    model_revision: Optional[str] = DEFAULT_GOEMOTIONS_MODEL_REVISION,
) -> dict[str, Optional[str]]:
    """Return the requested GoEmotions identity.

    ``model_revision`` is ``None`` when no immutable SHA or tag was supplied.
    An explicit model ID alone is not a pin and does not make a run
    scientifically reproducible or domain-validated.
    """
    return {
        "model_id": model_id,
        "model_revision": model_revision,
    }


def _reset_sentiment_runtimes() -> None:
    """Drop cached runtimes. Intended for tests."""
    global _goemotions_runtime, _vader_analyzer
    _goemotions_runtime = None
    _vader_analyzer = None


def _load_goemotions_tokenizer(model_id: str, **kwargs: Any) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def _load_goemotions_model(model_id: str, **kwargs: Any) -> Any:
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)


def _new_vader_analyzer() -> Any:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def _pretrained_kwargs(
    *,
    model_revision: Optional[str],
    local_files_only: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if model_revision is not None:
        kwargs["revision"] = model_revision
    return kwargs


def _runtime_matches(
    runtime: GoEmotionsRuntime,
    model_id: str,
    model_revision: Optional[str],
    local_files_only: bool,
) -> bool:
    return (
        runtime.model_id == model_id
        and runtime.model_revision == model_revision
        and runtime.local_files_only == local_files_only
    )


def load_goemotions_runtime(
    model_id: str = DEFAULT_GOEMOTIONS_MODEL_ID,
    model_revision: Optional[str] = DEFAULT_GOEMOTIONS_MODEL_REVISION,
    *,
    local_files_only: bool = DEFAULT_GOEMOTIONS_LOCAL_FILES_ONLY,
) -> GoEmotionsRuntime:
    """Always construct a GoEmotions tokenizer/model for the requested identity."""
    import torch

    logger.info(
        "Loading GoEmotions runtime (model_id=%s, revision=%s)",
        model_id,
        model_revision,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = _pretrained_kwargs(
        model_revision=model_revision,
        local_files_only=local_files_only,
    )
    tokenizer = _load_goemotions_tokenizer(model_id, **loader_kwargs)
    model = _load_goemotions_model(model_id, **loader_kwargs).to(device)
    return GoEmotionsRuntime(
        model_id=model_id,
        model_revision=model_revision,
        tokenizer=tokenizer,
        model=model,
        device=device,
        id2label=model.config.id2label,
        local_files_only=local_files_only,
    )


def get_goemotions_runtime(
    model_id: str = DEFAULT_GOEMOTIONS_MODEL_ID,
    model_revision: Optional[str] = DEFAULT_GOEMOTIONS_MODEL_REVISION,
    *,
    local_files_only: bool = DEFAULT_GOEMOTIONS_LOCAL_FILES_ONLY,
) -> GoEmotionsRuntime:
    """Return a cached runtime, reloading when model identity changes."""
    global _goemotions_runtime
    cached = _goemotions_runtime
    if cached is not None and _runtime_matches(
        cached, model_id, model_revision, local_files_only
    ):
        return cached
    runtime = load_goemotions_runtime(
        model_id,
        model_revision,
        local_files_only=local_files_only,
    )
    _goemotions_runtime = runtime
    return runtime


def get_vader_analyzer() -> Any:
    """Return a cached VADER analyzer, creating it on first use."""
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = _new_vader_analyzer()
    return _vader_analyzer


def batch_tokenize(texts, tokenizer, max_length=DEFAULT_GOEMOTIONS_MAX_LENGTH):
    return tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )


def run_goemotions(
    texts,
    batch_size=32,
    *,
    model_id: str = DEFAULT_GOEMOTIONS_MODEL_ID,
    model_revision: Optional[str] = DEFAULT_GOEMOTIONS_MODEL_REVISION,
):
    import torch

    runtime = get_goemotions_runtime(
        model_id=model_id,
        model_revision=model_revision,
    )
    runtime.model.eval()
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = batch_tokenize(batch, runtime.tokenizer)
        inputs = {k: v.to(runtime.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = runtime.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()

        for i_probs in probs:
            all_probs = {
                runtime.id2label[i]: float(score) for i, score in enumerate(i_probs)
            }
            sorted_filtered = sorted(
                all_probs.items(), key=lambda x: x[1], reverse=True
            )
            results.append(sorted_filtered)
    return results


def run_vader(text):
    return get_vader_analyzer().polarity_scores(text)


def plot_goemotion_distribution(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("📊 Plotting GoEmotions distribution...")
    all_labels = (
        df["goemotions_top"]
        .explode()
        .dropna()
        .apply(
            lambda x: (
                ast.literal_eval(x)[0] if isinstance(x, str) else x[0] if x else None
            )
        )
    )
    sns.countplot(y=all_labels)
    plt.title("Top Predicted Emotions (GoEmotions)")
    plt.tight_layout()
    plt.show()


def plot_vader_distribution(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("📊 Plotting VADER compound score distribution...")
    compound_scores = df["vader"].apply(
        lambda x: (
            ast.literal_eval(x)["compound"] if isinstance(x, str) else x["compound"]
        )
    )
    sns.histplot(compound_scores, bins=30, kde=True)
    plt.title("VADER Compound Score Distribution")
    plt.xlabel("Compound Score")
    plt.tight_layout()
    plt.savefig(VADER_PLOT_PATH, dpi=300)
    plt.show()


def analyze_comments(comments):
    comment_emotions = []
    for c in comments:
        if not isinstance(c, str) or not c.strip():
            comment_emotions.append({"goemotions": [], "vader": {}})
            continue
        ge = run_goemotions([c])[0]
        vd = run_vader(c)
        comment_emotions.append({"goemotions": ge, "vader": vd})
    return comment_emotions


def main():
    df = pd.read_csv(FINAL_ANALYSIS_INPUT)
    df["text"] = df["text"].fillna("")

    texts = [t for t in df["text"].tolist() if t.strip()]
    print(f"🧠 Running GoEmotions on {len(texts)} texts...")
    ge_outputs = run_goemotions(texts)
    vader_outputs = [run_vader(t) for t in texts]

    df = df.loc[df["text"].str.strip() != ""].copy()
    df["goemotions_top"] = ge_outputs
    df["vader"] = vader_outputs

    if "comments" in df.columns:
        print("🧠 Analyzing comments...")
        df["comments"] = df["comments"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df["comment_sentiment"] = df["comments"].apply(analyze_comments)

    df.to_csv(SENTIMENT_OUTPUT, index=False)
    print(f"✅ Sentiment analysis complete. Output saved to: {SENTIMENT_OUTPUT}")

    plot_goemotion_distribution(df)
    plot_vader_distribution(df)


if __name__ == "__main__":
    main()
