"""Offline tests for lazy sentiment loading. No model download or network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from analysis.sentiment_analysis import (
    DEFAULT_GOEMOTIONS_MODEL_ID,
    DEFAULT_GOEMOTIONS_MODEL_REVISION,
    GoEmotionsRuntime,
    _reset_sentiment_runtimes,
    get_goemotions_runtime,
    get_vader_analyzer,
    goemotions_provenance,
    run_goemotions,
    run_vader,
)


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        n = len(texts)
        return {
            "input_ids": torch.zeros(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }


class _FakeModel:
    def __init__(self, label: str = "joy"):
        self.config = SimpleNamespace(id2label={0: label, 1: "anger"})
        self._eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self._eval_called = True

    def __call__(self, **inputs):
        n = inputs["input_ids"].shape[0]
        logits = torch.tensor([[4.0, -4.0]] * n)
        return SimpleNamespace(logits=logits)


class _FakeVader:
    def polarity_scores(self, text):
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}


@pytest.fixture(autouse=True)
def _clear_sentiment_runtimes():
    _reset_sentiment_runtimes()
    yield
    _reset_sentiment_runtimes()


def _install_fake_goemotions(monkeypatch, load_calls):
    import analysis.sentiment_analysis as sa

    def fake_tokenizer(model_id, **kwargs):
        load_calls.append(("tokenizer", model_id, dict(kwargs)))
        return _FakeTokenizer()

    def fake_model(model_id, **kwargs):
        load_calls.append(("model", model_id, dict(kwargs)))
        return _FakeModel()

    monkeypatch.setattr(sa, "_load_goemotions_tokenizer", fake_tokenizer)
    monkeypatch.setattr(sa, "_load_goemotions_model", fake_model)


def test_importing_sentiment_analysis_does_not_initialize_runtimes(monkeypatch, capsys):
    tokenizer_calls = []
    model_calls = []
    vader_calls = []

    import vaderSentiment.vaderSentiment as vader_mod
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    def fake_tokenizer(*args, **kwargs):
        tokenizer_calls.append((args, kwargs))
        raise AssertionError("GoEmotions tokenizer loaded during import")

    def fake_model(*args, **kwargs):
        model_calls.append((args, kwargs))
        raise AssertionError("GoEmotions model loaded during import")

    def fake_vader(*args, **kwargs):
        vader_calls.append((args, kwargs))
        raise AssertionError("VADER initialized during import")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(
        AutoModelForSequenceClassification, "from_pretrained", fake_model
    )
    monkeypatch.setattr(vader_mod, "SentimentIntensityAnalyzer", fake_vader)

    path = Path(__file__).resolve().parents[2] / "analysis" / "sentiment_analysis.py"
    spec = importlib.util.spec_from_file_location(
        "analysis._sentiment_import_probe", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        captured = capsys.readouterr()

        assert tokenizer_calls == []
        assert model_calls == []
        assert vader_calls == []
        assert module._goemotions_runtime is None
        assert module._vader_analyzer is None
        assert captured.out == ""
        assert "Loading" not in captured.err
        assert "Model loaded" not in captured.err
    finally:
        sys.modules.pop(spec.name, None)


def test_goemotions_initializes_lazily_and_reuses_runtime(monkeypatch):
    load_calls = []
    _install_fake_goemotions(monkeypatch, load_calls)

    import analysis.sentiment_analysis as sa

    assert load_calls == []
    assert sa._goemotions_runtime is None

    first = run_goemotions(["synthetic post about images"])
    assert len(load_calls) == 2
    assert all(kind in {"tokenizer", "model"} for kind, _, _ in load_calls)

    second = run_goemotions(["another synthetic post"])
    assert len(load_calls) == 2
    assert sa._goemotions_runtime is not None
    assert first[0][0][0] == "joy"
    assert isinstance(second, list)
    assert isinstance(second[0], list)
    assert isinstance(second[0][0], tuple)
    assert isinstance(second[0][0][0], str)
    assert isinstance(second[0][0][1], float)


def test_goemotions_cache_is_configuration_sensitive(monkeypatch):
    load_calls = []
    _install_fake_goemotions(monkeypatch, load_calls)

    first = get_goemotions_runtime(model_id="model-a", model_revision="R1")
    second = get_goemotions_runtime(model_id="model-a", model_revision="R1")
    assert first is second
    assert len(load_calls) == 2

    third = get_goemotions_runtime(model_id="model-a", model_revision="R2")
    assert third is not first
    assert third.model_revision == "R2"
    assert first.model_revision == "R1"
    assert len(load_calls) == 4
    assert load_calls[-1][1] == "model-a"
    assert load_calls[-1][2]["revision"] == "R2"

    fourth = get_goemotions_runtime(model_id="model-b")
    assert fourth is not third
    assert fourth.model_id == "model-b"
    assert fourth.model_revision is None
    assert len(load_calls) == 6
    assert load_calls[-1][1] == "model-b"
    assert "revision" not in load_calls[-1][2]


def test_revision_is_propagated_to_pretrained_loaders(monkeypatch):
    load_calls = []
    _install_fake_goemotions(monkeypatch, load_calls)

    get_goemotions_runtime(model_id="model-a", model_revision="abc123")
    assert load_calls[0][2]["local_files_only"] is True
    assert load_calls[1][2]["local_files_only"] is True
    assert load_calls[0][2]["revision"] == "abc123"
    assert load_calls[1][2]["revision"] == "abc123"

    _reset_sentiment_runtimes()
    load_calls.clear()
    get_goemotions_runtime(model_id="model-a", model_revision=None)
    assert "revision" not in load_calls[0][2]
    assert "revision" not in load_calls[1][2]
    assert load_calls[0][2]["local_files_only"] is True


def test_vader_initializes_lazily_and_is_cached(monkeypatch):
    import analysis.sentiment_analysis as sa

    calls = {"n": 0}

    def fake_new():
        calls["n"] += 1
        return _FakeVader()

    monkeypatch.setattr(sa, "_new_vader_analyzer", fake_new)

    assert calls["n"] == 0
    first = run_vader("synthetic text")
    assert calls["n"] == 1
    second = run_vader("another synthetic text")
    assert calls["n"] == 1
    assert first["compound"] == 0.0
    assert second["neu"] == 1.0
    assert get_vader_analyzer() is sa._vader_analyzer


def test_goemotions_provenance_reports_requested_identity():
    assert goemotions_provenance() == {
        "model_id": DEFAULT_GOEMOTIONS_MODEL_ID,
        "model_revision": None,
    }
    assert DEFAULT_GOEMOTIONS_MODEL_REVISION is None
    assert DEFAULT_GOEMOTIONS_MODEL_ID == "SamLowe/roberta-base-go_emotions"

    runtime = GoEmotionsRuntime(
        model_id="model-a",
        model_revision="R1",
        tokenizer=None,
        model=None,
        device="cpu",
        id2label={},
        local_files_only=True,
    )
    assert runtime.provenance() == {"model_id": "model-a", "model_revision": "R1"}
    unpinned = goemotions_provenance(model_id="model-a", model_revision=None)
    assert unpinned["model_revision"] is None
    assert unpinned["model_id"] == "model-a"
