"""Tests for the embedder's API selection, task instructions and normalisation.

The real model is never loaded: the tests drive the private helpers with a
stand-in model object so the branching logic is still covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from bookrag.accelerator.embedder import ClipEmbedder, l2_normalise, resolve_device, resolve_dtype
from bookrag.errors import ModelUnavailableError
from bookrag.Rag_app.settings import AppSettings


class _FakeModel:
    """Stands in for ``JinaCLIPModel``."""

    def __init__(self, api: str = "encode") -> None:
        self.api = api
        self.calls: list[dict] = []
        self._tasks = {"retrieval.query": "Represent the query for retrieving evidence documents: "}

        class _Tower:
            def __init__(self, outer: _FakeModel) -> None:
                self.outer = outer

            def get_instruction_from_task(self, task: str | None) -> str | None:
                if task is None:
                    return None
                if task not in self.outer._tasks:
                    raise ValueError(f"Unsupported task '{task}'")
                return self.outer._tasks[task]

        self.text_model = _Tower(self)

    def encode_text(self, sentences, **kwargs):
        self.calls.append({"kind": "text", "sentences": list(sentences), **kwargs})
        vectors = np.arange(len(sentences) * 4, dtype=np.float32).reshape(len(sentences), 4)
        return vectors

    def encode_image(self, images, **kwargs):
        self.calls.append({"kind": "image", "count": len(images), **kwargs})
        return np.arange(len(images) * 4, dtype=np.float32).reshape(len(images), 4)

    def get_text_features(self, **kwargs):
        self.calls.append({"kind": "features-text"})
        raise AssertionError("should not be reached")

    def get_image_features(self, **kwargs):
        raise AssertionError("should not be reached")


def _embedder_with(model: _FakeModel, **overrides) -> ClipEmbedder:
    settings = AppSettings(**overrides)
    embedder = ClipEmbedder(settings)
    embedder._model = model
    embedder._api = embedder._detect_api()
    return embedder


def test_detect_api_prefers_encode():
    assert _embedder_with(_FakeModel("encode")).api == "encode"


def test_detect_api_falls_back_to_features():
    class FeaturesOnly:
        def get_text_features(self, **kwargs): ...
        def get_image_features(self, **kwargs): ...

    embedder = ClipEmbedder(AppSettings())
    embedder._model = FeaturesOnly()
    assert embedder._detect_api() == "features"


def test_detect_api_falls_back_to_legacy_embeddings():
    class Legacy:
        def get_text_embeddings(self, **kwargs): ...
        def get_image_embeddings(self, **kwargs): ...

    embedder = ClipEmbedder(AppSettings())
    embedder._model = Legacy()
    assert embedder._detect_api() == "embeddings"


def test_detect_api_raises_for_unknown_model():
    embedder = ClipEmbedder(AppSettings())
    embedder._model = object()
    with pytest.raises(ModelUnavailableError):
        embedder._detect_api()


def test_encode_text_passes_task():
    model = _FakeModel()
    embedder = _embedder_with(model)
    embedder.embed_texts(["hello", "world"], task="retrieval.query")
    assert model.calls[0]["task"] == "retrieval.query"


def test_unknown_task_is_dropped():
    model = _FakeModel()
    embedder = _embedder_with(model)
    embedder.embed_texts(["hello"], task="not.a.task")
    assert "task" not in model.calls[0]


def test_no_task_means_no_instruction():
    model = _FakeModel()
    embedder = _embedder_with(model)
    embedder.embed_texts(["hello"], task=None)
    assert "task" not in model.calls[0]


def test_text_embeddings_are_normalised():
    embedder = _embedder_with(_FakeModel())
    vectors = embedder.embed_texts(["a", "b"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_image_embeddings_are_normalised():
    from PIL import Image

    embedder = _embedder_with(_FakeModel())
    images = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    vectors = embedder.embed_images(images)
    assert vectors.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_max_text_length_is_forwarded():
    model = _FakeModel()
    embedder = _embedder_with(model, max_text_length=256)
    embedder.embed_texts(["hello"])
    assert model.calls[0]["max_length"] == 256


def test_l2_normalise_handles_zero_rows():
    matrix = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    result = l2_normalise(matrix)
    assert result[0].tolist() == [0.0, 0.0]
    assert np.allclose(result[1], [0.6, 0.8])


def test_l2_normalise_promotes_1d_to_2d():
    result = l2_normalise(np.array([3.0, 4.0], dtype=np.float32))
    assert result.shape == (1, 2)


def test_resolve_dtype_auto_is_float32_on_cpu():
    assert resolve_dtype("auto", "cpu") == "float32"


def test_resolve_dtype_respects_explicit_choice():
    assert resolve_dtype("bfloat16", "cpu") == "bfloat16"


def test_resolve_dtype_on_xpu_is_bfloat16():
    # Arc / AI Boost: bf16 matches the published checkpoint, so the weights
    # never have to be cast - which is what kept peak RSS at 5.5 GB on CPU.
    assert resolve_dtype("auto", "xpu") == "bfloat16"


def test_resolve_device_explicit_cpu():
    pytest.importorskip("torch")
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_a_known_device():
    pytest.importorskip("torch")
    assert resolve_device("auto") in {"cpu", "cuda", "mps", "cuda:0", "xpu"}
