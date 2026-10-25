"""Optional test against the real Jina CLIP v2 checkpoint.

Skipped by default so the suite stays fast and dependency-free.  Run it with:

    pytest -m real_model

The first run downloads roughly 1.8 GB of weights from HuggingFace.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.real_model

MODEL_ID = "jinaai/jina-clip-v2"


@pytest.fixture(scope="module")
def embedder():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from bookrag.accelerator.embedder import ClipEmbedder
    from bookrag.Rag_app.settings import AppSettings

    settings = AppSettings(model_id=MODEL_ID, device="cpu", dtype="float32", batch_size=4)
    instance = ClipEmbedder(settings)
    try:
        instance.load()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load {MODEL_ID}: {exc}")
    yield instance
    instance.unload()


def test_model_exposes_a_supported_api(embedder):
    assert embedder.api in {"encode", "features", "embeddings"}


def test_text_embeddings_are_unit_length(embedder):
    vectors = embedder.embed_texts(
        [
            "The formula for kinetic energy is E = 1/2 m v^2.",
            "A line graph showing pressure against temperature.",
        ]
    )
    assert vectors.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)


def test_image_and_text_share_a_vector_space(embedder, tmp_path):
    from PIL import Image, ImageDraw

    graph = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(graph)
    draw.line([40, 260, 360, 40], fill="crimson", width=5)
    for step in range(5):
        x = 60 + step * 70
        y = 240 - step * 45
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill="navy")
    path = tmp_path / "graph.png"
    graph.save(path)

    captions = [
        "The formula for kinetic energy is E = 1/2 m v^2.",
        "A line graph showing the linear relationship between temperature and pressure.",
        "A schematic diagram of a closed thermodynamic system.",
    ]
    text_vectors = embedder.embed_texts(captions)
    image_vector = embedder.embed_image(path)

    similarities = text_vectors @ image_vector
    assert int(np.argmax(similarities)) == 1, (
        f"the graph matched {captions[int(np.argmax(similarities))]!r}"
    )


def test_task_instruction_changes_the_embedding(embedder):
    plain = embedder.embed_text("kinetic energy", task=None)
    instructed = embedder.embed_text("kinetic energy", task="retrieval.query")
    assert float(np.linalg.norm(plain - instructed)) > 1e-4


def test_unknown_task_is_ignored(embedder):
    vectors = embedder.embed_texts(["hello"], task="not.a.real.task")
    assert vectors.shape == (1, 1024)
