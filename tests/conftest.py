"""Shared pytest fixtures.

The suite runs without torch / transformers / chromadb installed: a fake
``chromadb`` module is injected and the real model is replaced by a
deterministic hash embedder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fakes import FakeEmbedder, install_fake_chromadb  # noqa: E402

install_fake_chromadb()


@pytest.fixture(autouse=True)
def hermetic_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every path helper at a throwaway directory."""
    home = tmp_path / "bookrag-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("BOOKRAG_HOME", str(home))
    return home


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def settings():
    from bookrag.Rag_app.settings import AppSettings

    return AppSettings(chunk_size=200, chunk_overlap=40, batch_size=4, top_k=3)


@pytest.fixture
def store(settings):
    from bookrag.Rag_app.store import VectorStore

    return VectorStore(settings).connect()


@pytest.fixture
def pipeline(settings, fake_embedder, store):
    from bookrag.Rag_app.pipeline import BookRagPipeline

    return BookRagPipeline(settings, embedder=fake_embedder, store=store)
