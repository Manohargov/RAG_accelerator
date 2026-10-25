"""Test doubles used to exercise the app without the heavy ML stack.

The fakes implement just enough of the ChromaDB and CLIP behaviour to
validate the application's own logic (metadata coercion, filtering, scoring,
idempotent re-ingestion) on a machine where torch and chromadb are absent.
"""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Fake ChromaDB
# --------------------------------------------------------------------------- #
def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Evaluate a (simplified) ChromaDB where-clause against one record."""
    if not where:
        return True
    if "$and" in where:
        return all(_matches(metadata, clause) for clause in where["$and"])
    if "$or" in where:
        return any(_matches(metadata, clause) for clause in where["$or"])
    for key, expected in where.items():
        if key.startswith("$"):
            continue
        if metadata.get(key) != expected:
            return False
    return True


class FakeCollection:
    """In-memory stand-in for a ChromaDB collection."""

    def __init__(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        self.name = name
        self.metadata = metadata or {}
        self.records: dict[str, dict[str, Any]] = {}

    # -- writes ---------------------------------------------------------- #
    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        for index, record_id in enumerate(ids):
            self.records[str(record_id)] = {
                "embedding": list(embeddings[index]),
                "document": documents[index],
                "metadata": dict(metadatas[index]) if metadatas else {},
            }

    def add(self, **kwargs: Any) -> None:
        for record_id in list(self.records):
            raise RuntimeError(f"Duplicate id: {record_id}")
        self.upsert(**kwargs)

    def delete(
        self,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        for record_id in list(self.records):
            if ids is not None and record_id in {str(i) for i in ids}:
                del self.records[record_id]
                continue
            if ids is None and _matches(self.records[record_id]["metadata"], where):
                del self.records[record_id]

    def modify(self, metadata: dict[str, Any] | None = None) -> None:
        """ChromaDB replaces the mapping outright; it does not merge keys."""
        if metadata is not None:
            self.metadata = dict(metadata)

    # -- reads ----------------------------------------------------------- #
    def count(self) -> int:
        return len(self.records)

    def get(
        self,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        include: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        selected = [
            (record_id, record)
            for record_id, record in self.records.items()
            if (ids is None or record_id in {str(i) for i in ids})
            and _matches(record["metadata"], where)
        ]
        return {
            "ids": [record_id for record_id, _ in selected],
            "documents": [record["document"] for _, record in selected],
            "metadatas": [record["metadata"] for _, record in selected],
            "embeddings": [record["embedding"] for _, record in selected],
        }

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query = list(query_embeddings[0])
        candidates = [
            (record_id, record)
            for record_id, record in self.records.items()
            if _matches(record["metadata"], where)
        ]

        scored: list[tuple[str, dict[str, Any], float]] = []
        for record_id, record in candidates:
            distance = _cosine_distance(query, record["embedding"])
            scored.append((record_id, record, distance))
        scored.sort(key=lambda item: item[2])

        top = scored[:n_results]
        return {
            "ids": [[record_id] for record_id, _, _ in top],
            "documents": [[record["document"] for _, record, _ in top]],
            "metadatas": [[record["metadata"] for _, record, _ in top]],
            "distances": [[distance for _, _, distance in top]],
        }


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return 1.0 - (dot / (norm_a * norm_b))


class FakeClient:
    """In-memory stand-in for ``chromadb.PersistentClient``."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> FakeCollection:
        if name not in self.collections:
            self.collections[name] = FakeCollection(name, metadata)
        return self.collections[name]

    def create_collection(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> FakeCollection:
        if name in self.collections:
            raise ValueError(f"Collection {name} already exists.")
        self.collections[name] = FakeCollection(name, metadata)
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)


#: One client per on-disk directory, the way a real persistent client behaves.
_PERSISTENT_CLIENTS: dict[str, FakeClient] = {}


def install_fake_chromadb() -> types.ModuleType:
    """Register a fake ``chromadb`` module in ``sys.modules``."""
    module = types.ModuleType("chromadb")

    def persistent_client(path: str | None = None) -> FakeClient:
        # Real ChromaDB keeps everything under one directory, so a second
        # client on the same path sees what the first one wrote.  Reconnects
        # rely on that, and the per-test temp directory keeps the clients
        # from leaking into one another.
        key = str(path)
        if key not in _PERSISTENT_CLIENTS:
            _PERSISTENT_CLIENTS[key] = FakeClient(path)
        return _PERSISTENT_CLIENTS[key]

    module.PersistentClient = persistent_client  # type: ignore[attr-defined]
    module.Client = persistent_client  # type: ignore[attr-defined]
    sys.modules["chromadb"] = module
    return module


# --------------------------------------------------------------------------- #
# Fake embedder
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    """Deterministic hash-based embedder with the CLIPEmbedder interface."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self.device = "fake"
        self.is_loaded = True
        self._cache: dict[str, np.ndarray] = {}

    def _vector(self, payload: str) -> np.ndarray:
        if payload in self._cache:
            return self._cache[payload]
        vector = np.zeros(self.dimension, dtype=np.float32)
        for index, char in enumerate(payload):
            vector[index % self.dimension] += (ord(char) % 17) / 17.0
        norm = float((vector**2).sum() ** 0.5) or 1.0
        vector = vector / norm
        self._cache[payload] = vector
        return vector

    def embed_texts(self, texts: Sequence[str], task: str | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts])

    def embed_text(self, text: str, task: str | None = None) -> np.ndarray:
        return self._vector(text)

    def embed_images(self, images: Sequence[Any]) -> np.ndarray:
        return self.embed_texts(
            [f"image:{getattr(image, 'filename', id(image))}" for image in images]
        )

    def embed_image_paths(self, paths: Sequence[str | Path]) -> np.ndarray:
        return self.embed_texts([f"image:{Path(p).name}" for p in paths])

    def embed_image(self, path: str | Path) -> np.ndarray:
        return self._vector(f"image:{Path(path).name}")

    def unload(self) -> None:
        self._cache.clear()
