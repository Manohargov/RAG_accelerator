"""Retrieval over the embedded corpus.

Because Jina CLIP v2 puts text and images in a shared vector space, a single
retriever serves both directions:

* a **text** query can return a paragraph *or* the figure that illustrates it;
* an **image** query can return similar figures *or* the text near them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..accelerator.embedder import ClipEmbedder
from ..errors import QueryError
from ..logs.logging_setup import get_logger
from ..logs.monitoring import stage
from .settings import AppSettings
from .store import VectorStore

log = get_logger("retriever")


@dataclass
class SearchHit:
    """One retrieved record."""

    id: str
    score: float
    distance: float
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """``text`` or ``image``."""
        return str(self.metadata.get("type", "unknown"))

    @property
    def page(self) -> int | None:
        value = self.metadata.get("page")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": round(self.score, 4),
            "distance": round(self.distance, 4),
            "document": self.document,
            "metadata": self.metadata,
        }


class Retriever:
    """Combines the embedder and the store into a single search interface."""

    def __init__(
        self,
        embedder: ClipEmbedder | None = None,
        store: VectorStore | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.embedder = embedder or ClipEmbedder(self.settings)
        self.store = store or VectorStore(self.settings)

    # ------------------------------------------------------------------ #
    def search_text(
        self,
        query: str,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Search the corpus with a natural-language question."""
        if not query or not query.strip():
            raise QueryError("Query text must not be empty.")

        # A container: embed.text and store.query inside it are the leaves.
        with stage(
            "query.total",
            container=True,
            kind="text",
            top_k=int(top_k or self.settings.top_k),
        ):
            # Queries carry Jina's retrieval instruction; documents do not.
            # This asymmetry is what the model was trained for.
            vector = self.embedder.embed_text(query, task=self.settings.query_task or None)
            return self._search(vector, top_k, filters)

    def search_image(
        self,
        image_path: str | Path,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Search the corpus with an image (e.g. find a similar figure)."""
        path = Path(image_path)
        if not path.exists():
            raise QueryError(f"Image not found: {path}")

        with stage(
            "query.total",
            container=True,
            kind="image",
            top_k=int(top_k or self.settings.top_k),
        ):
            vector = self.embedder.embed_image(path)
            return self._search(vector, top_k, filters)

    def search(
        self,
        text: str | None = None,
        image: str | Path | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Search with text, an image, or both.

        When both are supplied the two vectors are averaged (after L2
        normalisation) so the query sits between the two modalities.
        """
        if not text and not image:
            raise QueryError("Provide a text query, an image, or both.")

        with stage(
            "query.total", container=True, kind="combined", top_k=int(top_k or self.settings.top_k)
        ):
            vectors = []
            if text and text.strip():
                vectors.append(
                    self.embedder.embed_text(text, task=self.settings.query_task or None)
                )
            if image:
                path = Path(image)
                if not path.exists():
                    raise QueryError(f"Image not found: {path}")
                vectors.append(self.embedder.embed_image(path))

            if len(vectors) == 1:
                combined = vectors[0]
            else:
                combined = np.mean(np.vstack(vectors), axis=0)
                norm = float((combined**2).sum() ** 0.5) or 1.0
                combined = combined / norm

            return self._search(combined, top_k, filters)

    # ------------------------------------------------------------------ #
    def _search(
        self,
        vector: Any,
        top_k: int | None,
        filters: Mapping[str, Any] | None,
    ) -> list[SearchHit]:
        k = int(top_k or self.settings.top_k)
        raw = self.store.query(vector, top_k=k, where=filters)
        hits = [
            SearchHit(
                id=item["id"],
                score=float(item["score"]),
                distance=float(item["distance"]),
                document=str(item.get("document") or ""),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in raw
        ]
        log.info("Retrieved %d hits (top_k=%d).", len(hits), k)
        return hits
