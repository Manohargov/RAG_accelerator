"""BookRAG - multimodal retrieval over books and PDFs.

Jina CLIP v2 embeds text and images into a shared vector space; ChromaDB
persists those vectors and answers nearest-neighbour queries.

The package is organised into four subpackages:

* ``accelerator`` — model loading, device management, and embedding
* ``Rag_app`` — pipeline, retrieval, ingestion, and storage
* ``logs`` — logging and monitoring infrastructure
* ``data`` — filesystem paths and directory layout
"""

from .accelerator.embedder import ClipEmbedder
from .errors import (
    BookRagError,
    ConfigError,
    DependencyError,
    IngestError,
    ModelUnavailableError,
    QueryError,
    StoreError,
)
from .Rag_app.pipeline import BookRagPipeline, IngestReport
from .Rag_app.retriever import Retriever, SearchHit
from .Rag_app.settings import AppSettings
from .Rag_app.store import VectorStore
from .version import __version__

__all__ = [
    "__version__",
    "AppSettings",
    "BookRagError",
    "BookRagPipeline",
    "ClipEmbedder",
    "ConfigError",
    "DependencyError",
    "IngestError",
    "IngestReport",
    "ModelUnavailableError",
    "QueryError",
    "Retriever",
    "SearchHit",
    "StoreError",
    "VectorStore",
]
