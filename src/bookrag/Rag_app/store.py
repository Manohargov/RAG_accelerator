"""ChromaDB persistence layer.

ChromaDB is a strict API: metadata values must be scalars, every vector needs
a matching string document, and duplicate ids raise.  This module absorbs
those rules so the rest of the app can pass rich Python structures.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import BookRagError, DependencyError, StoreError
from ..logs.logging_setup import get_logger
from ..logs.monitoring import stage
from .settings import AppSettings

log = get_logger("store")

_INSTALL_HINT = "pip install -r requirements.txt"

#: ChromaDB only accepts these scalar types inside a metadata mapping.
_SCALARS = (str, int, float, bool)

#: Guard against pathological metadata blowing up the database file.
_MAX_METADATA_VALUE = 4096

#: Collection metadata key recording which checkpoint produced the vectors in
#: it.  Every model this app supports is 1024-dimensional, so a collection
#: built by one of them loads happily under any other and returns confidently
#: wrong scores with no error to give it away - this is the only thing that
#: catches that.
MODEL_FINGERPRINT_KEY = "bookrag:model_id"


def _domain_error(exc: Exception, message: str) -> StoreError:
    """Let an error that already explains itself through unchanged.

    The wrapping sites below catch everything coming out of chromadb, which is
    right for chromadb's own failures and wrong for ours: the fingerprint check
    raises a StoreError that names the command which fixes it, and re-wrapping
    that as "Count failed: ..." keeps the text while dropping the hint the user
    is meant to act on.
    """
    return exc if isinstance(exc, BookRagError) else StoreError(message)


def clean_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Coerce a metadata mapping into something ChromaDB will accept.

    ``None`` values are dropped, containers are stringified, and every value
    is truncated so a whole page of text cannot silently land in the store.
    """
    if not metadata:
        return {}
    clean: dict[str, Any] = {}
    for raw_key, raw_value in metadata.items():
        key = str(raw_key).strip()
        if not key:
            continue
        if raw_value is None:
            continue
        if isinstance(raw_value, (bool, int, float)):
            clean[key] = raw_value
        elif isinstance(raw_value, str):
            clean[key] = raw_value[:_MAX_METADATA_VALUE]
        else:
            clean[key] = str(raw_value)[:_MAX_METADATA_VALUE]
    return clean


def build_where(filters: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Build a ChromaDB ``where`` clause from simple equality filters.

    Multiple keys are combined with ``$and``; a single key is passed through
    directly because ChromaDB rejects ``$and`` with one operand on some
    versions.
    """
    if not filters:
        return None
    clauses = [{str(key): value} for key, value in filters.items() if value is not None]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class VectorStore:
    """Persistent ChromaDB collection of text and image embeddings."""

    def __init__(self, settings: AppSettings | None = None, path: Path | None = None) -> None:
        self.settings = settings or AppSettings()
        self.path = Path(path) if path else None
        self._client: Any = None
        self._collection: Any = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _import_chromadb(self) -> Any:
        try:
            import chromadb
        except ImportError as exc:
            raise DependencyError("ChromaDB is not installed.", _INSTALL_HINT) from exc
        return chromadb

    def connect(self, *, verify_fingerprint: bool = True) -> VectorStore:
        """Open (or create) the persistent client and collection.

        ``verify_fingerprint`` exists for the one caller that has to be able to
        open a collection it cannot vouch for: :meth:`reset`, which is the
        remedy the check itself recommends.
        """
        if self._collection is not None:
            return self

        with stage("store.connect"):
            chromadb = self._import_chromadb()
            from ..data import paths

            db_path = self.path or paths.db_dir()
            db_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(path=str(db_path))
            # Passing the fingerprint here only ever stamps a collection this
            # call creates: chromadb ignores the metadata argument for a
            # collection that already exists, so an index someone else built
            # keeps its own provenance.  Verified against chromadb 1.5.9.
            self._collection = self._client.get_or_create_collection(
                name=self.settings.collection_name,
                metadata=self._new_collection_metadata(),
            )
            if verify_fingerprint:
                try:
                    self._verify_model_fingerprint()
                except StoreError:
                    self._collection = None  # leave no half-open state behind
                    raise
        log.info(
            "Connected to collection %r at %s (space=%s).",
            self.settings.collection_name,
            db_path,
            self.settings.hnsw_space,
        )
        return self

    def _new_collection_metadata(self) -> dict[str, Any]:
        """The metadata every collection this code creates must carry.

        Both keys have to be written together, in every place a collection is
        brought into existence - ``connect``, ``reset`` and the fingerprint
        rebuild.  Writing one without the other is how an index ends up with a
        distance metric but no provenance, which is precisely the state the
        check below has no answer for and so has to refuse.
        """
        return {
            "hnsw:space": self.settings.hnsw_space,
            MODEL_FINGERPRINT_KEY: self.settings.model_id,
        }

    def _verify_model_fingerprint(self) -> None:
        """Refuse to let one checkpoint's vectors be searched by another's.

        Three states, ranked by how much is actually known:

        recorded and matching
            Proven safe; proceed.
        recorded and different
            Proven wrong; stop, because the scores would still look plausible.
        not recorded
            An empty collection is safe to adopt - there are no vectors to be
            inconsistent with, so it is rebuilt carrying the fingerprint.  A
            populated one has unknown provenance and is stopped: the
            alternative is answering from a checkpoint that may have nothing
            to do with those vectors.
        """
        stored = (self._collection.metadata or {}).get(MODEL_FINGERPRINT_KEY)
        current = self.settings.model_id

        if stored is None:
            if self._collection.count():
                raise StoreError(
                    f"Collection {self.settings.collection_name!r} holds "
                    f"{self._collection.count()} records but does not record "
                    f"which checkpoint built them, so BookRAG cannot check "
                    f"that {current!r} will read them correctly.",
                    "Rebuild the index with `bookrag reset` followed by "
                    "`bookrag ingest <file>`; embeddings from different models "
                    "are not comparable even at the same dimensionality.",
                )
            self._stamp_fingerprint()
            return

        if stored != current:
            raise StoreError(
                f"Collection {self.settings.collection_name!r} was built with "
                f"{stored!r}, but {current!r} is loaded now. Both produce "
                f"1024-dimensional vectors, so this would return confident, "
                f"plausible-looking and wrong results rather than failing.",
                f"Either rebuild under {current!r} - `bookrag reset` then "
                f"`bookrag ingest <file>` - or keep using this index exactly "
                f"as it is by running with `--model {stored}`.",
            )

    def _stamp_fingerprint(self) -> None:
        """Record which checkpoint produced the vectors in this collection.

        Rebuilding is the only route, because chromadb's ``modify()`` cannot
        be used here: it rejects any payload containing ``hnsw:space``, and
        replaces the remainder of the mapping outright, so there is no way to
        add a key without losing the distance metric.  That is safe precisely
        because this is only ever reached on a collection holding no vectors.
        """
        name = self.settings.collection_name
        self._client.delete_collection(name)
        self._collection = self._client.create_collection(
            name=name,
            metadata=self._new_collection_metadata(),
        )

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self.connect()
        return self._collection

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]] | np.ndarray,
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        """Insert or replace vectors. Returns the number of records written."""
        if len(ids) == 0:
            return 0
        if not (len(ids) == len(documents) == len(embeddings)):
            raise StoreError(
                "ids, embeddings and documents must have the same length "
                f"(got {len(ids)}, {len(embeddings)}, {len(documents)})."
            )
        if metadatas is not None and len(metadatas) != len(ids):
            raise StoreError("metadatas must have the same length as ids.")

        payload = np.asarray(embeddings, dtype=np.float32)
        if payload.ndim != 2:
            raise StoreError(f"Embeddings must be 2-D, got shape {payload.shape}.")

        clean_meta: list[dict[str, Any]] | None = None
        if metadatas is not None:
            clean_meta = [clean_metadata(m) for m in metadatas]

        # Resolved before the timer so a lazy connect is not billed to the write.
        collection = self.collection
        with stage("store.upsert", items=len(ids), collection=self.settings.collection_name):
            try:
                collection.upsert(
                    ids=[str(i) for i in ids],
                    embeddings=payload.tolist(),
                    documents=[str(d) for d in documents],
                    metadatas=clean_meta,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
                raise _domain_error(exc, f"Failed to write {len(ids)} records: {exc}") from exc

        log.info("Upserted %d records into %r.", len(ids), self.settings.collection_name)
        return len(ids)

    def delete(
        self, ids: Sequence[str] | None = None, where: Mapping[str, Any] | None = None
    ) -> None:
        """Delete records by id and/or metadata filter."""
        if not ids and not where:
            raise StoreError("delete() needs at least one of ids or where.")
        collection = self.collection
        with stage("store.delete", items=len(ids) if ids else 0, filtered=where is not None):
            try:
                collection.delete(
                    ids=[str(i) for i in ids] if ids else None,
                    where=dict(where) if where else None,
                )
            except Exception as exc:  # noqa: BLE001
                raise _domain_error(exc, f"Delete failed: {exc}") from exc

    def delete_where(self, filters: Mapping[str, Any]) -> None:
        """Delete every record matching a metadata filter."""
        clause = build_where(filters)
        if clause is None:
            return
        self.delete(where=clause)

    def delete_all(self) -> int:
        """Delete every record in the collection, keeping the collection itself.

        Returns the number of records removed.
        """
        try:
            raw = self.collection.get(include=[])
            ids = [str(i) for i in (raw.get("ids") or [])]
        except Exception as exc:  # noqa: BLE001
            raise _domain_error(exc, f"Could not list records: {exc}") from exc

        if not ids:
            return 0
        self.delete(ids=ids)
        return len(ids)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def query(
        self,
        embedding: Sequence[float] | np.ndarray,
        top_k: int | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a nearest-neighbour query and return flat, typed results."""
        k = int(top_k or self.settings.top_k)
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1).tolist()
        clause = build_where(where)

        # Resolved before the timer so a lazy connect is not billed to the search.
        collection = self.collection
        with stage("store.query", items=k, filtered=clause is not None):
            try:
                raw = collection.query(
                    query_embeddings=vector,
                    n_results=k,
                    where=clause,
                )
            except Exception as exc:  # noqa: BLE001
                raise _domain_error(exc, f"Query failed: {exc}") from exc

            ids = (raw.get("ids") or [[]])[0]
            documents = (raw.get("documents") or [[]])[0]
            metadatas = (raw.get("metadatas") or [[]])[0]
            distances = (raw.get("distances") or [[]])[0]

            results: list[dict[str, Any]] = []
            for index, record_id in enumerate(ids):
                distance = float(distances[index]) if index < len(distances) else float("nan")
                results.append(
                    {
                        "id": str(record_id),
                        "document": documents[index] if index < len(documents) else "",
                        "metadata": metadatas[index] if index < len(metadatas) else {},
                        "distance": distance,
                        # ChromaDB reports cosine distance, i.e. 1 - similarity.
                        "score": 1.0 - distance if distance == distance else 0.0,
                    }
                )
        return results

    def get(
        self, ids: Sequence[str] | None = None, where: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Fetch raw records (mainly for inspection and debugging)."""
        try:
            return self.collection.get(
                ids=[str(i) for i in ids] if ids else None,
                where=build_where(where),
            )
        except Exception as exc:  # noqa: BLE001
            raise _domain_error(exc, f"Get failed: {exc}") from exc

    def count(self) -> int:
        """Number of records currently stored."""
        try:
            return int(self.collection.count())
        except Exception as exc:  # noqa: BLE001
            raise _domain_error(exc, f"Count failed: {exc}") from exc

    def stats(self) -> dict[str, Any]:
        """Summary information about the collection."""
        total = self.count()
        breakdown: dict[str, int] = {}
        sources: set[str] = set()
        try:
            raw = self.collection.get(include=["metadatas"])
            for meta in raw.get("metadatas") or []:
                meta = meta or {}
                kind = str(meta.get("type", "unknown"))
                breakdown[kind] = breakdown.get(kind, 0) + 1
                if meta.get("source"):
                    sources.add(str(meta["source"]))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read collection metadata for stats: %s", exc)

        return {
            "collection": self.settings.collection_name,
            "total": total,
            "by_type": breakdown,
            "sources": sorted(sources),
            "space": self.settings.hnsw_space,
        }

    def reset(self) -> None:
        """Delete the collection and recreate it empty.

        Deliberately skips the fingerprint check.  Rebuilding under a
        different checkpoint is precisely what that check tells a user to do,
        so refusing to open the collection here would leave them unable to
        follow the instruction the error has just handed them.
        """
        client = self._client
        if client is None:
            self.connect(verify_fingerprint=False)
            client = self._client
        with contextlib.suppress(Exception):
            client.delete_collection(self.settings.collection_name)
        self._collection = client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata=self._new_collection_metadata(),
        )
        log.info("Collection %r reset.", self.settings.collection_name)
