"""Tests for the ChromaDB wrapper (metadata coercion, filtering, scoring)."""

from __future__ import annotations

import numpy as np
import pytest

from bookrag.errors import StoreError
from bookrag.Rag_app.store import VectorStore, build_where, clean_metadata


def test_clean_metadata_drops_none():
    assert clean_metadata({"a": 1, "b": None}) == {"a": 1}


def test_clean_metadata_stringifies_containers():
    result = clean_metadata({"tags": ["a", "b"], "count": 3, "flag": True})
    assert result["tags"] == "['a', 'b']"
    assert result["count"] == 3
    assert result["flag"] is True


def test_clean_metadata_truncates_long_values():
    result = clean_metadata({"note": "x" * 10_000})
    assert len(result["note"]) == 4096


def test_clean_metadata_handles_empty():
    assert clean_metadata(None) == {}
    assert clean_metadata({}) == {}


def test_build_where_single_clause():
    assert build_where({"type": "text"}) == {"type": "text"}


def test_build_where_multiple_clauses():
    clause = build_where({"type": "text", "source": "a.pdf"})
    assert clause == {"$and": [{"type": "text"}, {"source": "a.pdf"}]}


def test_build_where_ignores_none_values():
    assert build_where({"type": None}) is None
    assert build_where(None) is None


def test_upsert_and_query_roundtrip(store: VectorStore):
    store.upsert(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["first", "second"],
        metadatas=[{"type": "text"}, {"type": "image"}],
    )
    assert store.count() == 2

    results = store.query([1.0, 0.0], top_k=2)
    assert results[0]["id"] == "a"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-5)


def test_query_respects_metadata_filter(store: VectorStore):
    store.upsert(
        ids=["t", "i"],
        embeddings=[[1.0, 0.0], [1.0, 0.0]],
        documents=["text doc", "image doc"],
        metadatas=[{"type": "text"}, {"type": "image"}],
    )
    results = store.query([1.0, 0.0], top_k=5, where={"type": "image"})
    assert [r["id"] for r in results] == ["i"]


def test_upsert_is_idempotent(store: VectorStore):
    store.upsert(
        ids=["a"], embeddings=[[1.0, 0.0]], documents=["one"], metadatas=[{"type": "text"}]
    )
    store.upsert(
        ids=["a"], embeddings=[[0.0, 1.0]], documents=["two"], metadatas=[{"type": "text"}]
    )
    assert store.count() == 1
    assert store.get(["a"])["documents"] == ["two"]


def test_upsert_rejects_mismatched_lengths(store: VectorStore):
    with pytest.raises(StoreError):
        store.upsert(ids=["a", "b"], embeddings=[[1.0, 0.0]], documents=["x", "y"])


def test_upsert_rejects_non_2d_embeddings(store: VectorStore):
    with pytest.raises(StoreError):
        store.upsert(ids=["a"], embeddings=[1.0, 0.0], documents=["x"], metadatas=None)  # type: ignore[arg-type]


def test_delete_where_removes_matching_records(store: VectorStore):
    store.upsert(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["one", "two"],
        metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}],
    )
    store.delete_where({"source": "a.pdf"})
    assert store.count() == 1
    assert store.get()["ids"] == ["b"]


def test_stats_reports_breakdown(store: VectorStore):
    store.upsert(
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        documents=["x", "y", "z"],
        metadatas=[
            {"type": "text", "source": "a.pdf"},
            {"type": "text", "source": "a.pdf"},
            {"type": "image", "source": "b.pdf"},
        ],
    )
    stats = store.stats()
    assert stats["total"] == 3
    assert stats["by_type"] == {"text": 2, "image": 1}
    assert stats["sources"] == ["a.pdf", "b.pdf"]


def test_reset_clears_the_collection(store: VectorStore):
    store.upsert(ids=["a"], embeddings=[[1.0, 0.0]], documents=["x"], metadatas=[{"type": "text"}])
    store.reset()
    assert store.count() == 0


def test_query_accepts_numpy_vector(store: VectorStore):
    store.upsert(ids=["a"], embeddings=[[1.0, 0.0]], documents=["x"], metadatas=[{"type": "text"}])
    results = store.query(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert results[0]["id"] == "a"


def test_upsert_without_metadata_is_allowed(store: VectorStore):
    """ChromaDB rejects an empty metadatas list that does not match the ids."""
    store.upsert(ids=["a", "b"], embeddings=[[1.0, 0.0], [0.0, 1.0]], documents=["x", "y"])
    assert store.count() == 2
    assert store.get(["a"])["metadatas"] in ([None], [{}])


def test_delete_requires_a_filter(store: VectorStore):
    with pytest.raises(StoreError):
        store.delete()


def test_delete_all_empties_the_collection(store: VectorStore):
    store.upsert(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["x", "y"],
        metadatas=[{"type": "text"}, {"type": "image"}],
    )
    assert store.delete_all() == 2
    assert store.count() == 0
    assert store.delete_all() == 0


# --------------------------------------------------------------------------- #
# Model fingerprint
#
# Every checkpoint BookRAG supports is 1024-dimensional, so an index built by
# one of them loads happily under any other and returns confident, plausible
# and wrong scores.  Nothing in the vectors themselves can catch that.
#
# Provenance is built here the honest way - one run creating the index and a
# second run returning to it - because that is the only way the state arises
# in practice.  chromadb's modify() cannot be used to fake it: it rejects any
# payload carrying hnsw:space and replaces the rest of the mapping, so it can
# neither add a key nor keep the metric.
# --------------------------------------------------------------------------- #
def test_a_new_collection_is_stamped_with_the_model(store: VectorStore) -> None:
    from bookrag.Rag_app.store import MODEL_FINGERPRINT_KEY

    assert store.collection.metadata[MODEL_FINGERPRINT_KEY] == store.settings.model_id


def test_stamping_keeps_the_distance_metric(store: VectorStore) -> None:
    assert store.collection.metadata["hnsw:space"] == store.settings.hnsw_space


def test_a_matching_fingerprint_is_accepted(store: VectorStore) -> None:
    assert store.connect().collection is not None


def test_a_mismatched_fingerprint_stops_the_run(settings) -> None:
    from bookrag.errors import StoreError
    from bookrag.Rag_app.settings import AppSettings
    from bookrag.Rag_app.store import VectorStore

    # An earlier run built the index with CLIP v2 ...
    VectorStore(AppSettings(model_id="jinaai/jina-clip-v2")).connect()

    # ... and the configured checkpoint has since changed.
    settings.model_id = "jinaai/jina-embeddings-v5-omni-small"
    with pytest.raises(StoreError) as excinfo:
        VectorStore(settings).connect()

    assert "jinaai/jina-clip-v2" in str(excinfo.value)
    # Naming the wall is not enough; the message has to name the way out.
    assert "--model jinaai/jina-clip-v2" in (excinfo.value.install_hint or "")


def test_an_unstamped_populated_collection_is_refused(settings, store: VectorStore) -> None:
    from bookrag.errors import StoreError
    from bookrag.Rag_app.store import MODEL_FINGERPRINT_KEY, VectorStore

    store.upsert(ids=["a"], embeddings=[[1.0, 0.0]], documents=["hello"])

    # An index built before BookRAG recorded provenance at all.
    store.collection.metadata.pop(MODEL_FINGERPRINT_KEY, None)

    with pytest.raises(StoreError, match="does not record which checkpoint"):
        VectorStore(settings).connect()


def test_an_unstamped_but_empty_collection_is_adopted(settings, store: VectorStore) -> None:
    from bookrag.Rag_app.store import MODEL_FINGERPRINT_KEY, VectorStore

    store.collection.metadata.pop(MODEL_FINGERPRINT_KEY, None)

    adopted = VectorStore(settings).connect()

    # Asserted on the store that did the adopting: the rebuild leaves the
    # fixture holding the collection object that was replaced.
    assert adopted.collection.metadata[MODEL_FINGERPRINT_KEY] == settings.model_id
    assert adopted.collection.metadata["hnsw:space"] == settings.hnsw_space


def test_a_failed_connect_leaves_the_store_retryable(settings) -> None:
    from bookrag.errors import StoreError
    from bookrag.Rag_app.settings import AppSettings
    from bookrag.Rag_app.store import VectorStore

    VectorStore(AppSettings(model_id="jinaai/jina-clip-v2")).connect()
    settings.model_id = "jinaai/jina-embeddings-v5-omni-small"

    # A second attempt must re-run the check rather than short-circuit on a
    # collection the failed one had already opened.
    for _ in range(2):
        with pytest.raises(StoreError):
            VectorStore(settings).connect()


def test_reset_leaves_the_collection_stamped(store: VectorStore) -> None:
    from bookrag.Rag_app.store import MODEL_FINGERPRINT_KEY

    store.upsert(ids=["a"], embeddings=[[1.0, 0.0]], documents=["hello"])
    store.reset()

    # The verification above has no answer for an index with no provenance, so
    # a reset that recreated the collection without it would leave every later
    # command refusing to run - which is what happened before the metadata was
    # factored into one place.
    assert store.collection.count() == 0
    assert store.collection.metadata[MODEL_FINGERPRINT_KEY] == store.settings.model_id
    assert store.collection.metadata["hnsw:space"] == store.settings.hnsw_space


def test_the_ingest_path_stays_verifiable(settings) -> None:
    from bookrag.errors import StoreError
    from bookrag.Rag_app.settings import AppSettings
    from bookrag.Rag_app.store import VectorStore

    # The sequence every real ingest takes: reset, write, then come back.
    written = VectorStore(AppSettings(model_id="jinaai/jina-clip-v2"))
    written.connect()
    written.reset()
    written.upsert(ids=["a"], embeddings=[[1.0, 0.0]], documents=["hello"])

    VectorStore(AppSettings(model_id="jinaai/jina-clip-v2")).connect()

    settings.model_id = "jinaai/jina-embeddings-v5-omni-small"
    with pytest.raises(StoreError):
        VectorStore(settings).connect()


def test_reset_works_on_an_index_the_check_refuses(settings) -> None:
    from bookrag.Rag_app.settings import AppSettings
    from bookrag.Rag_app.store import MODEL_FINGERPRINT_KEY, VectorStore

    # An index built by CLIP v2 ...
    VectorStore(AppSettings(model_id="jinaai/jina-clip-v2")).connect()

    # ... and a configured checkpoint that refuses to read it.  The refusal
    # tells the user to run `bookrag reset`, so that command had better be able
    # to open the collection it is being asked to delete.
    settings.model_id = "jinaai/jina-embeddings-v5-omni-small"
    store = VectorStore(settings)
    store.reset()

    assert store.collection.count() == 0
    # Rebuilt for the checkpoint that will refill it, not the one that built it.
    assert store.collection.metadata[MODEL_FINGERPRINT_KEY] == settings.model_id
    assert store.collection.metadata["hnsw:space"] == settings.hnsw_space


def test_the_fix_hint_survives_the_count_wrapper(settings) -> None:
    from bookrag.errors import StoreError
    from bookrag.Rag_app.settings import AppSettings
    from bookrag.Rag_app.store import VectorStore

    VectorStore(AppSettings(model_id="jinaai/jina-clip-v2")).connect()
    settings.model_id = "jinaai/jina-embeddings-v5-omni-small"

    # stats() reaches the store through count(), which wraps chromadb failures
    # as "Count failed: ...".  An error that already names its remedy must come
    # through that wrapper intact, hint included.
    with pytest.raises(StoreError) as excinfo:
        VectorStore(settings).stats()

    assert "Count failed" not in str(excinfo.value)
    assert "--model jinaai/jina-clip-v2" in (excinfo.value.install_hint or "")
