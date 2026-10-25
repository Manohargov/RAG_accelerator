"""Tests for the retriever and the end-to-end pipeline."""

from __future__ import annotations

import threading

import pytest

from bookrag.errors import QueryError, StoreError
from bookrag.Rag_app.pipeline import BookRagPipeline
from bookrag.Rag_app.retriever import Retriever
from bookrag.Rag_app.store import VectorStore


def test_search_text_returns_hits(pipeline: BookRagPipeline):
    pipeline.ingest_text("Kinetic energy is E = 1/2 m v squared.", source="physics")
    hits = pipeline.search_text("kinetic energy", top_k=3)
    assert hits
    assert hits[0].kind == "text"
    assert 0.0 <= hits[0].score <= 1.0


def test_search_text_rejects_empty_query(pipeline: BookRagPipeline):
    with pytest.raises(QueryError):
        pipeline.search_text("   ")


def test_search_image_requires_existing_file(pipeline: BookRagPipeline):
    with pytest.raises(QueryError):
        pipeline.search_image("does-not-exist.png")


def test_search_requires_text_or_image(pipeline: BookRagPipeline):
    with pytest.raises(QueryError):
        pipeline.search()


def test_type_filter_restricts_results(pipeline: BookRagPipeline):
    pipeline.ingest_text("Some text passage.", source="notes")
    hits = pipeline.search_text("passage", top_k=5, filters={"type": "image"})
    assert hits == []


def test_reingest_replaces_old_records(pipeline: BookRagPipeline):
    pipeline.ingest_text("Original content about magnets.", source="book")
    first = pipeline.store.count()
    assert first > 0

    pipeline.ingest_text("Completely different content about optics.", source="book")
    second = pipeline.store.count()
    assert second == first, "re-ingesting a source must not duplicate records"


def test_ingest_text_uses_stable_ids(pipeline: BookRagPipeline):
    pipeline.ingest_text("Stable content.", source="book")
    ids_first = set(pipeline.store.get()["ids"])
    pipeline.ingest_text("Stable content.", source="book")
    assert set(pipeline.store.get()["ids"]) == ids_first


def test_ingest_images_stores_records(pipeline: BookRagPipeline, tmp_path):
    from PIL import Image

    image = tmp_path / "figure.png"
    Image.new("RGB", (64, 64), color="lightgreen").save(image)

    report = pipeline.ingest_images([image], descriptions=["A green square"])
    assert report.images == 1
    hits = pipeline.search_image(image, top_k=3)
    assert hits


def test_two_images_with_the_same_caption_do_not_collide(pipeline: BookRagPipeline, tmp_path):
    """Figures on one page share a caption; their ids must still be distinct."""
    from PIL import Image

    from bookrag.Rag_app.ingest import ExtractedImage, SourceDocument

    paths = []
    for name in ("a.png", "b.png"):
        path = tmp_path / name
        Image.new("RGB", (64, 64), color="lightgreen").save(path)
        paths.append(path)

    images = [
        ExtractedImage(path=path, page=3, index=index, description="Figure on page 3")
        for index, path in enumerate(paths)
    ]
    pipeline.ingest_source(SourceDocument(source="book.pdf", images=images))

    stored = pipeline.store.get()
    assert len(stored["ids"]) == 2, "one figure overwrote the other"


def test_search_hit_as_dict_is_serialisable(pipeline: BookRagPipeline):
    pipeline.ingest_text("A passage worth finding.", source="book")
    payload = pipeline.search_text("passage", top_k=1)[0].as_dict()
    assert set(payload) == {"id", "score", "distance", "document", "metadata"}


def test_retriever_uses_injected_components(settings, fake_embedder, store):
    retriever = Retriever(fake_embedder, store, settings)
    retriever.store.upsert(
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        documents=["hello"],
        metadatas=[{"type": "text"}],
    )
    hits = retriever.search_text("hello", top_k=1)
    assert hits[0].document == "hello"


def test_stats_reflect_ingested_content(pipeline: BookRagPipeline):
    pipeline.ingest_text("Another passage for statistics.", source="book")
    stats = pipeline.stats()
    assert stats["total"] >= 1
    assert "book" in stats["sources"]


def test_pipeline_stats_without_data(store: VectorStore):
    from bookrag.Rag_app.pipeline import BookRagPipeline

    pipeline = BookRagPipeline(store=store)
    assert pipeline.stats()["total"] == 0


# --------------------------------------------------------------------------- #
# The encode/write pipeline
# --------------------------------------------------------------------------- #
def _document(count: int, source: str = "book.pdf"):
    """A SourceDocument holding exactly ``count`` text chunks."""
    from bookrag.Rag_app.ingest import Chunk, SourceDocument

    chunks = [
        Chunk(
            text=f"passage {index} describing the kinetic energy of a body. ",
            index=index,
            metadata={"type": "text", "source": source},
        )
        for index in range(count)
    ]
    return SourceDocument(source=source, chunks=chunks, page_count=1)


def test_every_batch_is_written_before_ingest_returns(pipeline: BookRagPipeline):
    # 8 chunks at batch_size=4 is two batches, so the second is only queued
    # when _store_chunks returns - the writer must drain before we report.
    report = pipeline.ingest_source(_document(8))

    assert report.text_chunks == 8
    assert pipeline.store.count() == 8, "the final write had not drained"


def test_record_ids_do_not_depend_on_batch_size(pipeline: BookRagPipeline):
    document = _document(6)

    pipeline.settings.batch_size = 1
    pipeline.ingest_source(document)
    one_at_a_time = set(pipeline.store.get()["ids"])

    pipeline.settings.batch_size = 999
    pipeline.ingest_source(document)
    all_at_once = set(pipeline.store.get()["ids"])

    assert len(one_at_a_time) == 6
    assert one_at_a_time == all_at_once, "splitting the work changed what was stored"


def test_batch_size_is_never_zero(pipeline: BookRagPipeline):
    # range() would raise on a step of 0; settings validation normally
    # prevents this, but the guard belongs where the loop is.
    pipeline.settings.batch_size = 0
    assert pipeline._batch_size() >= 1


def test_a_failed_write_surfaces_and_does_not_hang(pipeline: BookRagPipeline, monkeypatch):
    def explode(**kwargs):
        raise StoreError("disk is full")

    monkeypatch.setattr(pipeline.store, "upsert", explode)

    with pytest.raises(StoreError, match="disk is full"):
        pipeline.ingest_source(_document(8))


def test_next_batch_encodes_while_the_previous_write_is_in_flight(
    pipeline: BookRagPipeline,
):
    """The point of the rewrite: the encoder must not block on ChromaDB."""
    write_begun = threading.Event()
    release_write = threading.Event()
    real_upsert = pipeline.store.upsert
    real_embed = pipeline.embedder.embed_texts

    def gated_upsert(**kwargs):
        write_begun.set()
        if not release_write.wait(timeout=10):
            raise TimeoutError("the previous write was never released")
        return real_upsert(**kwargs)

    embed_calls: list[list[str]] = []

    def recording_embed(texts, task=None):
        embed_calls.append(list(texts))
        if len(embed_calls) == 2:
            # Encoding batch 2 while batch 1's write is parked on the gate
            # means the two are genuinely running at the same time. Had
            # add() blocked, we would never have reached this call and the
            # test would fail on the timeout instead.
            assert write_begun.wait(timeout=5), "the first write never started"
            release_write.set()
        return real_embed(texts, task=task)

    pipeline.store.upsert = gated_upsert
    pipeline.embedder.embed_texts = recording_embed

    report = pipeline.ingest_source(_document(8))

    assert len(embed_calls) == 2, "encoding did not resume before the write finished"
    assert report.text_chunks == 8
    assert pipeline.store.count() == 8
