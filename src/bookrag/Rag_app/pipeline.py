"""High-level orchestration: ingest documents, then query them.

``BookRagPipeline`` is the object the CLI (and any future HTTP API) talks to.
It owns the embedder, the store and the retriever, and it hides the
lifecycle details - the model is only loaded when the first embedding is
actually needed.

Ingestion is shaped around the accelerator rather than around the database.
Encoding is the dominant cost and is the one stage worth putting on a GPU or
an NPU, so the pipeline feeds the embedder in device-sized batches and hands
the results off to a background writer: upsert(k-1) runs while batch k is
being encoded.  See :class:`_StreamedWriter`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from ..accelerator.embedder import ClipEmbedder
from ..data import paths
from ..errors import IngestError
from ..logs.logging_setup import get_logger
from ..logs.monitoring import stage
from .ingest import (
    Chunk,
    ExtractedImage,
    SourceDocument,
    discover_documents,
    extract_pdf,
    extract_text_file,
    image_descriptions,
    make_id,
    source_key,
)
from .retriever import Retriever, SearchHit
from .settings import AppSettings
from .store import VectorStore

log = get_logger("pipeline")


@dataclass
class IngestReport:
    """Summary of a single ingestion run."""

    source: str
    text_chunks: int = 0
    images: int = 0
    skipped_images: int = 0
    pages: int = 0
    ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.text_chunks + self.images

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "text_chunks": self.text_chunks,
            "images": self.images,
            "skipped_images": self.skipped_images,
            "pages": self.pages,
            "total": self.total,
            "errors": list(self.errors),
        }


class _StreamedWriter:
    """Hands encoded batches to ChromaDB while the caller encodes the next one.

    Encoding is the expensive stage - the one an accelerator is bought for -
    while the upsert is cheap but synchronous.  Run serially, the device goes
    idle for the duration of every write.  This keeps *exactly one* write in
    flight on a worker thread, so ``upsert(k-1)`` overlaps ``encode(k)``:
    enough overlap to hide the write, bounded so a ten-thousand-chunk
    document never materialises every vector at once.

    Exactly one thread ever touches the store - this worker.  The calling
    thread only embeds, which keeps ``VectorStore``'s lazy collection creation
    and ChromaDB's own threading out of the picture entirely.  The caller must
    ``connect()`` the store before the first :meth:`add`.
    """

    def __init__(self, store: VectorStore) -> None:
        self._store = store
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bookrag-upsert")
        self._pending: Future[None] | None = None

    def __enter__(self) -> _StreamedWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        try:
            if exc_type is None:
                # Nothing is unwinding yet, so a write failure must surface.
                self._wait()
            else:
                # The caller already has the better error; don't replace it.
                with contextlib.suppress(Exception):
                    self._wait()
        finally:
            self._pool.shutdown(wait=True, cancel_futures=True)
        return False

    def add(
        self,
        *,
        ids: Sequence[str],
        embeddings: Any,
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        """Queue one prepared batch and return without waiting for it."""
        self._wait()  # keeps at most one write in flight
        self._pending = self._pool.submit(
            self._store.upsert,
            ids=list(ids),
            embeddings=embeddings,
            documents=list(documents),
            metadatas=list(metadatas),
        )

    def _wait(self) -> None:
        """Block for the in-flight write, re-raising anything it hit."""
        if self._pending is None:
            return
        pending, self._pending = self._pending, None
        # Deliberately not swallowed: a failed upsert is a failed ingestion.
        pending.result()


class BookRagPipeline:
    """The public facade of the application."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        embedder: ClipEmbedder | None = None,
        store: VectorStore | None = None,
        db_path: Path | None = None,
    ) -> None:
        self.settings = settings or AppSettings()
        self.embedder = embedder or ClipEmbedder(self.settings)
        self.store = store or VectorStore(self.settings, path=db_path)
        self._retriever: Retriever | None = None

    # ------------------------------------------------------------------ #
    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.embedder, self.store, self.settings)
        return self._retriever

    @classmethod
    def from_environment(cls, db_path: Path | None = None) -> BookRagPipeline:
        """Build a pipeline from the saved settings plus environment overrides."""
        settings = AppSettings().load()
        settings.apply_environment()
        return cls(settings, db_path=db_path)

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_document(self, path: str | Path, replace: bool = True) -> IngestReport:
        """Ingest a single PDF, text or markdown file."""
        path = Path(path)
        suffix = path.suffix.lower()

        # Container: `parse.pdf` and `ingest.store` inside it are timed separately.
        with stage("ingest.total", container=True, source=path.name):
            if suffix in {".pdf"}:
                document = extract_pdf(path, self.settings, self._image_dir())
            elif suffix in {".txt", ".md", ".markdown", ".rst"}:
                document = extract_text_file(path, self.settings)
            else:
                raise IngestError(f"Unsupported file type: {path.suffix!r}")

            return self.ingest_source(document, replace=replace)

    def ingest_path(
        self, target: str | Path, recursive: bool = True, replace: bool = True
    ) -> list[IngestReport]:
        """Ingest every supported document found at ``target``."""
        files = discover_documents(target, recursive=recursive)
        if not files:
            raise IngestError(f"No ingestible documents found at {target}.")
        return [self.ingest_document(path, replace=replace) for path in files]

    def ingest_source(self, document: SourceDocument, replace: bool = True) -> IngestReport:
        """Embed and store the contents of an already-parsed document."""
        report = IngestReport(source=document.source, pages=document.page_count)

        # Container: `embed.*` and `store.*` inside it are counted on their own,
        # so this is excluded from the timed-work sum rather than added to it.
        with stage(
            "ingest.store",
            container=True,
            source=document.source,
            chunks=len(document.chunks),
            figures=len(document.images),
        ):
            if replace:
                # Drop anything previously indexed for this source so re-ingesting
                # an edited file does not leave stale chunks behind.  This must
                # finish before the writer starts, or it would race with - and
                # potentially delete - what the pipeline has just written.
                self.store.delete_where({"source": document.source})

            # Connected on this thread so the worker never races the lazy
            # collection creation.
            self.store.connect()

            with _StreamedWriter(self.store) as writer:
                if document.chunks:
                    report.text_chunks = self._store_chunks(
                        document.chunks, document.source, writer
                    )
                if document.images:
                    stored, skipped = self._store_images(document.images, document.source, writer)
                    report.images = stored
                    report.skipped_images = skipped
            # Leaving the writer drains it: every upsert has landed by here,
            # so the report never counts a batch that did not make it.

        log.info(
            "Ingested %s: %d text chunks, %d images.",
            document.source,
            report.text_chunks,
            report.images,
        )
        return report

    def ingest_text(
        self,
        text: str,
        source: str = "inline",
        replace: bool = True,
    ) -> IngestReport:
        """Ingest a raw string of text."""
        from .ingest import chunk_text

        chunks = [
            Chunk(
                text=piece,
                index=index,
                metadata={"type": "text", "source": source},
            )
            for index, piece in enumerate(
                chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
            )
        ]
        document = SourceDocument(source=source, chunks=chunks, page_count=1)
        return self.ingest_source(document, replace=replace)

    def ingest_images(
        self,
        paths_: Sequence[str | Path],
        descriptions: Sequence[str] | None = None,
        replace: bool = True,
    ) -> IngestReport:
        """Ingest standalone image files."""
        source = source_key(paths_[0]) if paths_ else "images"
        images = [
            ExtractedImage(
                path=Path(p),
                page=0,
                index=index,
                description=descriptions[index]
                if descriptions and index < len(descriptions)
                else f"Image: {Path(p).name}",
            )
            for index, p in enumerate(paths_)
        ]
        document = SourceDocument(source=source, images=images)
        report = self.ingest_source(document, replace=replace)
        return report

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #
    def search_text(
        self,
        query: str,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        return self.retriever.search_text(query, top_k=top_k, filters=filters)

    def search_image(
        self,
        image_path: str | Path,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        return self.retriever.search_image(image_path, top_k=top_k, filters=filters)

    def search(
        self,
        text: str | None = None,
        image: str | Path | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        return self.retriever.search(text=text, image=image, top_k=top_k, filters=filters)

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        return self.store.stats()

    def reset(self) -> None:
        """Wipe the collection."""
        self.store.reset()

    def close(self) -> None:
        """Release model resources."""
        self.embedder.unload()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _image_dir(self) -> Path | None:
        if not self.settings.keep_extracted_images:
            return None
        return paths.images_dir()

    def _store_chunks(self, chunks: Sequence[Chunk], source: str, writer: _StreamedWriter) -> int:
        # Documents are indexed without a task instruction; the query side
        # carries "retrieval.query" (see AppSettings.query_task).
        task = self.settings.index_task or None
        size = self._batch_size()
        written = 0

        for start in range(0, len(chunks), size):
            group = chunks[start : start + size]
            texts = [chunk.text for chunk in group]
            # One group is one forward pass: the device is fed at a steady
            # rate instead of seeing the whole document at once, and
            # `embed.text` reports a per-batch time that means something.
            vectors = self.embedder.embed_texts(texts, task=task)

            records = [self._text_record(chunk, source) for chunk in group]
            ids = [record_id for record_id, _ in records]
            metadatas = [metadata for _, metadata in records]
            # Returns while this batch is still being written, so the next
            # `embed_texts` call overlaps it.
            writer.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
            written += len(ids)

        return written

    @staticmethod
    def _text_record(chunk: Chunk, source: str) -> tuple[str, dict[str, Any]]:
        metadata = dict(chunk.metadata)
        metadata.setdefault("type", "text")
        metadata["source"] = source
        metadata["chunk_index"] = chunk.index
        if chunk.page is not None:
            metadata["page"] = chunk.page
        metadata["chars"] = len(chunk.text)
        return make_id("text", source, chunk.text, chunk.page), metadata

    def _store_images(
        self, images: Sequence[ExtractedImage], source: str, writer: _StreamedWriter
    ) -> tuple[int, int]:
        usable = [image for image in images if image.path is not None and Path(image.path).exists()]
        skipped = len(images) - len(usable)
        if not usable:
            return 0, skipped

        size = self._batch_size()
        written = 0
        for start in range(0, len(usable), size):
            group = usable[start : start + size]
            # Decoding happens inside this call, so it overlaps the previous
            # batch's write rather than adding to the critical path.
            vectors = self.embedder.embed_image_paths([Path(image.path) for image in group])
            documents = image_descriptions(group)

            records = [
                self._image_record(image, caption, source)
                for image, caption in zip(group, documents, strict=True)
            ]
            ids = [record_id for record_id, _ in records]
            metadatas = [metadata for _, metadata in records]
            writer.add(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)
            written += len(ids)

        return written, skipped

    @staticmethod
    def _image_record(
        image: ExtractedImage, caption: str, source: str
    ) -> tuple[str, dict[str, Any]]:
        metadata = {
            "type": "image",
            "source": source,
            "page": image.page,
            "path": str(image.path),
            "width": image.width,
            "height": image.height,
            "caption": caption,
        }
        # Two figures on the same page share a caption, so the path has to
        # take part in the hash - otherwise one would overwrite the other.
        return make_id("image", source, f"{caption}|{image.path}", image.page), metadata

    def _batch_size(self) -> int:
        """Batch size for the encode/write pipeline.

        Deliberately not special-cased per device: ``settings.batch_size`` is
        the one knob worth turning when moving to an accelerator, and a
        device-blind default keeps behaviour identical until someone measures
        a reason to change it.  Guarded against 0 so ``range()`` cannot be
        asked to step by zero.
        """
        return max(1, int(self.settings.batch_size or 16))
