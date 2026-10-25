"""Integration check against the REAL chromadb and pymupdf.

This script is not part of the pytest suite (the suite uses fakes so it can
run without the ML stack).  Run it manually to prove the production code
paths work against the actual libraries:

    python scripts/verify_real_backends.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bookrag.Rag_app.ingest import extract_pdf  # noqa: E402
from bookrag.Rag_app.settings import AppSettings  # noqa: E402
from bookrag.Rag_app.store import VectorStore  # noqa: E402


def make_sample_pdf(path: Path) -> None:
    """Build a small PDF with text and an embedded figure."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()

    page.insert_text((72, 100), "The formula for kinetic energy is E = 1/2 m v^2.", fontsize=14)
    page.insert_text(
        (72, 130), "Pressure rises linearly with temperature at constant volume.", fontsize=14
    )
    page.insert_text(
        (72, 160), "Chapter 1 covers the fundamentals of motion and energy.", fontsize=14
    )

    # A recognisable figure (a filled circle) so image extraction has something.
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 160, 120), False)
    pix.set_rect(pymupdf.IRect(0, 0, 160, 120), (255, 255, 255))
    for y in range(0, 120, 4):
        for x in range(0, 160, 4):
            if (x - 80) ** 2 + (y - 60) ** 2 < 50**2:
                pix.set_pixel(x, y, (200, 30, 30))
    page.insert_image(pymupdf.Rect(72, 200, 232, 320), pixmap=pix)

    doc.save(str(path))
    doc.close()


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="bookrag-verify-"))
    failures: list[str] = []
    try:
        settings = AppSettings(chunk_size=300, chunk_overlap=50, min_image_pixels=100)

        # --- 1. Real PDF parsing ---------------------------------------- #
        pdf_path = workdir / "sample.pdf"
        make_sample_pdf(pdf_path)
        document = extract_pdf(pdf_path, settings, workdir / "images")
        print(
            f"[pdf]    pages={document.page_count} chunks={len(document.chunks)} images={len(document.images)}"
        )
        if document.page_count != 1:
            failures.append(f"expected 1 page, got {document.page_count}")
        if not document.chunks:
            failures.append("no text chunks extracted")
        if not document.images:
            failures.append("no images extracted")
        for image in document.images:
            if image.path and not Path(image.path).exists():
                failures.append(f"image not written to disk: {image.path}")

        # --- 2. Real ChromaDB -------------------------------------------- #
        store = VectorStore(settings, path=workdir / "db").connect()
        vectors = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
        store.upsert(
            ids=["t1", "t2", "i1"],
            embeddings=vectors,
            documents=["kinetic energy", "thermodynamics", "a figure"],
            metadatas=[
                {"type": "text", "source": "sample.pdf", "page": 1},
                {"type": "text", "source": "sample.pdf", "page": 1},
                {"type": "image", "source": "sample.pdf", "page": 1},
            ],
        )
        count = store.count()
        print(f"[chroma] count={count}")
        if count != 3:
            failures.append(f"expected 3 records, got {count}")

        results = store.query([1.0, 0.0], top_k=3)
        print(f"[chroma] top={results[0]['id']} score={results[0]['score']:.4f}")
        if results[0]["id"] != "t1":
            failures.append(f"nearest neighbour was {results[0]['id']}, expected t1")

        filtered = store.query([1.0, 0.0], top_k=5, where={"type": "image"})
        print(f"[chroma] filtered hits={len(filtered)}")
        if len(filtered) != 1:
            failures.append(f"metadata filter returned {len(filtered)} hits, expected 1")

        # Re-upserting the same ids must not duplicate rows.
        store.upsert(
            ids=["t1"],
            embeddings=[[1.0, 0.0]],
            documents=["kinetic energy v2"],
            metadatas=[{"type": "text", "source": "sample.pdf", "page": 1}],
        )
        if store.count() != 3:
            failures.append(f"upsert duplicated rows: count={store.count()}")

        stats = store.stats()
        print(f"[chroma] stats={stats['by_type']}")
        if stats["by_type"] != {"text": 2, "image": 1}:
            failures.append(f"unexpected breakdown: {stats['by_type']}")

        # Writing without any metadata must be accepted by the real client.
        store.upsert(ids=["nometa"], embeddings=[[0.3, 0.4]], documents=["no metadata"])
        if store.count() != 4:
            failures.append(f"upsert without metadata failed: count={store.count()}")
        print("[chroma] upsert without metadata ok")

        store.delete_where({"source": "sample.pdf"})
        # The metadata-less record has no "source" key, so it must survive.
        remaining = store.count()
        print(f"[chroma] after delete_where, count={remaining}")
        if remaining != 1:
            failures.append(f"delete_where left {remaining} records, expected 1")
        print("[chroma] delete_where ok")

        # --- 3. Round trip through a fresh client ------------------------ #
        store.delete_all()
        reopened = VectorStore(settings, path=workdir / "db").connect()
        reopened.upsert(
            ids=["x"],
            embeddings=[[0.1, 0.2]],
            documents=["persisted"],
            metadatas=[{"type": "text", "source": "sample.pdf"}],
        )
        again = VectorStore(settings, path=workdir / "db").connect()
        if again.count() != 1:
            failures.append(
                f"record did not persist across client restarts (count={again.count()})"
            )
        print("[chroma] persistence across restarts ok")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nAll real-backend checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
