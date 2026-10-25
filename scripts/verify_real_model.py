"""End-to-end check against the REAL Jina CLIP v2 model.

This is the highest-value verification: it proves that the embedding API
usage, the task instructions, the shared vector space and the ChromaDB
round-trip all work with the actual checkpoint.

    python scripts/verify_real_model.py

Requires the full ``requirements.txt`` install.  The first run downloads
~1.8 GB of weights from HuggingFace.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bookrag.accelerator.embedder import ClipEmbedder  # noqa: E402
from bookrag.Rag_app.settings import AppSettings  # noqa: E402
from bookrag.Rag_app.store import VectorStore  # noqa: E402


def main() -> int:
    from PIL import Image, ImageDraw

    failures: list[str] = []

    settings = AppSettings(device="cpu", dtype="float32", batch_size=4)
    print(f"Loading {settings.model_id} ...")
    embedder = ClipEmbedder(settings).load()
    print(f"  device={embedder.device} api={embedder.api}")

    # --- 1. Text embeddings -------------------------------------------- #
    texts = [
        "The formula for kinetic energy is E = 1/2 m v^2.",
        "A line graph showing the linear relationship between temperature and pressure.",
        "A schematic diagram of a closed thermodynamic system.",
    ]
    text_vectors = embedder.embed_texts(texts, task=None)
    print(f"[text]  shape={text_vectors.shape}")
    if text_vectors.shape != (3, 1024):
        failures.append(f"unexpected text embedding shape: {text_vectors.shape}")

    norms = np.linalg.norm(text_vectors, axis=1)
    print(f"[text]  norms={np.round(norms, 4)}")
    if not np.allclose(norms, 1.0, atol=1e-3):
        failures.append(f"text embeddings are not unit length: {norms}")

    # --- 2. Image embeddings ------------------------------------------- #
    workdir = Path(tempfile.mkdtemp(prefix="bookrag-model-"))
    try:
        graph = Image.new("RGB", (400, 300), color="white")
        draw = ImageDraw.Draw(graph)
        draw.line([40, 260, 360, 40], fill="crimson", width=5)
        for step in range(5):
            x = 60 + step * 70
            y = 240 - step * 45
            draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill="navy")
        graph_path = workdir / "graph.png"
        graph.save(graph_path)

        image_vectors = embedder.embed_image_paths([graph_path])
        print(f"[image] shape={image_vectors.shape}")
        if image_vectors.shape != (1, 1024):
            failures.append(f"unexpected image embedding shape: {image_vectors.shape}")
        image_norm = float(np.linalg.norm(image_vectors[0]))
        print(f"[image] norm={image_norm:.4f}")
        if abs(image_norm - 1.0) > 1e-3:
            failures.append(f"image embedding is not unit length: {image_norm}")

        # --- 3. The shared vector space ------------------------------- #
        # The graph image should sit closest to the sentence that describes it.
        similarities = text_vectors @ image_vectors[0]
        print(f"[space] image-vs-text cosine = {np.round(similarities, 4)}")
        best = int(np.argmax(similarities))
        print(f"[space] best match: {texts[best]!r}")
        if best != 1:
            failures.append(
                f"the graph image matched {texts[best]!r} instead of the line-graph caption"
            )

        # --- 4. Task instructions ------------------------------------- #
        plain = embedder.embed_text("kinetic energy", task=None)
        instructed = embedder.embed_text("kinetic energy", task="retrieval.query")
        delta = float(np.linalg.norm(plain - instructed))
        print(f"[task]  ||plain - instructed|| = {delta:.4f}")
        if delta < 1e-4:
            failures.append("the task instruction had no effect on the embedding")

        # --- 5. Round trip through ChromaDB --------------------------- #
        store = VectorStore(settings, path=workdir / "db").connect()
        store.upsert(
            ids=["t0", "t1", "t2", "i0"],
            embeddings=np.vstack([text_vectors, image_vectors]).tolist(),
            documents=texts + ["a line graph"],
            metadatas=[
                {"type": "text", "source": "verify"},
                {"type": "text", "source": "verify"},
                {"type": "text", "source": "verify"},
                {"type": "image", "source": "verify"},
            ],
        )
        print(f"[chroma] count={store.count()}")

        hits = store.query(embedder.embed_text("a graph of pressure against temperature"), top_k=4)
        for hit in hits:
            print(f"[chroma] {hit['id']} score={hit['score']:.4f} {hit['document'][:52]!r}")
        if not hits:
            failures.append("the query returned nothing")

        # The image query should retrieve the figure itself.
        image_hits = store.query(embedder.embed_image(graph_path), top_k=4)
        print(f"[chroma] image query top id = {image_hits[0]['id']} (expected i0)")
        if image_hits[0]["id"] != "i0":
            failures.append(f"image query returned {image_hits[0]['id']} instead of i0")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nAll real-model checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
