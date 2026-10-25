"""Integration check for the model fingerprint against the REAL chromadb.

The pytest suite runs the same scenarios against `tests/fakes.py`, but that
fake has already been proven wrong once: its `modify()` semantics differed
from chromadb's, and a probe written against it wiped `hnsw:space` from a real
index.  So the guard that stops one checkpoint's vectors being searched by
another's is re-checked here against the library that actually ships.

Each scenario gets its own collection, because a stamped collection cannot
change its mind.  No model is loaded.  Run it manually:

    python scripts/verify_fingerprint.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

scratch = Path(tempfile.mkdtemp(prefix="bookrag-fingerprint-"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bookrag.errors import StoreError  # noqa: E402
from bookrag.Rag_app.settings import AppSettings  # noqa: E402
from bookrag.Rag_app.store import MODEL_FINGERPRINT_KEY, VectorStore  # noqa: E402

CLIP = "jinaai/jina-clip-v2"
OMNI = "jinaai/jina-embeddings-v5-omni-small"

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


def open_store(collection: str, model_id: str) -> VectorStore:
    """Open a collection under one checkpoint, the way the CLI does."""
    settings = AppSettings(collection_name=collection, model_id=model_id)
    return VectorStore(settings, path=scratch / "db").connect()


try:
    # --- a collection created under a known checkpoint -------------------- #
    store = open_store("stamped", OMNI)
    meta = dict(store.collection.metadata or {})
    check("new collection stamped", meta.get(MODEL_FINGERPRINT_KEY) == OMNI, str(meta))
    check("metric survives stamping", meta.get("hnsw:space") == "cosine", str(meta))
    check("matching fingerprint accepted", open_store("stamped", OMNI).collection is not None)

    # --- the same collection under a different checkpoint ------------------ #
    # "stamped" was built by omni, so opening it with clip is the mismatch,
    # and the hint must name omni - the checkpoint to pass back with --model.
    try:
        open_store("stamped", CLIP)
        check("mismatch refused", False, "connect() did not raise")
    except StoreError as exc:
        hint = getattr(exc, "install_hint", "")
        check("mismatch refused", True, str(exc)[:60])
        check("hint names the stored checkpoint", f"--model {OMNI}" in hint, hint[-75:])
        check("hint not swallowed into str()", not str(exc).startswith("("), str(exc)[:45])

    # --- an index built before provenance was recorded -------------------- #
    # Built through chromadb directly with no fingerprint, to stand for a
    # pre-upgrade index.  Mutating collection.metadata would not do: that dict
    # is a local copy and chromadb re-reads the persisted value.
    import chromadb

    raw = chromadb.PersistentClient(path=str(scratch / "db"))
    legacy = raw.create_collection("legacy", metadata={"hnsw:space": "cosine"})
    legacy.add(ids=["a"], embeddings=[[1.0, 0.0]], documents=["hello"])
    check(
        "legacy index has no provenance",
        "bookrag:model_id" not in dict(legacy.metadata or {}),
        str(dict(legacy.metadata or {})),
    )

    try:
        open_store("legacy", CLIP)
        check("unstamped+populated refused", False, "connect() did not raise")
    except StoreError as exc:
        check("unstamped+populated refused", "does not record" in str(exc), str(exc)[:55])

    # An empty collection of unknown provenance is safe to adopt: there are
    # no vectors to be inconsistent with.
    legacy.delete(ids=["a"])
    adopted = open_store("legacy", CLIP)
    meta = dict(adopted.collection.metadata or {})
    check("empty unstamped adopted", meta.get(MODEL_FINGERPRINT_KEY) == CLIP, str(meta))
    check("adoption keeps metric", meta.get("hnsw:space") == "cosine", str(meta))

    # --- the adopted collection is genuinely usable ------------------------ #
    adopted.upsert(
        ids=["b"], embeddings=[[0.0, 1.0]], documents=["round trip"], metadatas=[{"page": 1}]
    )
    hits = adopted.query([0.0, 1.0], top_k=1)
    check("records round-trip after adoption", hits[0]["document"] == "round trip", str(hits)[:70])

    # --- provenance written at creation survives a later reconnect --------- #
    meta = dict(open_store("stamped", OMNI).collection.metadata or {})
    check("fingerprint persists", meta.get(MODEL_FINGERPRINT_KEY) == OMNI, str(meta))
finally:
    shutil.rmtree(scratch, ignore_errors=True)

width = max(len(name) for name, _, _ in results)
for name, passed, detail in results:
    print(f"  {'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")

failed = [name for name, passed, _ in results if not passed]
print(f"\n{len(results) - len(failed)}/{len(results)} passed on real chromadb")
if failed:
    sys.exit(1)
