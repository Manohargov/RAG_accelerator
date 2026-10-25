# Diagrams

Architecture diagrams for BookRAG, authored as native draw.io XML.

| File | What it is |
| --- | --- |
| `bookrag-architecture.drawio` | **Editable source.** Two pages: *Architecture* and *Data flow*. |
| `bookrag-architecture.png` / `.svg` | Page 1 — layered system architecture. |
| `bookrag-data-flow.png` / `.svg` | Page 2 — how text and figures share one vector space. |
| `bookrag-architecture.pdf` | Both pages in a single PDF. |

The PNGs are exported with **embedded diagram XML**, so they can be dragged
back into draw.io and edited. The SVG exports embed images rather than linking
them, so they render identically anywhere.

---

## Page 1 — Architecture

Four layers, top to bottom:

| Layer | Contains |
| --- | --- |
| **Interface** | `bookrag` CLI (8 commands), Python API |
| **Application** | `BookRagPipeline` (orchestration), `Retriever` (search + scoring) |
| **Core services** | `Ingest` (PDF parse + chunking), `ClipEmbedder` (Jina CLIP v2), `VectorStore` (ChromaDB) |
| **External systems & storage** | PyMuPDF, HuggingFace Hub, ChromaDB on disk, the data directory |

Solid arrows are data flow, the dashed arrow is configuration
(`AppSettings` reaching the pipeline from `settings.json` / `BOOKRAG_*` env vars).

## Page 2 — Data flow

Shows the property the whole project exists for: **text and images land in
the same 1024-d space**, so a text query can return a figure and an image
query can return the paragraph that explains it.

It also makes the asymmetric retrieval setup explicit — documents are indexed
with no task instruction, queries carry Jina's `retrieval.query` instruction.

---

## Regenerating

Requires the draw.io Desktop app (`drawio` on `PATH`).

```bash
make diagrams           # validate the .drawio source
make export-diagrams    # regenerate PNG / SVG / PDF
```

Or directly:

```bash
python scripts/validate_drawio.py docs/bookrag-architecture.drawio
python scripts/export_drawio.py docs/bookrag-architecture.drawio --prefix bookrag
```

`export_drawio.py` splits the multi-page source into one file per page
because the draw.io CLI only exports the first page for PNG and SVG, then
merges the per-page PDFs back into one document.

### Editing

1. Open `bookrag-architecture.drawio` in draw.io (or the [app.diagrams.net](https://app.diagrams.net) web editor).
2. Edit the shapes and edges.
3. Re-run `make export-diagrams` to refresh the exports.

Keep these conventions so the diagrams stay consistent:

- Layer containers are swimlanes with `startSize=30`; child coordinates are **relative** to the swimlane origin.
- Palette: blue `#dae8fc` (interface), green `#d5e8d4` (application), orange `#ffe6cc` (core services), grey `#f5f5f5` (external).
- Solid arrow = data flow, dashed = configuration.
- Never reuse the reserved cell ids `0` and `1`; every edge needs `<mxGeometry relative="1" as="geometry"/>`.
