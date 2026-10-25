# AGENTS.md - Project Memory

Context for anyone (human or AI) working on this repository.

## What this project is

BookRAG is a multimodal retrieval-augmented-generation app for books, papers
and PDFs. It embeds **text and images into one shared vector space** - by
default with `jinaai/jina-embeddings-v5-omni-small`, with
`jinaai/jina-clip-v2` still fully supported per run via `--model` - and
persists them in **ChromaDB** with cosine similarity.

The defining property: because both modalities share a space, a text query can
return a *figure* and an image query can return the *paragraph* that explains
it. Any change must preserve that shared space.

Because both checkpoints emit 1024-d vectors, their indexes are **not**
interchangeable: `store._verify_model_fingerprint` records which checkpoint
built a collection and refuses any command that would search it with a
different one.

## Origin

Started from a single script that embedded two text chunks and two images
with Jina CLIP v2, stored them in ChromaDB, and ran one text query. That script
was the proof of concept; this repo is the productionised version of it. The
`bookrag demo` command reproduces the original behaviour (same sample
content, same query) so the transformation stays verifiable.

## Architecture in one paragraph

`embedder.py` (model) → `store.py` (persistence) → `retriever.py` (search) →
`pipeline.py` (facade) → `cli.py` (interface). `ingest.py` handles parsing and
chunking and is deliberately free of model/database dependencies so it can be
unit-tested in isolation.

## Hard-won constraints

1. **Cosine is mandatory.** Both checkpoints return L2-normalised vectors. The
   collection is created with `hnsw:space=cosine` and the app normalises
   again on the way out. Changing the space silently breaks scoring.
2. **Jina needs `trust_remote_code=True`.** Its modelling code is not in
   `transformers`. This is set in `AppSettings.trust_remote_code` and must
   stay enabled.
3. **`transformers` must stay `>=4.57,<5.0`.** CLIP v2's remote code targets
   the v4 API, v5-omni's requires at least 4.57. `peft` is a hard dependency
   of v5-omni's remote modelling code - loading it without `peft` fails.
4. **ChromaDB metadata only accepts scalars** (`str`, `int`, `float`, `bool`).
   `store.clean_metadata` coerces everything else; never bypass it.
5. **ChromaDB requires a string `document` for every vector.** Image records
   store their caption as the document.
6. **`collection.add` raises on duplicate ids.** The app uses `upsert` with
   content-hash ids so re-ingestion is idempotent.
7. **Every record id is `SHA-256(kind|source|page|content)`.** Changing this
   format orphans existing records in a persisted database.
8. **Heavy imports are lazy.** `torch`, `transformers` and `chromadb` are
   imported inside functions. This keeps the CLI responsive and lets the test
   suite run without the ML stack. Do not hoist these to module level.
9. **The model API is `encode_text` / `encode_image` for CLIP v2 and `embed`
   for v5-omni, not `get_text_embeddings`.**
   `JinaCLIPModel` (see `jinaai/jina-clip-implementation/modeling_clip.py`)
   exposes `encode_text`, `encode_image`, `get_text_features` and
   `get_image_features`. It does **not** expose `get_text_embeddings` /
   `get_image_embeddings` - the original proof-of-concept script called those
   and would raise `AttributeError` on the current checkpoint.
   `embedder._detect_api()` probes the loaded model and picks the newest API
   available; the `embed` branch is checked **last** so the detection order
   every existing checkpoint relies on is untouched. Keep that fallback chain
   intact.
10. **Asymmetric retrieval, expressed two ways.** CLIP v2 takes a task
    instruction resolved from its instruction table
    (`retrieval.query` -> "Represent the query for retrieving evidence
    documents: "). Documents are indexed with `index_task=""` (none) and
    queries use `query_task="retrieval.query"`. `get_instruction_from_task`
    raises `ValueError` for unknown task names, so the embedder validates the
    task before use and falls back to no instruction. v5-omni dropped the
    instruction table for fixed role prefixes (`Query: ...` / `Document:
    ...`); the same two settings drive it, so a non-empty `query_task` means
    "query" and an empty `index_task` means "document" on both checkpoints.
    Neither setting changes when you switch with `--model`.
11. **`torchvision`, `timm` and `einops` are hard dependencies.** Jina's
    remote code imports `torchvision.transforms` (image pipeline), `timm` and
    `einops` (EVA02 vision tower) eagerly - `transformers` refuses to load
    the model without them. `apex`, `deepspeed` and `xformers` are optional
    speed-ups. Omitting `timm`/`einops` breaks model loading only, which is
    easy to miss because text-only paths look fine.
12. **The checkpoint is bfloat16.** `AutoModel.from_pretrained` defaults to
    `torch_dtype="auto"`, so the model arrives in bf16 - slow on CPU.
    `embedder.resolve_dtype` downgrades to fp32 off-GPU.
13. **The text tower supports 8192 tokens** but Jina's own `encode_text`
    default is 512. `max_text_length` defaults to 512 to match.
14. **Monitoring must never raise or alter control flow.** `monitoring.py` is
    best-effort: it runs from `finally` blocks inside `stage()`, so an exception
    escaping it would *replace* whatever the instrumented code was raising.
    Log emission and detail formatting are therefore suppressed, and the monitor
    logger has `propagate=False` with its own `monitoring.log` handler so its
    lines never contaminate stdout (which `--json` owns) or `bookrag.log`.
15. **Stage timings are either containers or leaves.** `run.total`,
    `ingest.total`, `ingest.store` and `query.total` are containers: shown in
    the report but excluded from `work_total_s`. Everything else is a leaf and
    is summed to produce the coverage figure. **A leaf must never contain
    another leaf** - otherwise the same wall-clock is billed twice. Two places
    exist specifically to preserve this: `embedder` calls `self.load()` *outside*
    the `embed.*` timers, and `store` hoists `self.collection` out of the
    `store.upsert/query/delete` timers so a lazy `store.connect` is not charged
    to them. If you add timing around code that touches either, keep them
    siblings.

16. **An index is bound to the checkpoint that built it.** `store.py` records
    `bookrag:model_id` in the collection metadata and checks it on every
    command that reaches the store (`_verify_model_fingerprint`). Recorded and
    matching -> proceed; recorded and different -> refuse naming the stored
    checkpoint (so `--model <stored>` is the recovery); recorded nowhere and
    populated -> refuse (unknown provenance); recorded nowhere and empty ->
    adopt and stamp. **`collection.modify()` cannot stamp an existing
    collection**: chromadb rejects a payload containing `hnsw:space` and
    replaces the rest of the mapping outright, so the two keys cannot be
    written together that way - `_stamp_fingerprint` therefore rebuilds the
    (necessarily empty) collection instead. Proven against real chromadb,
    not just `tests/fakes.py`: the fake's `modify()` misled us once and a
    probe wiped `hnsw:space` from the real index.
17. **v5-omni's processor is assembled by hand.** `embedder._build_omni_processor`
    constructs `Qwen3VLProcessor(tokenizer, image_processor, Qwen3VLVideoProcessor())`
    because transformers 4.57's auto video registry does not recognise
    `model_type: jina_embeddings_v5_omni`. Deleting `video_preprocessor_config.json`
    is **not** sufficient. Loading with `modality="vision"` skips the audio
    tower: 1.07 B parameters in use, not the 1.74 B headline.
18. **Windows cannot materialise the HF symlink cache.** `snapshot_download`
    with the default cache fails with WinError 1314; `_materialise` downloads
    to a real directory instead
    (`%LOCALAPPDATA%\BookRAG\models\v5-omni-small`, ~3.1 GB). Don't "simplify"
    it back to a plain `from_pretrained(hub_id)`.

## Verified behaviour

These were confirmed by running the code, not by reading docs
(`scripts/verify_real_model.py`, `scripts/verify_real_backends.py`):

- `JinaCLIPModel` reports `api=encode` - i.e. `encode_text` / `encode_image`
  are present and are the path the app takes.
- `jinaai/jina-embeddings-v5-omni-small` reports `api=embed`, loads in 12.0 s
  on XPU/bf16 (vs 22.0 s for CLIP v2), and reaches 73.1 text items/s and 6.6
  image items/s at batch 32 (vs 28.8 and ~3.1 for CLIP v2). Asymmetric ranking
  was checked directly: a matching query scored +0.77 over an unrelated one.
- Text and image embeddings come back as `(n, 1024)` with unit norm.
- The two modalities genuinely share a space: a rendered line graph scored
  highest against the caption *"A line graph showing the linear relationship
  between temperature and pressure."* (cosine 0.268) and lowest against the
  unrelated thermodynamics caption (0.078).
- The `retrieval.query` task instruction changes the vector substantially
  (L2 distance 0.90 between the plain and instructed embeddings).
- ChromaDB 1.5.9 honours `metadata={"hnsw:space": "cosine"}`; a query
  vector `[0.7, 0.7]` against a stored `[1, 0]` returns distance 0.2929,
  which is `1 - cos(45 deg)` and therefore confirms cosine, not L2.
- PyMuPDF 2.0 exposes `pymupdf` (not just `fitz`); both are handled.

## Invariants the tests protect

- Re-ingesting a source replaces its records instead of duplicating them.
- `chunk_text` never exceeds `size` and preserves paragraph/sentence seams.
- Overlap content is genuinely repeated between consecutive chunks.
- `clean_metadata` drops `None` and stringifies containers.
- Score is `1 - distance` (so `1.0` means identical).
- The CLI turns domain errors into `error: ...` on stderr, not tracebacks.
- Container stages never enter `work_total_s`, and coverage is capped at 1.0.
- Monitoring failures never propagate: a broken `__str__` or a failing logger
  still leaves the aggregates correct.
- `--profile` before *or* after a subcommand prints the report, and with
  `--json` the report moves to stderr so stdout stays parseable.
- A collection stamped by one checkpoint is refused by another, and the hint
  names the *stored* checkpoint (so `--model <stored>` is the way back). An
  unstamped but populated collection is refused outright; an unstamped empty
  one is adopted and stamped. `hnsw:space` survives every one of those paths.

## Conventions

- `src/` layout, `pyproject.toml` with setuptools.
- `from __future__ import annotations` at the top of every module.
- Type hints everywhere; `mypy --strict`-ish discipline is expected.
- Errors: a small hierarchy in `errors.py`, always raised as
  `BookRagError` subclasses so the CLI can catch one type.
- Paths: never hardcode; use `paths.py` helpers so `BOOKRAG_HOME` works.
- Logging: `get_logger(__name__)`; no `print()` outside `cli.py` and `demo.py`.
- Docstrings explain *why*, not *what*.

## Testing approach

`tests/fakes.py` provides an in-memory ChromaDB collection and a deterministic
hash embedder. `tests/conftest.py` injects the fake `chromadb` module into
`sys.modules` before the app imports it, and points `BOOKRAG_HOME` at a temp
directory so tests stay hermetic.

**The full suite must pass with only `pytest`, `numpy` and `pillow`
installed.** If a new test needs the real model, it must be marked and skipped
by default.

Four verification scripts cover the real libraries and are *not* part of the
pytest suite (`verify_fingerprint.py` loads no model; the rest need one):

| Script | Covers |
| --- | --- |
| `scripts/verify_real_backends.py` | Real ChromaDB 1.x + PyMuPDF 2.x: upsert, filter, delete, stats, persistence across restarts. |
| `scripts/verify_real_model.py` | The real Jina checkpoint (v5-omni by default): API detection, vector shapes and norms, cross-modal similarity, task/role instructions, ChromaDB round trip. |
| `scripts/verify_fingerprint.py` | Real ChromaDB: the model-fingerprint refusals and adoption rules that stop one checkpoint's vectors being searched by another's. |
| `scripts/verify_e2e.py` | The installed `bookrag` CLI as a user drives it: ingest a PDF, text and image queries, cross-modal retrieval in both directions, idempotent re-ingest, reset. |

`tests/test_real_model.py` mirrors the model checks as a marked module
(`pytest -m real_model`) and skips itself when torch/transformers are absent.

`scripts/verify_e2e.py` needs the real model and runs about a dozen CLI
invocations, so it is *not* part of pytest. It builds its corpus from
`scripts/make_sample_pdf.py` (a four-page sample book with a known layout) and
runs against a temporary `BOOKRAG_HOME`, so it never touches real data.

## Environment

- Python 3.10+ (the machine has 3.11, 3.13 and 3.14; use `py -3.11`).
- `pip install -r requirements.txt` then `pip install -e .` for the `bookrag`
  command. `requirements.txt` carries `transformers>=4.57,<5.0` and `peft`;
  both are needed by v5-omni.
- `auto` device picks `cuda` > `xpu` > `mps` > `cpu`; this machine runs the
  model on Intel Arc XPU in bf16.
- v5-omni weights are materialised as real files at
  `%LOCALAPPDATA%\BookRAG\models\v5-omni-small` (~3.1 GB), not in the HF
  symlink cache (see constraint 18).
- Logs land in `<project>/Log` when running from a checkout, and in
  `<data dir>/logs` otherwise (`BOOKRAG_LOG_DIR` overrides both).

## Common tasks

```bash
bookrag doctor                                  # what is installed?
bookrag demo                                    # end-to-end smoke test
bookrag ingest ./books/                         # index a folder
bookrag query "..." -k 8                        # search
bookrag query "..." --model jinaai/jina-clip-v2 # search with the legacy checkpoint
bookrag find-image ./fig.png                    # image search
bookrag settings --set top_k=10 --save          # persist config
bookrag ingest ./books/ --profile               # index + show stage timings
bookrag query "..." --profile -k 8              # search + show stage timings
pytest -q                                       # run the suite
ruff check src tests                            # lint

make verify-e2e                                 # real CLI run (needs the model)
make sample-pdf                                 # build the corpus it uses
make export-diagrams                            # refresh docs/*.png|svg|pdf
```

## Where things live

| Concern | File |
| --- | --- |
| Model loading, batching, device, v5-omni adapter | `src/bookrag/accelerator/embedder.py` |
| ChromaDB access, metadata coercion, model fingerprint | `src/bookrag/Rag_app/store.py` |
| PDF parsing, chunking, image extraction | `src/bookrag/Rag_app/ingest.py` |
| Query execution, score mapping | `src/bookrag/Rag_app/retriever.py` |
| Ingest/query orchestration | `src/bookrag/Rag_app/pipeline.py` |
| Stage timings, model runtime, `monitoring.log` | `src/bookrag/logs/monitoring.py` |
| Log file routing (`Log/` precedence) | `src/bookrag/logs/logging_setup.py` |
| CLI parsing and output | `src/bookrag/Rag_app/cli.py` |
| Configuration schema | `src/bookrag/Rag_app/settings.py` |
| Data locations | `src/bookrag/data/paths.py` |
| draw.io source and exports | `docs/bookrag-architecture.drawio` |
| Regenerating the diagrams | `scripts/export_drawio.py` |
| Sample PDF used by `verify-e2e` | `scripts/make_sample_pdf.py` |

## Possible next steps

- A captioning model (BLIP) to replace the placeholder figure captions with
  real descriptions - this would materially improve image-to-text retrieval.
- Hybrid search (BM25 + dense) via `chromadb.utils.query_products`.
- An HTTP API layer (`FastAPI`) wrapping `BookRagPipeline`.
- OCR for scanned PDFs (Tesseract / `ocrmypdf`) before chunking.
- A generation step so the retrieved chunks are fed to an LLM with citations.
