"""Turn source documents into embeddable records.

The pipeline is deliberately split into pure functions (``chunk_text``,
``make_id``) and I/O heavy ones (``extract_pdf``) so the interesting logic
can be unit-tested without touching the model or the database.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import DependencyError, IngestError
from ..logs.logging_setup import get_logger
from ..logs.monitoring import stage
from .settings import AppSettings

log = get_logger("ingest")

_WHITESPACE = re.compile(r"[ \t\x0b\f\r]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_ABBREVIATIONS = re.compile(
    r"\b(?:[A-Z]|Mr|Mrs|Ms|Dr|Prof|Fig|Eq|Vol|No|vs|etc|e\.g|i\.e|approx)\.$"
)

#: Extensions the PDF parser understands.
PDF_SUFFIXES = {".pdf"}


@dataclass
class Chunk:
    """A unit of text ready to be embedded."""

    text: str
    page: int | None = None
    index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class ExtractedImage:
    """An image pulled out of a source document."""

    path: Path | None = None
    page: int = 0
    index: int = 0
    width: int = 0
    height: int = 0
    description: str = ""
    raw: bytes = b""


@dataclass
class SourceDocument:
    """Everything extracted from one source file."""

    source: str
    chunks: list[Chunk] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    page_count: int = 0


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def normalise_text(text: str) -> str:
    """Collapse ragged whitespace while keeping paragraph structure."""
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _WHITESPACE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, tolerating common abbreviations."""
    if not text.strip():
        return []
    parts: list[str] = []
    buffer: list[str] = []
    for piece in _SENTENCE_END.split(text):
        buffer.append(piece)
        candidate = " ".join(buffer)
        if _ABBREVIATIONS.search(candidate.strip()):
            # Do not break after an abbreviation such as "Fig." or "e.g.".
            continue
        if candidate.strip():
            parts.append(candidate.strip())
        buffer = []
    tail = " ".join(buffer).strip()
    if tail:
        parts.append(tail)
    return parts


def _units(text: str) -> list[str]:
    """Split text into paragraph-sized units, breaking long paragraphs."""
    units: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= 2000:
            units.append(paragraph)
            continue
        current = ""
        for sentence in split_sentences(paragraph):
            if not current:
                current = sentence
            elif len(current) + 1 + len(sentence) <= 2000:
                current = f"{current} {sentence}"
            else:
                units.append(current)
                current = sentence
        if current:
            units.append(current)
    return units


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """Split ``text`` into overlapping chunks of roughly ``size`` characters.

    Chunk boundaries prefer paragraph and sentence seams so a chunk rarely
    starts mid-sentence.  Overlap is applied at the tail of the previous
    chunk to preserve context across the boundary.
    """
    if size <= 0:
        raise ValueError("size must be positive")

    cleaned = normalise_text(text)
    if not cleaned:
        return []

    overlap = max(0, min(overlap, size - 1))
    chunks: list[str] = []
    buffer = ""

    for unit in _units(cleaned):
        if len(unit) > size:
            # A single oversized unit: flush the buffer, then hard-split.
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            start = 0
            while start < len(unit):
                chunks.append(unit[start : start + size].strip())
                if start + size >= len(unit):
                    break
                start += size - overlap
            continue

        candidate = unit if not buffer else f"{buffer}\n\n{unit}"
        if len(candidate) <= size:
            buffer = candidate
            continue

        chunks.append(buffer.strip())
        tail = buffer[-overlap:] if overlap else ""
        buffer = f"{tail}\n\n{unit}" if tail else unit

    if buffer.strip():
        chunks.append(buffer.strip())

    return [chunk for chunk in chunks if chunk.strip()]


def make_id(kind: str, source: str, content: str, page: int | None = None) -> str:
    """Build a deterministic id so re-ingesting a document does not duplicate.

    The id hashes the record kind, the source name, the page and the content,
    so identical content on different pages stays distinct while a repeated
    ingest of an unchanged file maps to the same id (and therefore upserts).
    """
    parts = [kind, source, "" if page is None else str(page), content]
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return f"{kind}-{digest[:32]}"


def source_key(path: str | Path) -> str:
    """Stable identifier for a source document (its file name)."""
    return Path(path).name


# --------------------------------------------------------------------------- #
# PDF extraction
# --------------------------------------------------------------------------- #
def _import_fitz() -> Any:
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # PyMuPDF < 1.25 shipped only the `fitz` alias.
        except ImportError as exc:
            raise DependencyError(
                "PDF parsing needs PyMuPDF.", "pip install -r requirements.txt"
            ) from exc
    return pymupdf


def _is_decorative(width: int, height: int, min_pixels: int) -> bool:
    """Heuristic for 'this image is a bullet, rule or logo, not a figure'."""
    if width <= 0 or height <= 0:
        return True
    if width * height < min_pixels:
        return True
    return min(width, height) < 24


def extract_pdf(
    pdf_path: str | Path,
    settings: AppSettings | None = None,
    image_dir: Path | None = None,
) -> SourceDocument:
    """Parse a PDF into text chunks and extracted images."""
    settings = settings or AppSettings()
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise IngestError(f"File not found: {pdf_path}")

    fitz = _import_fitz()
    source = source_key(pdf_path)
    target_dir = Path(image_dir) if image_dir else None
    try:
        file_bytes = pdf_path.stat().st_size
    except OSError:
        file_bytes = 0

    result = SourceDocument(source=source)
    with stage("parse.pdf", source=source) as parse_stats:
        try:
            document = fitz.open(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            raise IngestError(f"Could not open {pdf_path}: {exc}") from exc

        try:
            result.page_count = document.page_count
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text") or ""
                for index, piece in enumerate(
                    chunk_text(text, settings.chunk_size, settings.chunk_overlap)
                ):
                    result.chunks.append(
                        Chunk(
                            text=piece,
                            page=page_number,
                            index=index,
                            metadata={"type": "text", "source": source, "page": page_number},
                        )
                    )

                if not settings.extract_images:
                    continue

                for image_index, extracted in enumerate(
                    _extract_page_images(document, page, page_number)
                ):
                    if _is_decorative(extracted.width, extracted.height, settings.min_image_pixels):
                        continue
                    if target_dir is not None:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        destination = (
                            target_dir / f"{Path(source).stem}_p{page_number}_{image_index}.png"
                        )
                        try:
                            _save_image_bytes(extracted.raw, destination)
                            extracted.path = destination
                        except Exception as exc:  # noqa: BLE001
                            log.warning("Could not save image %s: %s", destination, exc)
                    result.images.append(extracted)

            parse_stats.items = result.page_count
            parse_stats.detail.update(
                chunks=len(result.chunks),
                images=len(result.images),
                bytes=file_bytes,
            )
        finally:
            with contextlib.suppress(Exception):
                document.close()

    log.info(
        "Parsed %s: %d pages, %d text chunks, %d images.",
        source,
        result.page_count,
        len(result.chunks),
        len(result.images),
    )
    return result


def _extract_page_images(document: Any, page: Any, page_number: int) -> Iterator[ExtractedImage]:
    """Yield raw images embedded on a page, skipping ones PyMuPDF cannot read."""
    try:
        image_infos = page.get_images(full=True)
    except Exception:  # noqa: BLE001
        return

    for index, info in enumerate(image_infos):
        xref = info[0]
        try:
            payload = document.extract_image(xref)
        except Exception:  # noqa: BLE001
            continue
        raw = payload.get("image")
        if not raw:
            continue
        width = int(payload.get("width") or 0)
        height = int(payload.get("height") or 0)
        yield ExtractedImage(
            path=None,
            page=page_number,
            index=index,
            width=width,
            height=height,
            description=f"Figure on page {page_number}",
            raw=raw,
        )


def _save_image_bytes(raw: bytes, destination: Path) -> None:
    """Persist raw image bytes, converting to PNG when necessary."""
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(raw)) as image:
            image.convert("RGB").save(destination, format="PNG")
    except Exception:  # noqa: BLE001
        destination.write_bytes(raw)


# --------------------------------------------------------------------------- #
# Plain text files
# --------------------------------------------------------------------------- #
def extract_text_file(
    path: str | Path,
    settings: AppSettings | None = None,
) -> SourceDocument:
    """Parse a plain-text / markdown file into chunks."""
    settings = settings or AppSettings()
    path = Path(path)
    if not path.exists():
        raise IngestError(f"File not found: {path}")

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise IngestError(f"Could not read {path}: {exc}") from exc

    source = source_key(path)
    result = SourceDocument(source=source, page_count=1)
    for index, piece in enumerate(chunk_text(raw, settings.chunk_size, settings.chunk_overlap)):
        result.chunks.append(
            Chunk(
                text=piece,
                page=None,
                index=index,
                metadata={"type": "text", "source": source},
            )
        )
    return result


def discover_documents(
    target: str | Path,
    recursive: bool = True,
) -> list[Path]:
    """List ingestible files below ``target``."""
    target = Path(target)
    if target.is_file():
        return [target]
    if not target.exists():
        raise IngestError(f"Path not found: {target}")

    suffixes = PDF_SUFFIXES | {".txt", ".md", ".markdown", ".rst"}
    pattern = "**/*" if recursive else "*"
    found = [
        p for p in sorted(target.glob(pattern)) if p.is_file() and p.suffix.lower() in suffixes
    ]
    return found


def image_descriptions(images: Sequence[ExtractedImage]) -> list[str]:
    """Fallback captions used as the ChromaDB document for image vectors."""
    return [image.description or f"Figure on page {image.page}" for image in images]
