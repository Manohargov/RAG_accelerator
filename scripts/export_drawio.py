"""Export a multi-page .drawio file to PNG / SVG / PDF.

The draw.io CLI only exports the first page for PNG and SVG, so this script
splits the file into one page per file, exports each, and merges the PDF
pages back together.

Usage:
    python scripts/export_drawio.py docs/bookrag-architecture.drawio
    python scripts/export_drawio.py docs/bookrag-architecture.drawio --scale 2
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

#: Candidate locations of the draw.io executable.
DRAWIO_CANDIDATES = [
    "drawio",
    "draw.io",
    r"C:\Program Files\draw.io\draw.io.exe",
    r"C:\Program Files (x86)\draw.io\draw.io.exe",
    "/Applications/draw.io.app/Contents/MacOS/draw.io",
    "/Applications/draw.io.app/Contents/MacOS/draw.io",
    "/usr/bin/drawio",
    "/usr/local/bin/drawio",
    "/snap/bin/drawio",
]


def find_drawio() -> str | None:
    """Locate the draw.io executable."""
    for candidate in DRAWIO_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).exists():
            return candidate
    return None


def split_pages(source: Path, workdir: Path) -> list[tuple[str, Path]]:
    """Write one single-page .drawio file per page. Returns (name, path)."""
    tree = ET.parse(source)
    root = tree.getroot()
    pages = root.findall("diagram")
    if not pages:
        raise SystemExit(f"No <diagram> pages found in {source}")

    written: list[tuple[str, Path]] = []
    for index, page in enumerate(pages, start=1):
        single = ET.Element("mxfile", dict(root.attrib))
        single.append(page)
        target = workdir / f"_page{index}.drawio"
        ET.ElementTree(single).write(target, encoding="utf-8", xml_declaration=False)
        written.append((page.get("name") or f"page{index}", target))
    return written


def slug(name: str) -> str:
    """Turn a page name into a filename-safe slug."""
    cleaned = "".join(c if c.isalnum() else "-" for c in name.lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "page"


def prefixed_name(stem: str, page_name: str) -> str:
    """Combine the file stem with a page name without repeating the prefix.

    ``bookrag-architecture`` + ``Architecture`` -> ``bookrag-architecture``
    ``bookrag-architecture`` + ``Data flow``   -> ``bookrag-data-flow``
    """
    page = slug(page_name)
    prefix = slug(stem)
    if page == prefix:
        return prefix
    if page.startswith(prefix + "-"):
        return page
    if prefix.endswith(page):
        return prefix
    return f"{prefix}-{page}"


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def png_has_embedded_diagram(path: Path) -> bool:
    """Check whether a PNG carries an embedded, editable draw.io diagram."""
    blob = path.read_bytes()
    if blob[:8] != b"\x89PNG\r\n\x1a\n":
        return False
    pos = 8
    while pos < len(blob):
        length = struct.unpack(">I", blob[pos : pos + 4])[0]
        chunk_type = blob[pos + 4 : pos + 8]
        if chunk_type == b"zTXt":
            payload = blob[pos + 8 : pos + 8 + length]
            keyword, _, rest = payload.partition(b"\x00")
            if keyword == b"mxGraphModel" and rest:
                # rest = compression-method byte + zlib stream
                try:
                    return b"mxGraphModel" in zlib.decompress(rest[1:])
                except zlib.error:
                    return False
        pos += 12 + length
    return False


def merge_pdfs(parts: list[Path], target: Path) -> int:
    """Concatenate single-page PDFs into one document."""
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for part in parts:
        for page in PdfReader(part).pages:
            writer.add_page(page)
    with target.open("wb") as handle:
        writer.write(handle)
    return len(PdfReader(target).pages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="the .drawio file to export")
    parser.add_argument("--outdir", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--prefix",
        default=None,
        help="prefix for per-page image names (default: the file stem)",
    )
    parser.add_argument("--scale", type=int, default=2, help="PNG scale factor")
    parser.add_argument("--no-pdf", action="store_true", help="skip the PDF export")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"Not found: {source}")

    drawio = find_drawio()
    if not drawio:
        raise SystemExit(
            "draw.io executable not found. Install draw.io Desktop and make sure "
            "'drawio' is on PATH."
        )

    outdir = (args.outdir or source.parent).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    prefix = slug(args.prefix) if args.prefix else stem

    with tempfile.TemporaryDirectory(prefix="drawio-export-") as tmp:
        workdir = Path(tmp)
        pages = split_pages(source, workdir)
        print(f"Exporting {len(pages)} page(s) from {source.name} using {drawio}")

        pdf_parts: list[Path] = []
        for name, page_file in pages:
            base = prefixed_name(prefix, name)

            png = outdir / f"{base}.png"
            run(
                [
                    drawio,
                    "-x",
                    "-f",
                    "png",
                    "-e",
                    "-s",
                    str(args.scale),
                    "-o",
                    str(png),
                    str(page_file),
                ]
            )
            editable = png_has_embedded_diagram(png)
            print(f"  PNG  {png.name:<32} editable={editable}")

            svg = outdir / f"{base}.svg"
            run(
                [
                    drawio,
                    "-x",
                    "-f",
                    "svg",
                    "-e",
                    "--embed-svg-images",
                    "-o",
                    str(svg),
                    str(page_file),
                ]
            )
            print(f"  SVG  {svg.name:<32} {svg.stat().st_size // 1024} KB")

            if not args.no_pdf:
                part = workdir / f"{base}.pdf"
                run([drawio, "-x", "-f", "pdf", "-e", "-o", str(part), str(page_file)])
                pdf_parts.append(part)

        if pdf_parts:
            pdf = outdir / f"{stem}.pdf"
            count = merge_pdfs(pdf_parts, pdf)
            print(f"  PDF  {pdf.name:<32} {count} page(s)")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
