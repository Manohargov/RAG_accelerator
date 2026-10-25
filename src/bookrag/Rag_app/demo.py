"""Self-contained demo that reproduces the original script's behaviour.

Creates a tiny synthetic "book" (two text passages plus two figures), indexes
it, and runs both a text query and an image query so the whole loop can be
smoke-tested without a real PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..logs.logging_setup import get_logger

log = get_logger("demo")

SAMPLE_TEXTS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "text_page1",
        "The formula for kinetic energy is defined as $E_k = \\frac{1}{2}mv^2$ where m is mass.",
        {"type": "text", "page": 1, "section": "Physics Basics"},
    ),
    (
        "text_page2",
        "As shown in the graph, pressure increases linearly with temperature at constant volume.",
        {"type": "text", "page": 2, "section": "Thermodynamics"},
    ),
]

SAMPLE_IMAGES: list[tuple[str, str]] = [
    (
        "temp_graph.png",
        "A line graph showing the linear relationship between temperature and pressure.",
    ),
    (
        "temp_diagram.png",
        "A schematic diagram of a closed thermodynamic system.",
    ),
]


def build_sample_assets(target_dir: Path) -> tuple[list[Path], list[str]]:
    """Write the sample figures to disk and return their paths and captions."""
    from PIL import Image, ImageDraw

    target_dir.mkdir(parents=True, exist_ok=True)

    # A visible line graph rather than a flat rectangle, so the image
    # embedding carries actual signal.
    graph = Image.new("RGB", (320, 240), color="white")
    draw = ImageDraw.Draw(graph)
    draw.rectangle([40, 20, 300, 210], outline="black", width=2)
    draw.line([40, 210, 300, 30], fill="crimson", width=4)
    for offset in range(5):
        x = 60 + offset * 55
        y = 190 - offset * 38
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="navy")
    graph_path = target_dir / "temp_graph.png"
    graph.save(graph_path)

    diagram = Image.new("RGB", (320, 240), color="lightblue")
    draw = ImageDraw.Draw(diagram)
    draw.rounded_rectangle(
        [80, 70, 240, 180], radius=18, outline="black", width=3, fill="lightgreen"
    )
    draw.ellipse([140, 95, 180, 135], outline="black", width=3)
    draw.text((96, 150), "closed system", fill="black")
    diagram_path = target_dir / "temp_diagram.png"
    diagram.save(diagram_path)

    return [graph_path, diagram_path], [caption for _, caption in SAMPLE_IMAGES]


def run_demo(pipeline: Any, workdir: Path | None = None) -> dict[str, Any]:
    """Index the sample book and query it. Returns a summary dictionary."""
    from ..data import paths
    from .ingest import Chunk, ExtractedImage, SourceDocument

    if workdir is None:
        workdir = paths.data_dir() / "demo_assets"
    workdir = Path(workdir)
    image_paths, captions = build_sample_assets(workdir)

    chunks = [
        Chunk(
            text=text,
            page=metadata.get("page"),
            index=index,
            metadata={**metadata, "source": "demo_book"},
        )
        for index, (_id, text, metadata) in enumerate(SAMPLE_TEXTS)
    ]
    images = [
        ExtractedImage(
            path=path,
            page=3 + index,
            index=index,
            width=320,
            height=240,
            description=caption,
        )
        for index, (path, caption) in enumerate(zip(image_paths, captions, strict=True))
    ]

    document = SourceDocument(source="demo_book", chunks=chunks, images=images, page_count=4)
    report = pipeline.ingest_source(document, replace=True)

    print("\n--- Testing Text Query ---")
    text_hits = pipeline.search_text("What is the equation for kinetic energy?", top_k=2)
    for index, hit in enumerate(text_hits, start=1):
        print(f"Match {index}: {hit.document}")
        print(f"  -> Metadata: {hit.metadata}\n")

    print("--- Testing Image Query ---")
    image_hits = pipeline.search_image(image_paths[0], top_k=2)
    for index, hit in enumerate(image_hits, start=1):
        print(f"Match {index}: {hit.document}")
        print(f"  -> Metadata: {hit.metadata}\n")

    return {
        "ingest": report.as_dict(),
        "text_hits": [hit.as_dict() for hit in text_hits],
        "image_hits": [hit.as_dict() for hit in image_hits],
    }
