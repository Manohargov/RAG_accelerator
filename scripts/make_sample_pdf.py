"""Build a small, deterministic PDF used to exercise BookRAG end to end.

The PDF is a four page miniature "textbook" that deliberately contains the
things the pipeline has to handle:

* two pages of real prose, long enough to be split by ``chunk_text``;
* two pages carrying an embedded raster figure plus its caption, so image
  extraction has something to find;
* a formula described in words as well as symbols, because PDF text
  extraction flattens subscripts and superscripts.

Each page has a distinct topic so a retrieval result can be checked against a
known answer rather than judged by eye.

Usage:
    python scripts/make_sample_pdf.py [output.pdf]

Writes ``examples/sample-book.pdf`` plus the two source figures next to it.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

#: A4 at 72 dpi, the unit PyMuPDF works in.
PAGE_WIDTH = 595
PAGE_HEIGHT = 842

LEFT_MARGIN = 64
BODY_SIZE = 10.5
LEADING = BODY_SIZE * 1.5
CONTENT_WIDTH = PAGE_WIDTH - 2 * LEFT_MARGIN

FIGURE_WIDTH = 380
FIGURE_HEIGHT = 270


# --------------------------------------------------------------------------- #
# Figures (drawn with PIL, then embedded into the PDF as raster images)
# --------------------------------------------------------------------------- #
def draw_temperature_graph(path: Path) -> Path:
    """A rising line graph: temperature against pressure."""
    from PIL import Image, ImageDraw

    width, height = 420, 300
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    left, top, right, bottom = 55, 25, 400, 245

    # Plot frame and grid.
    draw.rectangle([left, top, right, bottom], outline="black", width=2)
    for step in range(1, 5):
        y = bottom - int((bottom - top) * step / 5)
        draw.line([left, y, right, y], fill=(220, 220, 220), width=1)

    # The measurement: pressure rising linearly with temperature.
    points = []
    for step in range(6):
        x = left + int((right - left) * step / 5)
        y = bottom - int((bottom - top) * step / 5)
        points.append((x, y))
    draw.line(points, fill="crimson", width=5)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill="navy", outline="white", width=2)

    # Axis labels.
    draw.text((left - 4, bottom + 8), "temperature (K)", fill="black")
    draw.text((left - 45, top - 18), "pressure", fill="black")

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def draw_closed_system(path: Path) -> Path:
    """A piston-and-cylinder schematic: clearly unlike the graph above."""
    from PIL import Image, ImageDraw

    width, height = 420, 300
    image = Image.new("RGB", (width, height), (232, 243, 252))
    draw = ImageDraw.Draw(image)

    # System boundary.
    draw.rectangle([40, 40, 380, 260], outline=(60, 90, 140), width=3)
    draw.text((48, 22), "closed system", fill=(60, 90, 140))

    # Cylinder wall.
    draw.rectangle([120, 90, 300, 220], outline="black", width=4, fill=(255, 255, 255))
    # Piston head and rod.
    draw.rectangle([120, 90, 300, 118], fill=(120, 120, 120), outline="black", width=3)
    draw.rectangle([205, 55, 215, 90], fill=(90, 90, 90))
    # Gas molecules.
    for offset in range(14):
        x = 135 + (offset * 37) % 155
        y = 135 + (offset * 53) % 72
        draw.ellipse([x, y, x + 9, y + 9], fill=(210, 60, 60))

    # Heat in, no mass crossing the boundary.
    draw.line([340, 155, 305, 155], fill="orange", width=5)
    draw.polygon([(305, 145), (305, 165), (285, 155)], fill="orange")
    draw.text((330, 172), "heat", fill=(180, 100, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


# --------------------------------------------------------------------------- #
# Page layout
# --------------------------------------------------------------------------- #
def _wrap(text: str, width: float, size: float) -> list[str]:
    """Wrap to a character budget derived from Helvetica's average width."""
    columns = max(20, int(width / (size * 0.5)))
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            continue
        lines.extend(textwrap.wrap(paragraph, width=columns) or [""])
    return lines


def _write(page: object, text: str, y: float, *, size: float, font: str) -> float:
    for line in _wrap(text, CONTENT_WIDTH, size):
        page.insert_text((LEFT_MARGIN, y), line, fontsize=size, fontname=font)  # type: ignore[attr-defined]
        y += size * 1.5
    return y


def _heading(page: object, text: str, y: float, *, size: float = 17) -> float:
    return _write(page, text, y, size=size, font="hebo") + 10


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #
KINETIC_ENERGY = [
    "Kinetic energy is the energy a body carries because it is moving. It is "
    "written Ek and, for any object whose speed is far below the speed of "
    "light, it equals one half of the mass times the square of the speed. In "
    "symbols the formula for kinetic energy is Ek = 1/2 m v^2, where m is the "
    "mass of the moving body and v is its speed.",
    "The dependence on the square of the speed is the practical consequence "
    "worth remembering. Doubling the speed of a car does not double its "
    "kinetic energy, it quadruples it, because the speed appears squared. "
    "Doubling the mass, by contrast, doubles the energy exactly. This is why "
    "a heavy lorry and a small car travelling at the same speed carry very "
    "different amounts of energy and need very different distances to stop.",
    "A worked example makes the arithmetic concrete. Take a car of mass 1200 "
    "kilograms travelling at 20 metres per second. Its kinetic energy is "
    "one half of 1200 multiplied by 20 squared, that is 600 times 400, which "
    "is 240 000 joules, or 240 kilojoules. Raise the speed to 40 metres per "
    "second and the figure becomes 960 kilojoules: four times as much, "
    "exactly as the square term predicts.",
    "Kinetic energy is also the quantity that has to go somewhere when a "
    "vehicle brakes. The brakes convert the stored kinetic energy into heat "
    "in the pads and discs, and the distance needed to come to rest grows "
    "with the energy that has to be dissipated. Because the energy grows with "
    "the square of the speed while the braking force is roughly constant, "
    "the stopping distance grows roughly with the square of the speed too.",
]

THERMODYNAMICS = [
    "Thermodynamics is the study of heat, work and the energy stored inside a "
    "system. The simplest starting point is the behaviour of a gas held at "
    "constant volume. As shown in the graph, pressure increases linearly with "
    "temperature at constant volume. This is Gay-Lussac's law, and the "
    "straight line it produces passes through the origin when both quantities "
    "are measured on an absolute scale.",
    "The ideal gas law collects the individual gas laws into a single "
    "relation. The pressure of a fixed mass of gas equals a constant times "
    "the amount of substance and the absolute temperature, divided by the "
    "volume. Holding the volume fixed removes it from the comparison, leaving "
    "pressure directly proportional to temperature, which is precisely the "
    "linear relationship drawn in the figure.",
    "The word system does a great deal of work in these definitions, so it is "
    "worth being precise about it. A closed thermodynamic system exchanges "
    "energy with its surroundings, as heat or as work, but exchanges no "
    "matter at all. A piston sealed against a cylinder containing a fixed "
    "mass of gas is the textbook example: heat can flow through the walls and "
    "the piston can move to deliver work, yet no gas ever crosses the "
    "boundary. An open system, by contrast, allows matter to cross the "
    "boundary as well.",
    "The distinction matters because it fixes which quantities are conserved. "
    "Inside a closed system the total energy is fixed even though it keeps "
    "changing form, and the mass of the contents never changes either. Those "
    "two conservation statements are what make the state of the gas "
    "predictable from its pressure, volume and temperature alone.",
]


def build(output: Path) -> tuple[Path, Path]:
    import pymupdf

    graph_path = draw_temperature_graph(
        output.parent / "figures" / "temperature_pressure_graph.png"
    )
    system_path = draw_closed_system(output.parent / "figures" / "closed_system_piston.png")

    document = pymupdf.open()

    # --- Pages 1 and 2: prose ---------------------------------------------
    for title, paragraphs in (
        ("Chapter 1 - Motion and Kinetic Energy", KINETIC_ENERGY),
        ("Chapter 2 - Thermodynamics and the Gas Laws", THERMODYNAMICS),
    ):
        page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        y = 82
        y = _heading(page, title, y)
        page.draw_line(
            (LEFT_MARGIN, y - 6), (PAGE_WIDTH - LEFT_MARGIN, y - 6), color=(0.7, 0.7, 0.7)
        )
        y += 22
        for paragraph in paragraphs:
            y = _write(page, paragraph, y, size=BODY_SIZE, font="helv")
            y += 12

    # --- Page 3: the graph -------------------------------------------------
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = 82
    y = _heading(page, "Figure 1 - Temperature and Pressure", y)
    page.draw_line((LEFT_MARGIN, y - 6), (PAGE_WIDTH - LEFT_MARGIN, y - 6), color=(0.7, 0.7, 0.7))
    y += 26
    page.insert_image(
        pymupdf.Rect(LEFT_MARGIN, y, LEFT_MARGIN + FIGURE_WIDTH, y + FIGURE_HEIGHT),
        filename=str(graph_path),
    )
    y += FIGURE_HEIGHT + 26
    y = _write(
        page,
        "Figure 1. A line graph showing the linear relationship between "
        "temperature and pressure. Pressure is plotted vertically and "
        "temperature horizontally; the five measurements fall on a straight "
        "line through the origin, which is the behaviour Gay-Lussac's law "
        "predicts for a gas held at constant volume.",
        y,
        size=BODY_SIZE,
        font="hebo",
    )

    # --- Page 4: the schematic ---------------------------------------------
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = 82
    y = _heading(page, "Figure 2 - A Closed System", y)
    page.draw_line((LEFT_MARGIN, y - 6), (PAGE_WIDTH - LEFT_MARGIN, y - 6), color=(0.7, 0.7, 0.7))
    y += 26
    page.insert_image(
        pymupdf.Rect(LEFT_MARGIN, y, LEFT_MARGIN + FIGURE_WIDTH, y + FIGURE_HEIGHT),
        filename=str(system_path),
    )
    y += FIGURE_HEIGHT + 26
    y = _write(
        page,
        "Figure 2. A schematic diagram of a closed thermodynamic system: a "
        "sealed cylinder and piston containing a fixed mass of gas. Heat "
        "crosses the boundary from the right, and the piston may move to "
        "deliver work, but no gas leaves or enters the cylinder.",
        y,
        size=BODY_SIZE,
        font="hebo",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(output))
    document.close()
    return output, graph_path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[1]
    output = Path(argv[0]) if argv else root / "examples" / "sample-book.pdf"

    pdf, graph = build(output)
    print(f"wrote {pdf}")
    print(f"wrote {graph}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
