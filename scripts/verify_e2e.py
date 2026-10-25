"""End-to-end verification of the `bookrag` command against the real stack.

Unlike the pytest suite (which runs on fakes), this script drives the installed
CLI as separate processes, exactly the way a user would, and checks the answers
against a known four page sample book:

* page 1  - the kinetic energy formula
* page 2  - thermodynamics and the gas laws
* page 3  - Figure 1, a temperature/pressure line graph
* page 4  - Figure 2, a closed-system schematic

The checks cover the four properties the project exists to guarantee:

1. a PDF becomes text chunks *and* extracted figures;
2. a text question finds the right page;
3. a text question can return a *figure* and an image query can return the
   *paragraph* that explains it (the shared vector space);
4. re-ingesting the same file is idempotent.

Usage:
    python scripts/verify_e2e.py [--keep]

Runs against a temporary ``BOOKRAG_HOME`` so it never touches real data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = ROOT / "examples" / "sample-book.pdf"
GRAPH_PNG = ROOT / "examples" / "figures" / "temperature_pressure_graph.png"

#: One result row: (check name, passed, detail).
Row = tuple[str, bool, str]


class Runner:
    """Runs the installed `bookrag` CLI against an isolated BOOKRAG_HOME."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.env = {**os.environ, "BOOKRAG_HOME": str(home)}
        self.last_seconds = 0.0

    def cli(self, *args: str, expect_json: bool = False) -> Any:
        """Invoke `bookrag` and return parsed JSON (or the raw text)."""
        command = [sys.executable, "-m", "bookrag", *args]
        started = time.perf_counter()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=self.env,
            cwd=ROOT,
        )
        self.last_seconds = time.perf_counter() - started
        if result.returncode != 0:
            raise RuntimeError(
                f"`bookrag {' '.join(args)}` exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if expect_json:
            return json.loads(result.stdout)
        return result.stdout


def check_ingest(runner: Runner) -> list[Row]:
    """A PDF becomes text chunks and extracted figures."""
    report = runner.cli("ingest", str(SAMPLE_PDF), "--json", expect_json=True)
    entry = report[0]
    rows: list[Row] = []

    rows.append(("ingest exits 0", True, f"{runner.last_seconds:.1f}s"))
    rows.append(
        ("parses 4 pages", entry["pages"] == 4, f"pages={entry['pages']}"),
    )
    rows.append(
        ("produces text chunks", entry["text_chunks"] == 6, f"chunks={entry['text_chunks']}"),
    )
    rows.append(
        ("extracts both figures", entry["images"] == 2, f"images={entry['images']}"),
    )
    rows.append(
        ("no extraction errors", not entry["errors"], f"errors={entry['errors']}"),
    )

    stats = runner.cli("stats", "--json", expect_json=True)
    rows.append(
        (
            "stats totals match ingest",
            stats["total"] == entry["total"],
            f"stats={stats['total']} ingest={entry['total']}",
        )
    )
    rows.append(
        (
            "stats counts by modality",
            stats["by_type"] == {"image": 2, "text": 6},
            f"{stats['by_type']}",
        )
    )
    rows.append(
        (
            "collection uses cosine",
            stats["space"] == "cosine",
            f"space={stats['space']}",
        )
    )
    return rows


def check_text_retrieval(runner: Runner) -> list[Row]:
    """A text question should surface the page that answers it."""
    rows: list[Row] = []

    hits = runner.cli(
        "query",
        "What is the formula for kinetic energy?",
        "-k",
        "3",
        "--json",
        expect_json=True,
    )
    top = hits[0]
    rows.append(
        (
            "kinetic-energy query hits page 1",
            top["metadata"].get("page") == 1 and top["metadata"].get("type") == "text",
            f"p{top['metadata'].get('page')} {top['metadata'].get('type')} score={top['score']}",
        )
    )
    rows.append(
        (
            "answer mentions the formula",
            "1/2 m v^2" in top["document"] or "Ek = 1/2" in top["document"],
            top["document"][:70].replace("\n", " "),
        )
    )
    rows.append(
        (
            "scores are cosine-derived (0,1]",
            all(0.0 < hit["score"] <= 1.0 for hit in hits),
            f"scores={[round(h['score'], 4) for h in hits]}",
        )
    )
    rows.append(
        (
            "results are ranked descending",
            all(a["score"] >= b["score"] for a, b in zip(hits, hits[1:], strict=False)),
            "ok",
        )
    )

    # Source filter narrows the corpus to one document.
    filtered = runner.cli(
        "query",
        "pressure and temperature",
        "-s",
        "sample-book.pdf",
        "-k",
        "2",
        "--json",
        expect_json=True,
    )
    rows.append(
        (
            "source filter respected",
            all(h["metadata"].get("source") == "sample-book.pdf" for h in filtered),
            f"{len(filtered)} hits, all from sample-book.pdf",
        )
    )

    # Type filter proves the metadata is stored and queryable.
    images_only = runner.cli(
        "query",
        "a diagram of a piston",
        "-t",
        "image",
        "-k",
        "2",
        "--json",
        expect_json=True,
    )
    rows.append(
        (
            "type filter returns only figures",
            bool(images_only) and all(h["metadata"].get("type") == "image" for h in images_only),
            f"{len(images_only)} hits, types={sorted({h['metadata'].get('type') for h in images_only})}",
        )
    )
    return rows


def check_shared_space(runner: Runner) -> list[Row]:
    """The defining property: text and images are mutually retrievable."""
    rows: list[Row] = []

    # Text -> image: ask about the graph, get the figure.
    text_to_image = runner.cli(
        "query",
        "a line graph of pressure against temperature",
        "-t",
        "image",
        "-k",
        "2",
        "--json",
        expect_json=True,
    )
    top = text_to_image[0]
    rows.append(
        (
            "text query returns a figure",
            top["metadata"].get("type") == "image",
            f"p{top['metadata'].get('page')} score={top['score']}",
        )
    )
    rows.append(
        (
            "the figure returned is the graph (page 3)",
            top["metadata"].get("page") == 3,
            f"page={top['metadata'].get('page')}",
        )
    )

    # Image -> text: show the graph, get the paragraphs that discuss it.
    image_to_text = runner.cli(
        "find-image",
        str(GRAPH_PNG),
        "-t",
        "text",
        "-k",
        "2",
        "--json",
        expect_json=True,
    )
    pages = {hit["metadata"].get("page") for hit in image_to_text}
    rows.append(
        (
            "image query returns the explaining paragraph",
            bool(image_to_text)
            and all(hit["metadata"].get("type") == "text" for hit in image_to_text),
            f"pages={sorted(p for p in pages if p is not None)}",
        )
    )

    # Image -> image: the graph should match itself over the schematic.
    image_to_image = runner.cli(
        "find-image",
        str(GRAPH_PNG),
        "-k",
        "3",
        "--json",
        expect_json=True,
    )
    top = image_to_image[0]
    rows.append(
        (
            "image query matches the same figure first",
            top["metadata"].get("type") == "image" and top["metadata"].get("page") == 3,
            f"p{top['metadata'].get('page')} {top['metadata'].get('type')} score={top['score']}",
        )
    )
    figure_scores = {
        hit["metadata"].get("page"): hit["score"]
        for hit in image_to_image
        if hit["metadata"].get("type") == "image"
    }
    rows.append(
        (
            "graph outranks the unrelated schematic",
            figure_scores.get(3, 0) > figure_scores.get(4, 0),
            f"graph={figure_scores.get(3)} schematic={figure_scores.get(4)}",
        )
    )
    return rows


def check_idempotency(runner: Runner) -> list[Row]:
    """Re-ingesting an unchanged file must not duplicate or lose records."""
    rows: list[Row] = []

    before = runner.cli("stats", "--json", expect_json=True)
    ids_before = _record_ids(runner)

    runner.cli("ingest", str(SAMPLE_PDF), "--json", expect_json=True)

    after = runner.cli("stats", "--json", expect_json=True)
    ids_after = _record_ids(runner)

    rows.append(
        (
            "re-ingest keeps the record count",
            before["total"] == after["total"],
            f"{before['total']} -> {after['total']}",
        )
    )
    rows.append(
        (
            "record ids are unchanged",
            ids_before == ids_after,
            f"{len(ids_before)} ids",
        )
    )

    # An edited file with new content adds records without orphaning the rest.
    edited = runner.home / "notes.md"
    edited.write_text(
        "# Marginalia\n\nThe specific heat capacity of water is 4.18 joules per gram per kelvin.\n",
        encoding="utf-8",
    )
    runner.cli("ingest", str(edited), "--json", expect_json=True)
    grown = runner.cli("stats", "--json", expect_json=True)
    rows.append(
        (
            "a new source is added alongside the old one",
            grown["total"] > after["total"],
            f"{after['total']} -> {grown['total']}",
        )
    )
    rows.append(
        (
            "the original document survives",
            all(
                h["metadata"].get("source") == "sample-book.pdf"
                for h in runner.cli(
                    "query", "kinetic energy", "-k", "1", "--json", expect_json=True
                )
            ),
            "sample-book.pdf still indexed",
        )
    )
    return rows


def _record_ids(runner: Runner) -> list[str]:
    """Collect every record id currently in the collection.

    ``top_k`` is capped at 200 by the settings, and the sample corpus holds
    fewer than a dozen records, so one unfiltered query returns all of them.
    """
    hits = runner.cli("query", "energy", "-k", "200", "--json", expect_json=True)
    return sorted(hit["id"] for hit in hits)


def check_reset(runner: Runner) -> list[Row]:
    """`reset` empties the collection and the app stays usable afterwards."""
    rows: list[Row] = []
    runner.cli("reset", "--yes")
    stats = runner.cli("stats", "--json", expect_json=True)
    rows.append(("reset empties the collection", stats["total"] == 0, f"total={stats['total']}"))
    return rows


def check_console_script() -> list[Row]:
    """The `bookrag` entry point is on PATH and reports a healthy environment."""
    rows: list[Row] = []
    located = shutil.which("bookrag")
    rows.append(("`bookrag` entry point on PATH", located is not None, located or "not found"))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the temp data dir afterwards")
    args = parser.parse_args()

    # Each CLI invocation can take minutes on CPU, so stream progress instead
    # of letting stdout sit in a block buffer when the output is redirected.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(line_buffering=True)

    if not SAMPLE_PDF.exists():
        print("Sample book missing; building it first...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_pdf.py")], check=True)

    home = Path(tempfile.mkdtemp(prefix="bookrag-e2e-"))
    runner = Runner(home)
    results: list[Row] = []

    print(f"BOOKRAG_HOME = {home}")
    print(f"sample book  = {SAMPLE_PDF}\n")

    try:
        for title, step in (
            ("Console script", check_console_script),
            ("Ingestion", lambda: check_ingest(runner)),
            ("Text retrieval", lambda: check_text_retrieval(runner)),
            ("Shared vector space", lambda: check_shared_space(runner)),
            ("Idempotent re-ingest", lambda: check_idempotency(runner)),
            ("Reset", lambda: check_reset(runner)),
        ):
            print(f"{title}")
            print("-" * len(title))
            try:
                step_rows = step()
            except Exception as exc:  # noqa: BLE001 - surface, do not abort the run
                step_rows = [(f"{title} raised", False, str(exc)[:160])]
            for name, passed, detail in step_rows:
                results.append((name, passed, detail))
                print(f"  {'PASS' if passed else 'FAIL'}  {name:<46} {detail}")
            print()

        passed = sum(1 for _, ok, _ in results if ok)
        print("=" * 78)
        print(f"{passed}/{len(results)} checks passed")
    finally:
        if args.keep:
            print(f"kept: {home}")
        else:
            shutil.rmtree(home, ignore_errors=True)

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
