"""Validate a .drawio file: XML well-formedness, cell references, geometry.

Usage:  python scripts/validate_drawio.py docs/bookrag-architecture.drawio
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def validate(path: Path) -> int:
    tree = ET.parse(path)
    root = tree.getroot()
    problems: list[str] = []

    if root.tag != "mxfile":
        problems.append(f"root element is <{root.tag}>, expected <mxfile>")

    pages = root.findall("diagram")
    if not pages:
        problems.append("no <diagram> pages found")

    for page in pages:
        name = page.get("name", "?")
        cells = page.findall(".//mxCell")
        ids = [c.get("id") for c in cells]

        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            problems.append(f"[{name}] duplicate ids: {sorted(dupes)}")

        if "0" not in ids or "1" not in ids:
            problems.append(f"[{name}] missing reserved cells 0/1")

        for cell in cells:
            cid = cell.get("id")
            # every referenced parent must exist
            parent = cell.get("parent")
            if parent is not None and parent not in ids:
                problems.append(f"[{name}] cell {cid}: unknown parent {parent}")

            geom = cell.find("mxGeometry")
            if cell.get("vertex") == "1" and geom is None:
                problems.append(f"[{name}] vertex {cid}: missing mxGeometry")
            if cell.get("edge") == "1":
                if geom is None:
                    problems.append(f"[{name}] edge {cid}: missing mxGeometry")
                elif geom.get("relative") != "1":
                    problems.append(f"[{name}] edge {cid}: mxGeometry relative != '1'")
                for role in ("source", "target"):
                    ref = cell.get(role)
                    if ref is None:
                        problems.append(f"[{name}] edge {cid}: missing {role}")
                    elif ref not in ids:
                        problems.append(f"[{name}] edge {cid}: unknown {role} '{ref}'")
                if cell.get("source") == cell.get("target"):
                    problems.append(f"[{name}] edge {cid}: self loop")

        vertices = [c for c in cells if c.get("vertex") == "1"]
        edges = [c for c in cells if c.get("edge") == "1"]
        print(f"[{name}] {len(vertices)} vertices, {len(edges)} edges")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nStructure OK.")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/bookrag-architecture.drawio")
    raise SystemExit(validate(target))
