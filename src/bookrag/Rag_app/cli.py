"""Command line interface for BookRAG.

bookrag doctor                     check the environment
bookrag ingest <file-or-dir>       index documents
bookrag query "your question"      search the corpus
bookrag find-image <image>         search the corpus with an image
bookrag demo                       run the built-in smoke test
bookrag stats                      show what is indexed
bookrag reset                      clear the collection
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..data import paths
from ..errors import BookRagError, DependencyError
from ..logs.logging_setup import get_logger, setup_logging
from ..logs.monitoring import get_monitor, stage
from .settings import AppSettings

log = get_logger("cli")


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _print_hits(hits: Sequence[Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps([hit.as_dict() for hit in hits], indent=2, ensure_ascii=False))
        return

    if not hits:
        print("No matches found.")
        return

    for index, hit in enumerate(hits, start=1):
        meta = hit.metadata
        location = meta.get("source", "unknown")
        if meta.get("page") not in (None, 0):
            location = f"{location} p.{meta['page']}"
        print(f"{index}. [score {hit.score:.4f}] {hit.kind.upper()} - {location}")
        body = hit.document.strip()
        if len(body) > 400:
            body = body[:400].rstrip() + "..."
        print(f"   {body}\n")


def _filters_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    filters: dict[str, Any] = {}
    if getattr(args, "type", None):
        filters["type"] = args.type
    if getattr(args, "source", None):
        filters["source"] = args.source
    return filters or None


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_doctor(args: argparse.Namespace) -> int:
    """Report the state of the environment."""
    print("BookRAG doctor")
    print("=" * 40)

    print(f"\n[1/5] Python           {sys.version.split()[0]}")

    print("\n[2/5] Python packages")
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("timm", "timm"),
        ("einops", "einops"),
        ("torchvision", "torchvision"),
        ("chromadb", "chromadb"),
        ("pymupdf", "pymupdf"),
        ("pillow", "PIL"),
        ("numpy", "numpy"),
    ]
    missing: list[str] = []
    for label, module in packages:
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "installed")
            print(f"      OK   {label:<14} {version}")
        except ImportError:
            print(f"      MISS {label:<14} not installed")
            missing.append(label)

    print("\n[3/5] Compute device")
    try:
        from ..accelerator.embedder import resolve_device

        print(f"      selected: {resolve_device('auto')}")
    except DependencyError:
        print("      unavailable (torch missing)")

    print("\n[4/5] Model cache")
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    print(f"      default cache: {cache}")
    print(f"      exists: {cache.exists()}")
    print("      note: weights download on first use (~3 GB for the v5-omni default)")

    print("\n[5/5] Data directory")
    print(f"      {paths.data_dir()}")
    print(f"      exists: {paths.data_dir().exists()}")

    if missing:
        print("\nInstall the missing packages with:")
        print("    pip install -r requirements.txt")
        return 1

    print("\nAll good. Run 'bookrag demo' for a smoke test.")
    return 0


def _apply_model_arg(args: argparse.Namespace, pipeline: Any) -> None:
    """Point this run at a different checkpoint without editing settings.json.

    Called the moment the pipeline exists and before anything has connected
    or loaded, so the store's fingerprint check compares against the override
    rather than against the configured default.
    """
    model = getattr(args, "model", None)
    if model:
        pipeline.settings.model_id = model


def cmd_ingest(args: argparse.Namespace) -> int:
    from .pipeline import BookRagPipeline

    pipeline = BookRagPipeline.from_environment(db_path=getattr(args, "db", None))
    _apply_model_arg(args, pipeline)
    if args.no_images:
        pipeline.settings.extract_images = False

    reports = pipeline.ingest_path(
        args.target, recursive=not args.no_recursive, replace=not args.keep_old
    )
    total_chunks = sum(r.text_chunks for r in reports)
    total_images = sum(r.images for r in reports)

    if args.json:
        print(json.dumps([r.as_dict() for r in reports], indent=2, ensure_ascii=False))
    else:
        for report in reports:
            print(
                f"  {report.source}: {report.text_chunks} text chunks, "
                f"{report.images} images ({report.pages} pages)"
            )
        print(f"\nIndexed {total_chunks} text chunks and {total_images} images.")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from .pipeline import BookRagPipeline

    pipeline = BookRagPipeline.from_environment(db_path=getattr(args, "db", None))
    _apply_model_arg(args, pipeline)
    hits = pipeline.search_text(
        args.text,
        top_k=args.top_k,
        filters=_filters_from_args(args),
    )
    _print_hits(hits, args.json)
    return 0


def cmd_find_image(args: argparse.Namespace) -> int:
    from .pipeline import BookRagPipeline

    pipeline = BookRagPipeline.from_environment(db_path=getattr(args, "db", None))
    _apply_model_arg(args, pipeline)
    hits = pipeline.search_image(
        args.image,
        top_k=args.top_k,
        filters=_filters_from_args(args),
    )
    _print_hits(hits, args.json)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo
    from .pipeline import BookRagPipeline

    pipeline = BookRagPipeline.from_environment(db_path=getattr(args, "db", None))
    _apply_model_arg(args, pipeline)
    result = run_demo(pipeline, workdir=Path(args.workdir) if args.workdir else None)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .pipeline import BookRagPipeline

    pipeline = BookRagPipeline.from_environment(db_path=getattr(args, "db", None))
    _apply_model_arg(args, pipeline)
    stats = pipeline.stats()
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    print(f"Collection : {stats['collection']}")
    print(f"Distance   : {stats['space']}")
    print(f"Records    : {stats['total']}")
    for kind, count in sorted(stats["by_type"].items()):
        print(f"  - {kind:<8} {count}")
    if stats["sources"]:
        print("Sources    :")
        for source in stats["sources"]:
            print(f"  - {source}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    from .pipeline import BookRagPipeline

    if not args.yes:
        answer = input("Delete every record in the collection? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return 1

    pipeline = BookRagPipeline.from_environment(db_path=getattr(args, "db", None))
    _apply_model_arg(args, pipeline)
    pipeline.reset()
    print("Collection reset.")
    return 0


def cmd_settings(args: argparse.Namespace) -> int:
    settings = AppSettings().load()
    settings.apply_environment()
    if args.set:
        for assignment in args.set:
            if "=" not in assignment:
                print(f"Skipping malformed --set value: {assignment!r}", file=sys.stderr)
                continue
            key, _, value = assignment.partition("=")
            key = key.strip()
            if key not in settings.to_dict():
                print(f"Unknown setting: {key!r}", file=sys.stderr)
                continue
            settings.update(**{key: value})
    if args.save:
        path = settings.save()
        print(f"Saved settings to {path}")
    print(json.dumps(settings.to_dict(), indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bookrag",
        description="Multimodal RAG over books and PDFs (Jina embeddings + ChromaDB).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print stage timings and model runtime when the command finishes",
    )
    parser.add_argument("--db", type=Path, default=None, help="path to the ChromaDB directory")
    parser.add_argument(
        "--model",
        default=None,
        help="use a different embedding checkpoint for this run (e.g. jinaai/jina-clip-v2)",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    doctor = subparsers.add_parser("doctor", help="check the environment")
    doctor.set_defaults(func=cmd_doctor)

    ingest = subparsers.add_parser("ingest", help="index documents")
    ingest.add_argument("target", help="file or directory to index")
    ingest.add_argument("--no-images", action="store_true", help="skip image extraction")
    ingest.add_argument(
        "--no-recursive", action="store_true", help="do not descend into subdirectories"
    )
    ingest.add_argument("--keep-old", action="store_true", help="keep previously indexed records")
    ingest.add_argument("--json", action="store_true", help="emit JSON")
    ingest.set_defaults(func=cmd_ingest)

    query = subparsers.add_parser("query", help="search the corpus with text")
    query.add_argument("text", help="the question to search for")
    query.add_argument("-k", "--top-k", type=int, default=None, help="number of results")
    query.add_argument(
        "-t", "--type", choices=["text", "image"], default=None, help="filter by record type"
    )
    query.add_argument("-s", "--source", default=None, help="filter by source document")
    query.add_argument("--json", action="store_true", help="emit JSON")
    query.set_defaults(func=cmd_query)

    find_image = subparsers.add_parser("find-image", help="search the corpus with an image")
    find_image.add_argument("image", help="path to the query image")
    find_image.add_argument("-k", "--top-k", type=int, default=None, help="number of results")
    find_image.add_argument(
        "-t", "--type", choices=["text", "image"], default=None, help="filter by record type"
    )
    find_image.add_argument("-s", "--source", default=None, help="filter by source document")
    find_image.add_argument("--json", action="store_true", help="emit JSON")
    find_image.set_defaults(func=cmd_find_image)

    demo = subparsers.add_parser("demo", help="run the built-in smoke test")
    demo.add_argument("--workdir", default=None, help="where to write the sample assets")
    demo.add_argument("--json", action="store_true", help="emit JSON")
    demo.set_defaults(func=cmd_demo)

    stats = subparsers.add_parser("stats", help="show what is indexed")
    stats.add_argument("--json", action="store_true", help="emit JSON")
    stats.set_defaults(func=cmd_stats)

    reset = subparsers.add_parser("reset", help="clear the collection")
    reset.add_argument("-y", "--yes", action="store_true", help="do not prompt")
    reset.set_defaults(func=cmd_reset)

    settings = subparsers.add_parser("settings", help="show or update configuration")
    settings.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a setting")
    settings.add_argument("--save", action="store_true", help="persist the settings")
    settings.set_defaults(func=cmd_settings)

    # `--profile` is a global flag but is accepted after the subcommand too.
    # SUPPRESS stops the sub-parser default from clobbering the global value.
    for child in subparsers.choices.values():
        child.add_argument(
            "--profile",
            action="store_true",
            default=argparse.SUPPRESS,
            help="print stage timings and model runtime when the command finishes",
        )

    # Likewise `--model`, but only where the store is reached: the fingerprint
    # check runs on every one of these, so each has to be able to name the
    # checkpoint an index was built with.  `settings` is left out - it edits
    # configuration and never opens the collection.
    for name in ("ingest", "query", "find-image", "demo", "stats", "reset"):
        subparsers.choices[name].add_argument(
            "--model",
            default=argparse.SUPPRESS,
            help="use a different embedding checkpoint for this run",
        )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(verbose=getattr(args, "verbose", False))

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    code = 0
    try:
        # Container stage: everything the command does sits underneath it, so
        # the coverage figure can say how much of the run was instrumented.
        with stage("run.total", container=True, command=args.command):
            code = int(args.func(args) or 0)
    except BookRagError as exc:
        hint = getattr(exc, "install_hint", "")
        print(f"error: {exc}", file=sys.stderr)
        if hint:
            print(f"hint: {hint}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        code = 130
    except Exception as exc:  # noqa: BLE001 - top level guard
        log.exception("Unhandled error")
        print(f"error: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            raise
        code = 1

    _finish_monitoring(args)
    return code


def _finish_monitoring(args: argparse.Namespace) -> None:
    """Persist the monitoring report, and print it when ``--profile`` was given.

    Monitoring is best-effort: a failure to render the report must never turn
    a successful command into a failed one.
    """
    try:
        monitor = get_monitor()
        monitor.write_report()
        if not getattr(args, "profile", False):
            return
        # --json owns stdout, so the profile moves to stderr there and the
        # machine-readable output stays parseable.
        stream = sys.stderr if getattr(args, "json", False) else sys.stdout
        print(file=stream)
        print(monitor.report(), file=stream)
    except Exception:  # noqa: BLE001
        log.debug("Could not render the monitoring report", exc_info=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
