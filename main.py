"""
main.py

CLI entry point. Builds the Container (composition root) once, then
dispatches to the requested service.
Phase 4: adds the `watch` command for event-driven landing zone ingestion.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.composition import Container


def _print_token(token: str) -> None:
    print(token, end="", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="RAG over Pinecone CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest (batch)
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the vector store")
    ingest_parser.add_argument(
        "source", nargs="?", default=None,
        help="Path to a file or directory. Defaults to <project_root>/data/landing_zone.",
    )

    # ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question against the index")
    ask_parser.add_argument("question", help="The question to ask")
    ask_parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    ask_parser.add_argument("--no-stream", action="store_true", dest="no_stream")

    # eval
    eval_parser = subparsers.add_parser("eval", help="Run the default evaluation suite")
    eval_parser.add_argument("--stream", action="store_true")

    # debug
    debug_parser = subparsers.add_parser("debug", help="Debug a single retrieval + prompt")
    debug_parser.add_argument("question", help="The question to debug")
    debug_parser.add_argument("--top-k", type=int, default=5, dest="top_k")

    # watch (Phase 4 — landing zone)
    watch_parser = subparsers.add_parser(
        "watch",
        help="Watch a directory for new files and ingest them automatically",
    )
    watch_parser.add_argument(
        "source", nargs="?", default=None,
        help="Directory to watch. Defaults to <project_root>/data/landing_zone.",
    )
    watch_parser.add_argument(
        "--recursive", action="store_true",
        help="Also watch subdirectories.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    container = Container.bootstrap(token_sink=_print_token)

    if args.command == "ingest":
        source = Path(args.source) if args.source else container.settings.data_raw
        total = container.ingestion_service.ingest_path(source)
        print(f"Indexed {total} vectors.")
        return 0

    if args.command == "ask":
        response = container.rag_query_service.ask(
            question=args.question,
            top_k=args.top_k,
            stream=not args.no_stream,
        )
        if args.no_stream:
            print(response.answer)
        return 0

    if args.command == "eval":
        container.evaluation_service.run_eval(stream=args.stream)
        return 0

    if args.command == "debug":
        container.evaluation_service.debug_query(args.question, top_k=args.top_k)
        return 0

    if args.command == "watch":
        source = Path(args.source) if args.source else container.settings.data_raw

        if not source.exists():
            print(f"Error: directory not found: '{source}'")
            return 1

        # Build a dedicated watcher (with recursive flag if requested)
        watcher = container._services.create_landing_zone_watcher(
            recursive=getattr(args, "recursive", False)
        )
        watcher.start(source)

        print(f"Watching '{source}' for new files. Press Ctrl+C to stop.")
        try:
            while watcher.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            watcher.stop()
            print("\nWatcher stopped.")

        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())