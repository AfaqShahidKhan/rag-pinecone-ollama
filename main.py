"""
main.py

CLI entry point. Builds the Container (composition root) once, then
dispatches to the requested service.
Phase 4: adds the `watch` command for event-driven landing zone ingestion.
Step 3: adds `--config` for per-user YAML config profiles.
Logging step: wraps dispatch in a top-level try/except so an uncaught
exception is always logged (with traceback) to the rotating file before
the process exits — not just whatever happened to reach a logger.error()
call before the crash.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.composition import Container


def _print_token(token: str) -> None:
    print(token, end="", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="RAG over Pinecone CLI")
    parser.add_argument(
        "--config", dest="config_file", default=None,
        help=(
            "Path to a per-user YAML config (e.g. config/user_afaq.yml). "
            "Overrides config/default.yml, which overrides .env / defaults."
        ),
    )
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


def _dispatch(container: Container, args: argparse.Namespace) -> int:
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

    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        container = Container.bootstrap(token_sink=_print_token, config_file=args.config_file)
        return _dispatch(container, args)
    except Exception:
        # Ensures a crash is always traceable in logs/rag.log, even if it
        # happens before any component-level logger.error() call runs.
        logging.getLogger("main").exception("Unhandled exception during CLI execution.")
        print("\nAn unexpected error occurred. See logs/rag.log for the full traceback.")
        return 1


if __name__ == "__main__":
    sys.exit(main())