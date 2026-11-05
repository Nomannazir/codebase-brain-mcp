"""Command line: run the MCP server (default), or index and search from the shell for debugging."""

from __future__ import annotations

import argparse
import sys

from .brain import Brain, BrainError
from .config import load_settings
from .sources import SourceError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="project-brain", description="Project Brain MCP server.")
    commands = parser.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="Run the MCP server (the default).")
    serve.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    index = commands.add_parser("index", help="Index a local path or an https git URL.")
    index.add_argument("source")
    index.add_argument("--name")
    search = commands.add_parser("search", help="Hybrid search an indexed repo.")
    search.add_argument("query")
    search.add_argument("--repo")
    search.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)

    if args.command in (None, "serve"):
        from .server import build_server

        server = build_server()
        if getattr(args, "transport", "stdio") == "streamable-http":
            server.run("streamable-http", host=args.host, port=args.port)
        else:
            server.run("stdio")
        return 0

    brain = Brain(load_settings())
    try:
        if args.command == "index":
            report = brain.index(args.source, name=args.name,
                                 progress=lambda done, total, msg: print(f"  [{done}/{total}] {msg}", file=sys.stderr))
            print(report)
        else:
            for hit in brain.search(args.query, args.repo, args.k):
                print(f"{hit.score:.4f}  {hit.chunk.citation}  {hit.chunk.kind} {hit.chunk.symbol or ''}")
    except (BrainError, SourceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        brain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
