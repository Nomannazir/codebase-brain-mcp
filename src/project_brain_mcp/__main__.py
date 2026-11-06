"""Command line: run the MCP server (default), or index and search from the shell for debugging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    evaluate = commands.add_parser("eval", help="Measure retrieval quality on a benchmark dataset.")
    evaluate.add_argument("dataset", type=Path, help="A TOML dataset, e.g. evals/datasets/itsdangerous.toml.")
    evaluate.add_argument("--modes", default="hybrid,keyword,vector", help="Comma-separated search modes.")
    evaluate.add_argument("--out", type=Path, help="Directory for the Markdown and JSON reports.")
    args = parser.parse_args(argv)

    if args.command == "eval":
        return _evaluate(args)

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
            report = brain.index(
                args.source,
                name=args.name,
                progress=lambda done, total, msg: print(f"  [{done}/{total}] {msg}", file=sys.stderr),
            )
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


def _evaluate(args: argparse.Namespace) -> int:
    from . import evals
    from .brain import SEARCH_MODES

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if unknown := [m for m in modes if m not in SEARCH_MODES]:
        print(f"error: unknown mode(s) {', '.join(unknown)}; use {', '.join(SEARCH_MODES)}", file=sys.stderr)
        return 2
    brain = Brain(load_settings())
    try:
        dataset = evals.load_dataset(args.dataset)
        repo = evals.prepare(brain, dataset)
        report = evals.run(brain, repo, dataset, modes)  # type: ignore[arg-type]
    except (evals.DatasetError, BrainError, SourceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        brain.close()
    markdown = evals.to_markdown(report)
    print(markdown)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{dataset.name}.md").write_text(markdown, encoding="utf-8")
        (args.out / f"{dataset.name}.json").write_text(json.dumps(evals.to_json(report), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
