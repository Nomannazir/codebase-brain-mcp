"""The MCP server: tools, a resource and a prompt over the Brain."""

from __future__ import annotations

import anyio
import anyio.from_thread
import anyio.to_thread
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .brain import Brain, BrainError, Hit
from .config import load_settings
from .sources import SourceError

INSTRUCTIONS = """\
Project Brain indexes code repositories and answers questions about them with citations.
Start with index_repo (a local path or an https git URL). Then use ask_codebase for questions,
search for raw hybrid search, and read_lines to open the exact lines a citation points at.
Always cite sources as path:start-end."""
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
MAX_RESULTS = 25
SNIPPET_CHARS = 1_800


class IndexResult(BaseModel):
    repo: str
    root: str
    commit: str | None
    files_seen: int
    files_indexed: int = Field(description="Files embedded this run (new or changed).")
    files_unchanged: int
    files_removed: int
    chunks_total: int
    seconds: float
    full_rebuild: bool


class Source(BaseModel):
    citation: str = Field(description="path:start-end, ready to quote.")
    path: str
    start_line: int
    end_line: int
    kind: str
    symbol: str | None
    score: float
    matched_by: list[str] = Field(description="Which retrievers found it: keyword, vector, or both.")
    snippet: str


class SearchResult(BaseModel):
    repo: str
    query: str
    results: list[Source]


class AskResult(BaseModel):
    repo: str
    question: str
    instructions: str
    sources: list[Source]


class RepoSummary(BaseModel):
    name: str
    source: str
    commit: str | None
    indexed_at: str
    files: int
    chunks: int
    languages: dict[str, int]


def build_server(brain: Brain | None = None) -> MCPServer:
    brain = brain or Brain(load_settings())
    server = MCPServer("project-brain", instructions=INSTRUCTIONS, version="0.1.0")

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True,
                                             open_world_hint=True))
    async def index_repo(source: str, ctx: Context, name: str | None = None) -> IndexResult:
        """Index a repository so it can be searched. `source` is a local directory or an https git URL
        (cloned shallowly). Re-running only re-embeds files that changed since the last run."""

        def progress(done: int, total: int, message: str) -> None:
            anyio.from_thread.run(ctx.report_progress, done, total, message)

        try:
            report = await anyio.to_thread.run_sync(lambda: brain.index(source, name=name, progress=progress))
        except (BrainError, SourceError) as exc:
            raise ToolError(str(exc)) from exc
        return IndexResult(**report.__dict__)

    @server.tool(annotations=READ_ONLY)
    async def search(query: str, repo: str | None = None, k: int = 8, path_prefix: str | None = None,
                     kinds: list[str] | None = None) -> SearchResult:
        """Hybrid search (BM25 keywords + embeddings, fused by rank). Filter by `path_prefix`
        (e.g. "src/auth/") or `kinds` (function, class, method, module, section, code)."""
        hits = await _search(brain, query, repo, k, path_prefix, kinds)
        return SearchResult(repo=brain.resolve_repo(repo), query=query, results=[_source(h) for h in hits])

    @server.tool(annotations=READ_ONLY)
    async def ask_codebase(question: str, repo: str | None = None, k: int = 8) -> AskResult:
        """Gather the code and docs most relevant to a question, as numbered, cited sources.
        Answer from these sources only, citing each claim as path:start-end."""
        hits = await _search(brain, question, repo, k, None, None)
        instructions = (
            "Answer the question using only these sources. Cite every claim as path:start-end. "
            "If the sources don't contain the answer, say so and suggest a better search or a file to read "
            "with read_lines."
            if hits else
            "Nothing relevant was found. Try search with different keywords, or check the repo is indexed."
        )
        return AskResult(repo=brain.resolve_repo(repo), question=question, instructions=instructions,
                         sources=[_source(h) for h in hits])

    @server.tool(annotations=READ_ONLY)
    async def read_lines(path: str, start: int, end: int, repo: str | None = None) -> str:
        """Exact, numbered lines from a file in an indexed repo, e.g. to open a citation in full."""
        try:
            return brain.read_lines(repo, path, start, end)
        except BrainError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(annotations=READ_ONLY)
    async def list_repos() -> list[RepoSummary]:
        """Repositories indexed so far, with size, languages and the commit they were indexed at."""
        return [_summary(info) for info in brain.repos()]

    @server.resource("brain://repos", name="indexed-repos", mime_type="application/json",
                     description="Every indexed repository and its stats.")
    def repos_resource() -> str:
        return "[" + ",".join(_summary(info).model_dump_json() for info in brain.repos()) + "]"

    @server.prompt(description="Answer a question about a codebase with strict citations.")
    def answer_with_citations(question: str, repo: str = "") -> str:
        target = f" in the {repo} repo" if repo else ""
        return (
            f"Question about the codebase{target}: {question}\n\n"
            "1. Call ask_codebase with this question"
            + (f' and repo="{repo}"' if repo else "") + ".\n"
            "2. If a source looks relevant but is cut off, open it with read_lines.\n"
            "3. Answer in a few short paragraphs. Cite every claim as path:start-end.\n"
            "4. If the sources don't answer it, say what's missing instead of guessing."
        )

    return server


async def _search(brain: Brain, query: str, repo: str | None, k: int, path_prefix: str | None,
                  kinds: list[str] | None) -> list[Hit]:
    k = max(1, min(k, MAX_RESULTS))
    try:
        return await anyio.to_thread.run_sync(lambda: brain.search(query, repo, k, path_prefix, kinds))
    except BrainError as exc:
        raise ToolError(str(exc)) from exc


def _source(hit: Hit) -> Source:
    chunk = hit.chunk
    matched = [name for name, rank in (("keyword", hit.keyword_rank), ("vector", hit.vector_rank)) if rank]
    snippet = chunk.text if len(chunk.text) <= SNIPPET_CHARS else chunk.text[:SNIPPET_CHARS] + "\n…"
    return Source(citation=chunk.citation, path=chunk.path, start_line=chunk.start_line, end_line=chunk.end_line,
                  kind=chunk.kind, symbol=chunk.symbol, score=hit.score, matched_by=matched, snippet=snippet)


def _summary(info) -> RepoSummary:
    return RepoSummary(name=info.name, source=info.source, commit=info.commit, indexed_at=info.indexed_at,
                       files=info.files, chunks=info.chunks, languages=info.languages)
