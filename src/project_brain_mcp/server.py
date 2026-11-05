"""The MCP server: tools, a resource and a prompt over the Brain."""

from __future__ import annotations

import anyio
import anyio.from_thread
import anyio.to_thread
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .analysis import DriftReport, ImpactReport
from .brain import Brain, BrainError, Hit, HistoryReport
from .history import Commit
from .config import load_settings
from .sources import SourceError

INSTRUCTIONS = """\
Project Brain indexes code repositories and answers questions about them with citations.
Start with index_repo (a local path or an https git URL). Then use ask_codebase for questions,
search for raw hybrid search, and read_lines to open the exact lines a citation points at.
For "why is it like this?" questions, use explain_history on a symbol or line range, and
search_history to find commits by topic. Before changing a function, run impact_of_change to see
what depends on it. find_doc_drift lists documentation that no longer matches the code.
Cite code as path:start-end and commits by short sha."""
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
    commits_added: int
    commits_total: int
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


class CommitInfo(BaseModel):
    sha: str = Field(description="Short sha, ready to cite.")
    author: str
    date: str
    subject: str
    body: str
    pull_requests: list[int]


class AskResult(BaseModel):
    repo: str
    question: str
    instructions: str
    sources: list[Source]
    related_commits: list[CommitInfo] = Field(description="Commits whose messages match the question.")


class Change(BaseModel):
    commit: CommitInfo
    diff: str = Field(description="The part of the commit's diff that touched the range.")


class Authorship(BaseModel):
    sha: str
    author: str
    date: str
    subject: str
    lines: int = Field(description="How many of the current lines come from this commit.")


class PullRequestInfo(BaseModel):
    number: int
    url: str
    title: str | None
    body: str | None
    state: str | None
    merged_at: str | None


class HistoryResult(BaseModel):
    repo: str
    citation: str
    symbol: str | None
    instructions: str
    authorship: list[Authorship]
    changes: list[Change] = Field(description="Commits that changed these lines, newest first.")
    pull_requests: list[PullRequestInfo]


class DependentInfo(BaseModel):
    citation: str
    symbol: str | None
    kind: str
    via: str = Field(description="The name through which it depends on the level above.")
    level: int = Field(description="1 = uses the target directly, 2 = uses something that does.")
    is_test: bool


class ImpactResult(BaseModel):
    target: str = Field(description="path:start-end of the definition analysed.")
    symbol: str
    summary: str
    direct: list[DependentInfo]
    indirect: list[DependentInfo]
    tests: list[str] = Field(description="Citations of tests that exercise the target, directly or not.")
    same_name_definitions: list[str]
    notes: list[str]


class FindingInfo(BaseModel):
    severity: str = Field(description="high (almost certainly wrong), medium (probably wrong), low (worth a look).")
    kind: str
    doc: str = Field(description="path:start-end of the doc section.")
    claim: str
    detail: str
    suggestion: str | None


class DriftResult(BaseModel):
    repo: str
    summary: str
    sections_checked: int
    references_checked: int
    findings: list[FindingInfo]


class RepoSummary(BaseModel):
    name: str
    source: str
    commit: str | None
    indexed_at: str
    files: int
    chunks: int
    commits: int
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
        commits = await _call(lambda: brain.search_history(question, repo, 3))
        instructions = (
            "Answer the question using only these sources. Cite every claim as path:start-end. "
            "If the sources don't contain the answer, say so and suggest a better search or a file to read "
            "with read_lines."
            if hits else
            "Nothing relevant was found. Try search with different keywords, or check the repo is indexed."
        )
        return AskResult(repo=brain.resolve_repo(repo), question=question, instructions=instructions,
                         sources=[_source(h) for h in hits], related_commits=[_commit(c) for c in commits])

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    async def explain_history(repo: str | None = None, symbol: str | None = None, path: str | None = None,
                              start_line: int | None = None, end_line: int | None = None,
                              max_commits: int = 8) -> HistoryResult:
        """Why code looks the way it does. Give a `symbol` (e.g. "unsign" or "Signer.sign"), or a `path`
        with optional line range. Returns who wrote the current lines, every commit that changed them
        (with the relevant diff), and the pull requests behind those commits."""
        report = await _call(lambda: brain.explain_history(repo, path, symbol, start_line, end_line, max_commits))
        return _history(report)

    @server.tool(annotations=READ_ONLY)
    async def impact_of_change(symbol: str, repo: str | None = None, path: str | None = None,
                               depth: int = 2) -> ImpactResult:
        """What breaks if `symbol` changes: code that uses it directly, code that uses that code
        (up to `depth`, max 3), and the tests involved. Python is analysed structurally."""
        report = await _call(lambda: brain.impact_of_change(repo, symbol, path, depth))
        return _impact(report)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    async def find_doc_drift(repo: str | None = None, path_prefix: str | None = None, check_history: bool = True,
                             limit: int = 50) -> DriftResult:
        """Documentation that no longer matches the code: files, functions, settings and flags that don't
        exist; examples passing parameters a function doesn't take; imports that would fail; and docs last
        edited long before the code they describe changed. Each finding cites the doc section."""
        report = await _call(lambda: brain.find_doc_drift(repo, path_prefix, check_history, max(1, min(limit, 200))))
        return _drift(brain.resolve_repo(repo), report)

    @server.tool(annotations=READ_ONLY)
    async def search_history(query: str, repo: str | None = None, k: int = 5) -> list[CommitInfo]:
        """Find commits by what their messages say, e.g. "fix timezone" or "drop python 3.8"."""
        commits = await _call(lambda: brain.search_history(query, repo, max(1, min(k, MAX_RESULTS))))
        return [_commit(c) for c in commits]

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
    return await _call(lambda: brain.search(query, repo, k, path_prefix, kinds))


async def _call(fn):
    """Run blocking engine work off the event loop; user-facing errors become tool errors."""
    try:
        return await anyio.to_thread.run_sync(fn)
    except BrainError as exc:
        raise ToolError(str(exc)) from exc


def _commit(commit: Commit) -> CommitInfo:
    return CommitInfo(sha=commit.sha[:10], author=commit.author, date=commit.date, subject=commit.subject,
                      body=commit.body, pull_requests=list(commit.prs))


def _impact(report: ImpactReport) -> ImpactResult:
    def info(d) -> DependentInfo:
        return DependentInfo(citation=d.chunk.citation, symbol=d.chunk.symbol, kind=d.chunk.kind, via=d.via,
                             level=d.level, is_test=d.is_test)

    direct = [info(d) for d in report.dependents if d.level == 1]
    indirect = [info(d) for d in report.dependents if d.level > 1]
    tests = [d.chunk.citation for d in report.dependents if d.is_test]
    files = len({d.chunk.path for d in report.dependents})
    summary = (f"{report.symbol} is used directly in {len(direct)} place(s) and indirectly in {len(indirect)}, "
               f"across {files} file(s); {len(tests)} test chunk(s) exercise it.")
    return ImpactResult(target=f"{report.path}:{report.start_line}-{report.end_line}", symbol=report.symbol,
                        summary=summary, direct=direct, indirect=indirect, tests=tests,
                        same_name_definitions=report.same_name_definitions, notes=report.notes)


def _drift(repo: str, report: DriftReport) -> DriftResult:
    counts = {level: sum(f.severity == level for f in report.findings) for level in ("high", "medium", "low")}
    summary = (f"{len(report.findings)} finding(s) in {report.sections_checked} doc section(s): "
               f"{counts['high']} high, {counts['medium']} medium, {counts['low']} low."
               if report.findings else f"No drift found in {report.sections_checked} doc section(s).")
    return DriftResult(repo=repo, summary=summary, sections_checked=report.sections_checked,
                       references_checked=report.references_checked,
                       findings=[FindingInfo(**f.__dict__) for f in report.findings])


def _history(report: HistoryReport) -> HistoryResult:
    citation = f"{report.path}:{report.start_line}-{report.end_line}"
    instructions = (
        "Explain why this code looks the way it does, oldest change first: what each commit changed and the "
        "reason its message or PR gives. Cite commits by short sha and code as path:start-end. If the messages "
        "don't say why, say so rather than guessing."
        if report.changes else "No commits changed these lines (they may be uncommitted)."
    )
    return HistoryResult(
        repo=report.repo, citation=citation, symbol=report.symbol, instructions=instructions,
        authorship=[Authorship(sha=b.sha[:10], author=b.author, date=b.date, subject=b.subject, lines=b.lines)
                    for b in report.blame],
        changes=[Change(commit=_commit(c.commit), diff=c.diff) for c in report.changes],
        pull_requests=[PullRequestInfo(**pr.__dict__) for pr in report.pull_requests],
    )


def _source(hit: Hit) -> Source:
    chunk = hit.chunk
    matched = [name for name, rank in (("keyword", hit.keyword_rank), ("vector", hit.vector_rank)) if rank]
    snippet = chunk.text if len(chunk.text) <= SNIPPET_CHARS else chunk.text[:SNIPPET_CHARS] + "\n…"
    return Source(citation=chunk.citation, path=chunk.path, start_line=chunk.start_line, end_line=chunk.end_line,
                  kind=chunk.kind, symbol=chunk.symbol, score=hit.score, matched_by=matched, snippet=snippet)


def _summary(info) -> RepoSummary:
    return RepoSummary(name=info.name, source=info.source, commit=info.commit, indexed_at=info.indexed_at,
                       files=info.files, chunks=info.chunks, commits=info.commits, languages=info.languages)
