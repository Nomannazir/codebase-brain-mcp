"""The engine behind the MCP tools: index repositories, search them, and gather cited evidence for questions."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from . import analysis, history, sources
from .analysis import DriftReport, ImpactReport, definition_span
from .chunking import chunk_file
from .config import Settings
from .embeddings import Embedder, make_embedder
from .history import BlameShare, Commit, LineChange, PullRequest
from .store import Store, StoredChunk

RRF_K = 60
"""Reciprocal-rank-fusion constant; 60 is the value from the original paper and works well untuned."""
CANDIDATES = 50
EMBED_BATCH_FILES = 16
HISTORY_LIMIT = 5_000
"""Commits read per indexing run; enough for the history that matters, bounded for huge repos."""
MAX_PULL_REQUESTS = 5
_NAME = re.compile(r"^[A-Za-z0-9._-]{1,100}$")

ProgressFn = Callable[[int, int, str], None]
SearchMode = Literal["hybrid", "keyword", "vector"]
SEARCH_MODES: tuple[SearchMode, ...] = ("hybrid", "keyword", "vector")


class BrainError(ValueError):
    """A request the brain can't serve, with a message meant for the user."""


@dataclass(frozen=True)
class IndexReport:
    repo: str
    root: str
    commit: str | None
    files_seen: int
    files_indexed: int
    files_unchanged: int
    files_removed: int
    chunks_total: int
    commits_added: int
    commits_total: int
    seconds: float
    full_rebuild: bool


@dataclass(frozen=True)
class Hit:
    chunk: StoredChunk
    score: float
    keyword_rank: int | None
    vector_rank: int | None


@dataclass(frozen=True)
class HistoryReport:
    repo: str
    path: str
    start_line: int
    end_line: int
    symbol: str | None
    blame: list[BlameShare]
    changes: list[LineChange]
    pull_requests: list[PullRequest]
    github: str | None


@dataclass(frozen=True)
class RepoInfo:
    name: str
    source: str
    root: str
    commit: str | None
    indexed_at: str
    files: int
    chunks: int
    commits: int = 0
    languages: dict[str, int] = field(default_factory=dict)


class Brain:
    def __init__(self, settings: Settings, embedder: Embedder | None = None):
        self.settings = settings
        self.embedder = embedder or make_embedder(settings)
        self._stores: dict[str, Store] = {}

    # ---- indexing -------------------------------------------------------------------------
    def index(self, source: str, name: str | None = None, progress: ProgressFn | None = None) -> IndexReport:
        """Index (or re-index) a local directory or an https git URL. Unchanged files are skipped."""
        started = time.monotonic()
        name = self._valid_name(name or sources.default_name(source))
        if not self.settings.allow_local_paths and not sources.is_url(source):
            raise BrainError("This server only indexes public https git URLs, not local paths.")
        root = sources.resolve(source, self.settings.clone_dir)
        files = sources.iter_files(root, self.settings.max_file_bytes)
        if len(files) > self.settings.max_files:
            raise BrainError(
                f"{source} has {len(files)} indexable files, over this server's limit of {self.settings.max_files}."
            )
        store = self._store(name)

        meta = store.get_meta()
        full_rebuild = (
            meta.get("embed_model") != self.embedder.model_id or meta.get("schema_version") != store.schema_version
        )
        if full_rebuild:
            store.clear()
        known = store.file_hashes()
        current = {f.path: f for f in files}

        removed = [path for path in known if path not in current]
        for path in removed:
            store.remove_file(path)

        changed = [f for f in files if known.get(f.path) != _sha(f.text)]
        notify = progress or (lambda done, total, message: None)
        notify(0, len(changed), f"{len(changed)} of {len(files)} files need indexing")
        for start in range(0, len(changed), EMBED_BATCH_FILES):
            batch = changed[start : start + EMBED_BATCH_FILES]
            per_file = [chunk_file(f.path, f.language, f.text) for f in batch]
            vectors = self.embedder.embed_documents([_embedding_text(c) for chunks in per_file for c in chunks])
            offset = 0
            for file, chunks in zip(batch, per_file, strict=True):
                store.replace_file(
                    file.path, _sha(file.text), file.language, chunks, vectors[offset : offset + len(chunks)]
                )
                offset += len(chunks)
            notify(min(start + len(batch), len(changed)), len(changed), batch[-1].path)

        commit = sources.head_commit(root)
        commits_added = 0
        if history.is_git_repo(root):
            notify(len(changed), len(changed), "reading git history")
            since = None if full_rebuild else meta.get("history_head") or None
            commits_added = store.add_commits(history.commits(root, since, HISTORY_LIMIT))
        store.set_meta(
            source=source,
            root=str(root),
            commit=commit or "",
            embed_model=self.embedder.model_id,
            history_head=commit or "",
            schema_version=store.schema_version,
            indexed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        return IndexReport(
            repo=name,
            root=str(root),
            commit=commit,
            files_seen=len(files),
            files_indexed=len(changed),
            files_unchanged=len(files) - len(changed),
            files_removed=len(removed),
            chunks_total=store.counts()["chunks"],
            commits_added=commits_added,
            commits_total=store.commit_count(),
            seconds=round(time.monotonic() - started, 2),
            full_rebuild=full_rebuild,
        )

    # ---- retrieval ------------------------------------------------------------------------
    def search(
        self,
        query: str,
        repo: str | None = None,
        k: int = 8,
        path_prefix: str | None = None,
        kinds: list[str] | None = None,
        mode: SearchMode = "hybrid",
    ) -> list[Hit]:
        """Hybrid search: BM25 keywords and vector similarity, merged with reciprocal-rank fusion.
        ``mode`` can switch to one retriever alone; the evals use that to measure what each adds."""
        query = query.strip()
        if not query:
            raise BrainError("The query is empty.")
        if mode not in SEARCH_MODES:
            raise BrainError(f"mode must be one of {', '.join(SEARCH_MODES)}.")
        store = self._store(self.resolve_repo(repo))
        keyword_ids = store.keyword_search(query, CANDIDATES) if mode != "vector" else []
        vector_ids = store.vector_search(self.embedder.embed_query(query), CANDIDATES) if mode != "keyword" else []

        scores: dict[int, float] = {}
        for ranked in (keyword_ids, vector_ids):
            for rank, chunk_id in enumerate(ranked, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        chunks = store.get_chunks(list(scores))
        keyword_rank = {cid: r for r, cid in enumerate(keyword_ids, start=1)}
        vector_rank = {cid: r for r, cid in enumerate(vector_ids, start=1)}

        hits = []
        for chunk_id in sorted(scores, key=scores.__getitem__, reverse=True):
            chunk = chunks[chunk_id]
            if path_prefix and not chunk.path.startswith(path_prefix):
                continue
            if kinds and chunk.kind not in kinds:
                continue
            hits.append(
                Hit(
                    chunk=chunk,
                    score=round(scores[chunk_id], 5),
                    keyword_rank=keyword_rank.get(chunk_id),
                    vector_rank=vector_rank.get(chunk_id),
                )
            )
            if len(hits) == k:
                break
        return hits

    # ---- history --------------------------------------------------------------------------
    def explain_history(
        self,
        repo: str | None,
        path: str | None = None,
        symbol: str | None = None,
        start: int | None = None,
        end: int | None = None,
        max_commits: int = 8,
    ) -> HistoryReport:
        """How a function, a line range or a whole file came to be: blame shares, the commits that
        changed it (with the relevant diff), and the pull requests those commits came from."""
        name = self.resolve_repo(repo)
        store = self._store(name)
        root = Path(store.get_meta()["root"])
        if symbol:
            matches = store.find_symbol(symbol, path)
            if not matches:
                where = f" in {path}" if path else ""
                raise BrainError(f"No function, class or section called {symbol!r}{where}. Try search first.")
            path, start, end, symbol = definition_span(matches)
        if not path:
            raise BrainError("Give a path (optionally with start and end lines) or a symbol.")
        target = self._file_in(root, path)
        if not history.is_git_repo(root):
            raise BrainError(f"{name} isn't a git repository, so it has no history to explain.")
        total = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
        start, end = max(1, start or 1), min(total, end or total)
        if start > end:
            raise BrainError(f"{path} has {total} lines.")
        try:
            changes = history.line_history(root, path, start, end, max(1, min(max_commits, 30)))
            shares = history.blame(root, path, start, end)[:5]
        except sources.SourceError as exc:
            raise BrainError(f"git couldn't read the history of {path}: {exc}") from exc
        slug = history.github_slug(root)
        return HistoryReport(
            repo=name,
            path=path,
            start_line=start,
            end_line=end,
            symbol=symbol,
            blame=shares,
            changes=changes,
            pull_requests=self._pull_requests(store, slug, changes),
            github=slug,
        )

    def search_history(self, query: str, repo: str | None = None, k: int = 5) -> list[Commit]:
        """Commits whose message matches the query (BM25)."""
        if not query.strip():
            raise BrainError("The query is empty.")
        return self._store(self.resolve_repo(repo)).search_commits(query, k)

    def _pull_requests(self, store: Store, slug: str | None, changes: list[LineChange]) -> list[PullRequest]:
        numbers = list(dict.fromkeys(n for change in changes for n in change.commit.prs))[:MAX_PULL_REQUESTS]
        if not slug:
            return []
        prs = []
        for number in numbers:
            pr = store.cached_pull_request(number)
            if pr is None:
                if self.settings.fetch_pull_requests:
                    pr = history.fetch_pull_request(slug, number, self.settings.github_token)
                    store.cache_pull_request(pr)
                else:
                    pr = PullRequest(number=number, url=f"https://github.com/{slug}/pull/{number}")
            prs.append(pr)
        return prs

    # ---- structure ------------------------------------------------------------------------
    def impact_of_change(self, repo: str | None, symbol: str, path: str | None = None, depth: int = 2) -> ImpactReport:
        """What uses ``symbol``, directly and transitively, including the tests that exercise it."""
        store = self._store(self.resolve_repo(repo))
        matches = store.find_symbol(symbol, path)
        if not matches:
            where = f" in {path}" if path else ""
            raise BrainError(f"No function, class or method called {symbol!r}{where}. Try search first.")
        target_path, start, end, _ = definition_span(matches)
        target = next(c for c in matches if c.path == target_path)
        return analysis.impact_of_change(store, target, (start, end), depth)

    def find_doc_drift(
        self, repo: str | None, path_prefix: str | None = None, check_history: bool = True, limit: int = 50
    ) -> DriftReport:
        """Documentation that no longer matches the code."""
        store = self._store(self.resolve_repo(repo))
        return analysis.find_doc_drift(store, Path(store.get_meta()["root"]), path_prefix, check_history, limit)

    # ---- repositories ---------------------------------------------------------------------
    def repos(self) -> list[RepoInfo]:
        index_dir = self.settings.index_dir
        names = sorted(p.stem for p in index_dir.glob("*.db")) if index_dir.is_dir() else []
        return [info for info in (self.repo_info(n) for n in names) if info is not None]

    def repo_info(self, name: str) -> RepoInfo | None:
        store = self._store(name)
        meta = store.get_meta()
        if "indexed_at" not in meta:
            return None
        counts = store.counts()
        return RepoInfo(
            name=name,
            source=meta.get("source", ""),
            root=meta.get("root", ""),
            commit=meta.get("commit") or None,
            indexed_at=meta["indexed_at"],
            files=counts["files"],
            chunks=counts["chunks"],
            commits=store.commit_count(),
            languages=store.languages(),
        )

    def resolve_repo(self, repo: str | None) -> str:
        """The repo to use: the one named, or the only one indexed."""
        if repo:
            name = self._valid_name(repo)
            if not (self.settings.index_dir / f"{name}.db").exists():
                raise BrainError(f"No index named {name!r}. Run index_repo first.")
            return name
        names = [info.name for info in self.repos()]
        if len(names) == 1:
            return names[0]
        if not names:
            raise BrainError("Nothing is indexed yet. Run index_repo with a local path or an https git URL.")
        raise BrainError(f"Several repos are indexed ({', '.join(names)}); say which one with repo=.")

    def read_lines(self, repo: str | None, path: str, start: int, end: int) -> str:
        """Exact lines from an indexed file, for following up on a citation."""
        info = self.repo_info(self.resolve_repo(repo))
        if info is None:
            raise BrainError("That repo isn't indexed.")
        target = self._file_in(Path(info.root), path)
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start, end = max(1, start), min(len(lines), end)
        if start > end:
            raise BrainError(f"{path} has {len(lines)} lines.")
        return "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))

    @staticmethod
    def _file_in(root: Path, path: str) -> Path:
        """``path`` inside ``root``, refusing anything that escapes it."""
        root = root.resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise BrainError(f"{path!r} isn't a file in this repository.")
        return target

    def close(self) -> None:
        for store in self._stores.values():
            store.close()
        self._stores.clear()

    def store_for(self, repo: str | None) -> Store:
        """The index of an indexed repo, for read-only analysis such as evals."""
        return self._store(self.resolve_repo(repo))

    def _store(self, name: str) -> Store:
        if name not in self._stores:
            self._stores[name] = Store(self.settings.index_dir / f"{name}.db")
        return self._stores[name]

    @staticmethod
    def _valid_name(name: str) -> str:
        if not _NAME.match(name) or name.startswith("."):
            raise BrainError("Repo names may use letters, digits, '.', '_' and '-' only.")
        return name


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _embedding_text(chunk) -> str:
    """What the model sees: where the chunk lives and what it's called, then the code itself."""
    header = f"File: {chunk.path}"
    if chunk.symbol:
        header += f"\n{chunk.kind.capitalize()}: {chunk.symbol}"
    return f"{header}\n\n{chunk.text}"
