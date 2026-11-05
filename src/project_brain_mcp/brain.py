"""The engine behind the MCP tools: index repositories, search them, and gather cited evidence for questions."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import sources
from .chunking import chunk_file
from .config import Settings
from .embeddings import Embedder, make_embedder
from .store import StoredChunk, Store

RRF_K = 60
"""Reciprocal-rank-fusion constant; 60 is the value from the original paper and works well untuned."""
CANDIDATES = 50
EMBED_BATCH_FILES = 16
_NAME = re.compile(r"^[A-Za-z0-9._-]{1,100}$")

ProgressFn = Callable[[int, int, str], None]


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
    seconds: float
    full_rebuild: bool


@dataclass(frozen=True)
class Hit:
    chunk: StoredChunk
    score: float
    keyword_rank: int | None
    vector_rank: int | None


@dataclass(frozen=True)
class RepoInfo:
    name: str
    source: str
    root: str
    commit: str | None
    indexed_at: str
    files: int
    chunks: int
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
        root = sources.resolve(source, self.settings.clone_dir)
        files = sources.iter_files(root, self.settings.max_file_bytes)
        store = self._store(name)

        meta = store.get_meta()
        full_rebuild = meta.get("embed_model") != self.embedder.model_id
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
            batch = changed[start:start + EMBED_BATCH_FILES]
            per_file = [chunk_file(f.path, f.language, f.text) for f in batch]
            vectors = self.embedder.embed_documents([_embedding_text(c) for chunks in per_file for c in chunks])
            offset = 0
            for file, chunks in zip(batch, per_file, strict=True):
                store.replace_file(file.path, _sha(file.text), file.language, chunks,
                                   vectors[offset:offset + len(chunks)])
                offset += len(chunks)
            notify(min(start + len(batch), len(changed)), len(changed), batch[-1].path)

        commit = sources.head_commit(root)
        store.set_meta(source=source, root=str(root), commit=commit or "", embed_model=self.embedder.model_id,
                       indexed_at=datetime.now(UTC).isoformat(timespec="seconds"))
        return IndexReport(
            repo=name, root=str(root), commit=commit, files_seen=len(files), files_indexed=len(changed),
            files_unchanged=len(files) - len(changed), files_removed=len(removed),
            chunks_total=store.counts()["chunks"], seconds=round(time.monotonic() - started, 2),
            full_rebuild=full_rebuild,
        )

    # ---- retrieval ------------------------------------------------------------------------
    def search(self, query: str, repo: str | None = None, k: int = 8, path_prefix: str | None = None,
               kinds: list[str] | None = None) -> list[Hit]:
        """Hybrid search: BM25 keywords and vector similarity, merged with reciprocal-rank fusion."""
        query = query.strip()
        if not query:
            raise BrainError("The query is empty.")
        store = self._store(self.resolve_repo(repo))
        keyword_ids = store.keyword_search(query, CANDIDATES)
        vector_ids = store.vector_search(self.embedder.embed_query(query), CANDIDATES)

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
            hits.append(Hit(chunk=chunk, score=round(scores[chunk_id], 5),
                            keyword_rank=keyword_rank.get(chunk_id), vector_rank=vector_rank.get(chunk_id)))
            if len(hits) == k:
                break
        return hits

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
        return RepoInfo(name=name, source=meta.get("source", ""), root=meta.get("root", ""),
                        commit=meta.get("commit") or None, indexed_at=meta["indexed_at"],
                        files=counts["files"], chunks=counts["chunks"], languages=store.languages())

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
        root = Path(info.root).resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise BrainError(f"{path!r} isn't a file in {info.name}.")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start, end = max(1, start), min(len(lines), end)
        if start > end:
            raise BrainError(f"{path} has {len(lines)} lines.")
        return "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))

    def close(self) -> None:
        for store in self._stores.values():
            store.close()
        self._stores.clear()

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
