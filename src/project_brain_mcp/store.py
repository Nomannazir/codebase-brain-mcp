"""One SQLite file per indexed repository: files, chunks, their vectors, and an FTS5 keyword index."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .history import Commit, PullRequest

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, sha TEXT NOT NULL, language TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    kind TEXT NOT NULL,
    symbol TEXT,
    text TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks (path);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (body, symbol, path, tokenize = 'porter unicode61');
CREATE TABLE IF NOT EXISTS commits (
    sha TEXT PRIMARY KEY,
    author TEXT NOT NULL,
    date TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    prs TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS commits_fts USING fts5 (subject, body, sha UNINDEXED, tokenize = 'porter unicode61');
CREATE TABLE IF NOT EXISTS pull_requests (number INTEGER PRIMARY KEY, data TEXT NOT NULL);
"""
_WORD = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


@dataclass(frozen=True)
class StoredChunk:
    id: int
    path: str
    start_line: int
    end_line: int
    kind: str
    symbol: str | None
    text: str

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


def identifier_terms(text: str) -> str:
    """``getUserById`` and ``get_user_by_id`` both also yield ``get user by id``, so keywords match either style."""
    extra = []
    for word in _WORD.findall(text):
        parts = [p.lower() for piece in word.split("_") for p in _CAMEL.findall(piece)]
        if len(parts) > 1:
            extra.extend(parts)
    return " ".join(extra)


def _match_expression(query: str) -> str:
    """An FTS5 OR-query of the query's words and identifier parts, each quoted so nothing is FTS syntax."""
    terms = {t.lower() for t in _WORD.findall(query)} | set(identifier_terms(query).split())
    return " OR ".join(f'"{t}"' for t in sorted(t for t in terms if len(t) > 1))


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._matrix: tuple[np.ndarray, np.ndarray] | None = None

    def close(self) -> None:
        self._db.close()

    # ---- metadata -------------------------------------------------------------------------
    def get_meta(self) -> dict[str, str]:
        return {row["key"]: row["value"] for row in self._db.execute("SELECT key, value FROM meta")}

    def set_meta(self, **values: str) -> None:
        with self._lock, self._db:
            self._db.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", values.items())

    # ---- files and chunks -----------------------------------------------------------------
    def file_hashes(self) -> dict[str, str]:
        return {row["path"]: row["sha"] for row in self._db.execute("SELECT path, sha FROM files")}

    def replace_file(self, path: str, sha: str, language: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        with self._lock, self._db:
            self._delete_chunks(path)
            for chunk, vector in zip(chunks, vectors, strict=True):
                cur = self._db.execute(
                    "INSERT INTO chunks (path, start_line, end_line, kind, symbol, text, vector)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chunk.path, chunk.start_line, chunk.end_line, chunk.kind, chunk.symbol, chunk.text,
                     np.asarray(vector, dtype=np.float32).tobytes()),
                )
                body = f"{chunk.text}\n{identifier_terms(chunk.text)}"
                self._db.execute("INSERT INTO chunks_fts (rowid, body, symbol, path) VALUES (?, ?, ?, ?)",
                                 (cur.lastrowid, body, chunk.symbol or "", f"{path} {identifier_terms(path)}"))
            self._db.execute("INSERT OR REPLACE INTO files (path, sha, language) VALUES (?, ?, ?)",
                             (path, sha, language))
            self._matrix = None

    def remove_file(self, path: str) -> None:
        with self._lock, self._db:
            self._delete_chunks(path)
            self._db.execute("DELETE FROM files WHERE path = ?", (path,))
            self._matrix = None

    def clear(self) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks_fts")
            self._db.execute("DELETE FROM chunks")
            self._db.execute("DELETE FROM files")
            self._db.execute("DELETE FROM commits_fts")
            self._db.execute("DELETE FROM commits")
            self._matrix = None

    def _delete_chunks(self, path: str) -> None:
        self._db.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE path = ?)", (path,))
        self._db.execute("DELETE FROM chunks WHERE path = ?", (path,))

    def counts(self) -> dict[str, int]:
        files = self._db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"files": files, "chunks": chunks}

    def languages(self) -> dict[str, int]:
        rows = self._db.execute("SELECT language, COUNT(*) AS n FROM files GROUP BY language ORDER BY n DESC")
        return {row["language"]: row["n"] for row in rows}

    def get_chunks(self, ids: list[int]) -> dict[int, StoredChunk]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._db.execute(
            f"SELECT id, path, start_line, end_line, kind, symbol, text FROM chunks WHERE id IN ({marks})", ids)
        return {row["id"]: StoredChunk(**dict(row)) for row in rows}

    def find_symbol(self, symbol: str, path: str | None = None) -> list[StoredChunk]:
        """Chunks whose symbol is ``symbol`` or ends with ``.symbol`` (so ``unsign`` finds ``Signer.unsign``)."""
        query = ("SELECT id, path, start_line, end_line, kind, symbol, text FROM chunks"
                 " WHERE (symbol = ? OR symbol LIKE ?)" + (" AND path = ?" if path else "") +
                 " ORDER BY start_line")
        params = [symbol, f"%.{symbol}"] + ([path] if path else [])
        return [StoredChunk(**dict(row)) for row in self._db.execute(query, params)]

    # ---- history --------------------------------------------------------------------------
    def add_commits(self, commits: list[Commit]) -> int:
        added = 0
        with self._lock, self._db:
            for commit in commits:
                cur = self._db.execute(
                    "INSERT OR IGNORE INTO commits (sha, author, date, subject, body, prs) VALUES (?, ?, ?, ?, ?, ?)",
                    (commit.sha, commit.author, commit.date, commit.subject, commit.body, json.dumps(commit.prs)))
                if cur.rowcount:
                    added += 1
                    self._db.execute("INSERT INTO commits_fts (subject, body, sha) VALUES (?, ?, ?)",
                                     (f"{commit.subject} {identifier_terms(commit.subject)}", commit.body, commit.sha))
        return added

    def commit_count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM commits").fetchone()[0]

    def search_commits(self, query: str, limit: int) -> list[Commit]:
        match = _match_expression(query)
        if not match:
            return []
        rows = self._db.execute(
            "SELECT c.sha, c.author, c.date, c.subject, c.body, c.prs FROM commits_fts f"
            " JOIN commits c ON c.sha = f.sha WHERE commits_fts MATCH ?"
            " ORDER BY bm25(commits_fts, 3.0, 1.0) LIMIT ?", (match, limit))
        return [Commit(sha=r["sha"], author=r["author"], date=r["date"], subject=r["subject"], body=r["body"],
                       prs=tuple(json.loads(r["prs"]))) for r in rows]

    def cached_pull_request(self, number: int) -> PullRequest | None:
        row = self._db.execute("SELECT data FROM pull_requests WHERE number = ?", (number,)).fetchone()
        return PullRequest(**json.loads(row["data"])) if row else None

    def cache_pull_request(self, pr: PullRequest) -> None:
        if pr.title is None:
            return  # don't cache failed lookups; they may work next time
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO pull_requests (number, data) VALUES (?, ?)",
                             (pr.number, json.dumps(pr.__dict__)))

    # ---- retrieval ------------------------------------------------------------------------
    def keyword_search(self, query: str, limit: int) -> list[int]:
        """Chunk ids ranked by BM25. Symbol and path matches count more than body matches."""
        match = _match_expression(query)
        if not match:
            return []
        rows = self._db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, 1.0, 4.0, 2.0) LIMIT ?",
            (match, limit))
        return [row[0] for row in rows]

    def vector_search(self, query_vector: np.ndarray, limit: int) -> list[int]:
        """Chunk ids ranked by cosine similarity (vectors are normalised, so a dot product)."""
        ids, matrix = self._vectors()
        if not len(ids):
            return []
        scores = matrix @ query_vector.astype(np.float32)
        top = np.argsort(-scores)[:limit]
        return [int(ids[i]) for i in top]

    def _vectors(self) -> tuple[np.ndarray, np.ndarray]:
        if self._matrix is None:
            rows = self._db.execute("SELECT id, vector FROM chunks ORDER BY id").fetchall()
            if rows:
                ids = np.array([row["id"] for row in rows])
                matrix = np.stack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows])
            else:
                ids, matrix = np.array([], dtype=int), np.zeros((0, 0), dtype=np.float32)
            self._matrix = (ids, matrix)
        return self._matrix
