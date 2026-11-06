"""Settings, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MAX_FILE_BYTES = 512_000
DEFAULT_MAX_FILES = 20_000


@dataclass(frozen=True)
class Settings:
    home: Path
    """Where indexes and cloned repositories live."""

    embedder: str
    """``fastembed`` (real local model) or ``hash`` (offline, for tests)."""

    embed_model: str
    max_file_bytes: int
    """Files larger than this are skipped; they are almost always generated or data."""

    github_token: str | None = None
    """Optional. Only used to read PR titles and descriptions (raises the API rate limit)."""

    fetch_pull_requests: bool = True

    allow_local_paths: bool = True
    """Whether index_repo may read local directories. Off by default when serving over HTTP, so remote
    clients can only index public git URLs, never the server's own files."""

    max_files: int = DEFAULT_MAX_FILES
    """Repositories with more indexable files than this are refused, to bound time and disk."""

    @property
    def index_dir(self) -> Path:
        return self.home / "indexes"

    @property
    def clone_dir(self) -> Path:
        return self.home / "clones"


def load_settings() -> Settings:
    home = Path(os.environ.get("PROJECT_BRAIN_HOME", Path.home() / ".project-brain")).expanduser()
    return Settings(
        home=home,
        embedder=os.environ.get("PROJECT_BRAIN_EMBEDDER", "fastembed"),
        embed_model=os.environ.get("PROJECT_BRAIN_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        max_file_bytes=int(os.environ.get("PROJECT_BRAIN_MAX_FILE_BYTES", DEFAULT_MAX_FILE_BYTES)),
        github_token=os.environ.get("GITHUB_TOKEN") or None,
        fetch_pull_requests=os.environ.get("PROJECT_BRAIN_FETCH_PRS", "1") != "0",
        allow_local_paths=os.environ.get("PROJECT_BRAIN_ALLOW_LOCAL_PATHS", "1") != "0",
        max_files=int(os.environ.get("PROJECT_BRAIN_MAX_FILES", DEFAULT_MAX_FILES)),
    )
