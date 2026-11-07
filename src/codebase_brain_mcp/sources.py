"""Find a repository on disk (cloning it first if it's a URL) and list the files worth indexing."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".md": "markdown",
    ".mdx": "markdown",
    ".rst": "text",
    ".txt": "text",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".sh": "shell",
    ".bash": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".toml": "config",
    ".yaml": "config",
    ".yml": "config",
    ".json": "config",
    ".ini": "config",
    ".cfg": "config",
}
NAMED_FILES = {"Dockerfile": "config", "Makefile": "config", "README": "text", "LICENSE": "text"}
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".next",
    "target",
    "vendor",
    ".idea",
    ".vscode",
}
SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "Cargo.lock"}
GIT_TIMEOUT_SECONDS = 300
_URL = re.compile(r"^https://[A-Za-z0-9.-]+/[A-Za-z0-9._/-]+?(\.git)?/?$")


class SourceError(ValueError):
    """The source can't be used: missing path, bad URL, or a failed clone."""


@dataclass(frozen=True)
class SourceFile:
    path: str
    """Path relative to the repository root, always with forward slashes."""
    language: str
    text: str


def is_url(source: str) -> bool:
    return source.startswith(("https://", "http://", "git@", "ssh://"))


def default_name(source: str) -> str:
    """``https://github.com/tiangolo/fastapi.git`` and ``~/code/fastapi`` both become ``fastapi``."""
    tail = source.rstrip("/").rsplit("/", 1)[-1]
    tail = tail.removesuffix(".git")
    name = re.sub(r"[^A-Za-z0-9._-]", "-", tail).strip("-.")
    if not name:
        raise SourceError(f"Can't work out a name from {source!r}; pass one explicitly.")
    return name


def resolve(source: str, clone_dir: Path) -> Path:
    """The repository's root on disk. URLs are cloned blobless (or refreshed) under ``clone_dir``."""
    if not is_url(source):
        root = Path(source).expanduser().resolve()
        if not root.is_dir():
            raise SourceError(f"{source!r} is not a directory.")
        return root
    if not _URL.match(source):
        # Only plain https URLs: no ssh, no git's ext:: transports, no options smuggled into the URL.
        raise SourceError("Only https:// repository URLs are supported, e.g. https://github.com/owner/repo.")
    target = clone_dir / default_name(source)
    # Blobless clones keep the full commit history (needed for explain_history) but fetch file
    # contents only for the checkout and for the commits we actually inspect.
    if (target / ".git").is_dir():
        if run_git(target, "rev-parse", "--is-shallow-repository").strip() == "true":
            run_git(target, "fetch", "--unshallow", "--filter=blob:none", "origin")
        else:
            run_git(target, "fetch", "--filter=blob:none", "origin")
        run_git(target, "reset", "--hard", "FETCH_HEAD")
    else:
        clone_dir.mkdir(parents=True, exist_ok=True)
        run_git(clone_dir, "clone", "--filter=blob:none", "--", source, str(target))
    return target


def head_commit(root: Path) -> str | None:
    try:
        return run_git(root, "rev-parse", "HEAD").strip() or None
    except SourceError:
        return None


def iter_files(root: Path, max_bytes: int) -> list[SourceFile]:
    """Text files worth indexing, sorted by path. Honours .gitignore when ``root`` is a git checkout."""
    files = []
    for rel in _candidate_paths(root):
        language = language_for(rel)
        if language is None:
            continue
        full = root / rel
        try:
            if not full.is_file() or full.is_symlink() or full.stat().st_size > max_bytes:
                continue
            raw = full.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue  # binary
        files.append(SourceFile(path=rel, language=language, text=raw.decode("utf-8", errors="replace")))
    return sorted(files, key=lambda f: f.path)


def language_for(rel: str) -> str | None:
    name = rel.rsplit("/", 1)[-1]
    if name in SKIP_FILES or name.endswith((".min.js", ".min.css", ".map")):
        return None
    if name in NAMED_FILES:
        return NAMED_FILES[name]
    return LANGUAGES.get(os.path.splitext(name)[1].lower())


def _candidate_paths(root: Path) -> list[str]:
    try:
        out = run_git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
        paths = [p for p in out.split("\0") if p]
        return [p for p in paths if not SKIP_DIRS.intersection(p.split("/")[:-1])]
    except SourceError:
        pass
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        for filename in filenames:
            paths.append(filename if rel_dir == "." else f"{rel_dir}/{filename}")
    return paths


def run_git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ALLOW_PROTOCOL": "https:file"}
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceError(f"git {args[0]} failed: {exc}") from exc
    if done.returncode != 0:
        raise SourceError(f"git {args[0]} failed: {done.stderr.strip()[:300]}")
    return done.stdout
