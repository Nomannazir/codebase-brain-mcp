"""Read a repository's history with git: commits, who wrote which lines, how a line range evolved, and PRs."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .sources import SourceError, run_git

FIELD = "\x1f"
RECORD = "\x1e"
COMMIT_MARKER = "\x1dcommit\x1d"
BODY_CHARS = 1_200
DIFF_CHARS = 1_500
GITHUB_TIMEOUT_SECONDS = 8
_PR_PATTERNS = (re.compile(r"\(#(\d{1,7})\)"), re.compile(r"Merge pull request #(\d{1,7})"))
_GITHUB_REMOTE = re.compile(r"github\.com[:/]([A-Za-z0-9-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$")


@dataclass(frozen=True)
class Commit:
    sha: str
    author: str
    date: str
    """ISO 8601, author date."""
    subject: str
    body: str
    prs: tuple[int, ...] = ()


@dataclass(frozen=True)
class LineChange:
    commit: Commit
    diff: str
    """The part of the commit's diff that touched the range, trimmed."""


@dataclass(frozen=True)
class BlameShare:
    sha: str
    author: str
    date: str
    subject: str
    lines: int


@dataclass(frozen=True)
class PullRequest:
    number: int
    url: str
    title: str | None = None
    body: str | None = None
    state: str | None = None
    merged_at: str | None = None


def is_git_repo(root: Path) -> bool:
    try:
        return run_git(root, "rev-parse", "--is-inside-work-tree").strip() == "true"
    except SourceError:
        return False


def commits(root: Path, since: str | None, limit: int) -> list[Commit]:
    """Commits reachable from HEAD, newest first; only those after ``since`` when it's an ancestor of HEAD."""
    revision = "HEAD"
    if since and _is_ancestor(root, since):
        revision = f"{since}..HEAD"
    out = run_git(root, "log", revision, f"--max-count={limit}",
               f"--format=%H{FIELD}%an{FIELD}%aI{FIELD}%s{FIELD}%b{RECORD}")
    return [_parse_commit(record) for record in out.split(RECORD) if record.strip()]


def line_history(root: Path, path: str, start: int, end: int, limit: int) -> list[LineChange]:
    """Commits that changed lines start..end of ``path`` (following the lines as they moved), newest first."""
    out = run_git(root, "log", f"-L{start},{end}:{path}", f"--max-count={limit}",
               f"--format={COMMIT_MARKER}%H{FIELD}%an{FIELD}%aI{FIELD}%s{FIELD}%b{RECORD}")
    changes = []
    for block in out.split(COMMIT_MARKER):
        if not block.strip() or RECORD not in block:
            continue
        header, diff = block.split(RECORD, 1)
        changes.append(LineChange(commit=_parse_commit(header), diff=_trim(diff.strip(), DIFF_CHARS)))
    return changes


def file_history(root: Path, path: str, limit: int) -> list[LineChange]:
    """Commits that touched ``path`` at all (following renames), newest first, without diffs."""
    out = run_git(root, "log", "--follow", f"--max-count={limit}",
               f"--format=%H{FIELD}%an{FIELD}%aI{FIELD}%s{FIELD}%b{RECORD}", "--", path)
    return [LineChange(commit=_parse_commit(r), diff="") for r in out.split(RECORD) if r.strip()]


def blame(root: Path, path: str, start: int, end: int) -> list[BlameShare]:
    """Which commits the current lines come from, biggest share first."""
    out = run_git(root, "blame", "--line-porcelain", f"-L{start},{end}", "--", path)
    counts: Counter[str] = Counter()
    info: dict[str, dict[str, str]] = {}
    sha = None
    for line in out.splitlines():
        if re.match(r"^[0-9a-f]{40} ", line):
            sha = line.split(" ", 1)[0]
            counts[sha] += 1
            info.setdefault(sha, {})
        elif sha and line.startswith(("author ", "author-time ", "summary ")):
            key, value = line.split(" ", 1)
            info[sha].setdefault(key, value)
    shares = []
    for sha, lines in counts.most_common():
        meta = info[sha]
        shares.append(BlameShare(sha=sha, author=meta.get("author", ""), date=_epoch_to_iso(meta.get("author-time")),
                                 subject=meta.get("summary", ""), lines=lines))
    return shares


def github_slug(root: Path) -> str | None:
    """``owner/repo`` when the origin remote is on GitHub."""
    try:
        url = run_git(root, "remote", "get-url", "origin").strip()
    except SourceError:
        return None
    match = _GITHUB_REMOTE.search(url)
    return f"{match.group(1)}/{match.group(2)}" if match else None


def pr_numbers(message: str) -> tuple[int, ...]:
    found: list[int] = []
    for pattern in _PR_PATTERNS:
        for match in pattern.finditer(message):
            number = int(match.group(1))
            if number not in found:
                found.append(number)
    return tuple(found)


def fetch_pull_request(slug: str, number: int, token: str | None) -> PullRequest:
    """The PR's title and description from the GitHub API. On any failure, just the link."""
    url = f"https://github.com/{slug}/pull/{number}"
    request = urllib.request.Request(f"https://api.github.com/repos/{slug}/pulls/{number}",
                                     headers={"Accept": "application/vnd.github+json",
                                              "User-Agent": "project-brain-mcp"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=GITHUB_TIMEOUT_SECONDS) as response:  # noqa: S310 (fixed host)
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return PullRequest(number=number, url=url)
    return PullRequest(number=number, url=url, title=data.get("title"), body=_trim(data.get("body") or "", BODY_CHARS),
                       state=data.get("state"), merged_at=data.get("merged_at"))


def _parse_commit(record: str) -> Commit:
    sha, author, date, subject, body = (record.strip("\n").split(FIELD) + [""] * 5)[:5]
    body = body.strip()
    return Commit(sha=sha.strip(), author=author, date=date, subject=subject, body=_trim(body, BODY_CHARS),
                  prs=pr_numbers(f"{subject}\n{body}"))


def _is_ancestor(root: Path, sha: str) -> bool:
    try:
        run_git(root, "merge-base", "--is-ancestor", sha, "HEAD")
        return True
    except SourceError:
        return False


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "\n…"


def _epoch_to_iso(value: str | None) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(int(value), UTC).isoformat(timespec="seconds")
