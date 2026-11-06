"""Structural analyses over an index: what depends on a symbol, and where the docs disagree with the code."""

from __future__ import annotations

import ast
import builtins
import difflib
import keyword
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from . import history
from .sources import SourceError
from .store import Store, StoredChunk

MAX_PER_LEVEL = 60
STALE_AFTER = timedelta(days=180)
MAX_STALENESS_CHECKS = 40
_FENCE = re.compile(r"^(`{3,}|~{3,})\s*([\w+-]*)[^\n]*\n(.*?)^\1\s*$", re.MULTILINE | re.DOTALL)
_INLINE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_DOUBLE = re.compile(r"``([^`\n]+)``")
_RST_ROLE = re.compile(r":(?:py:)?(?:func|meth|class|attr|mod|data|exc|obj|const):`([^`]+)`")
_RST_BLOCK = re.compile(
    r"^\.\. (?:code-block|code|sourcecode)::[ \t]*(\w*)[^\n]*\n((?:[ \t]*\n|[ \t]+[^\n]*\n?)+)", re.MULTILINE
)
_DOCTEST = re.compile(r"^([ \t]*>>> [^\n]*\n(?:[ \t]*(?:>>>|\.\.\.)[^\n]*\n|[ \t]+[^\n]+\n)*)", re.MULTILINE)
_PATH = re.compile(r"^\.?/?[\w.-]+(/[\w.-]+)+/?$")
_ENV = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)+$")
_FLAG = re.compile(r"^(--[a-z][a-z0-9-]+)(=\S*)?$")
_SYMBOL = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*(\(\))?$")
_TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]+$|_test\.\w+$")
_COMMON = set(dir(builtins)) | set(keyword.kwlist) | {"true", "false", "null", "none", "self", "cls"}
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_CHANGELOG = re.compile(
    r"(^|/)(CHANGES|CHANGELOG|HISTORY|NEWS|RELEASES?|RELEASE[-_]NOTES)[^/]*$|(^|/)changelog/", re.IGNORECASE
)
"""Changelogs are meant to mention removed and renamed things, so they're never drift."""


def definition_span(matches: list[StoredChunk]) -> tuple[str, int, int, str | None]:
    """The real definition among same-named chunks: windows of one long function are merged back
    together, and the biggest span wins, so ``@overload`` stubs lose to the implementation."""
    first = matches[0]
    same = sorted((c for c in matches if c.path == first.path and c.symbol == first.symbol), key=lambda c: c.start_line)
    spans: list[list[int]] = []
    for chunk in same:
        if spans and chunk.start_line <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], chunk.end_line)
        else:
            spans.append([chunk.start_line, chunk.end_line])
    start, end = max(spans, key=lambda span: span[1] - span[0])
    return first.path, start, end, first.symbol


# ---- impact_of_change -------------------------------------------------------------------------


@dataclass(frozen=True)
class Dependent:
    chunk: StoredChunk
    via: str
    """The name through which it depends on the level above."""
    level: int
    """1 = uses the target directly; 2 = uses something that uses it."""
    is_test: bool


@dataclass(frozen=True)
class ImpactReport:
    symbol: str
    path: str
    start_line: int
    end_line: int
    dependents: list[Dependent]
    same_name_definitions: list[str]
    notes: list[str] = field(default_factory=list)


def impact_of_change(store: Store, target: StoredChunk, span: tuple[int, int], depth: int = 2) -> ImpactReport:
    """Everything that uses ``target`` by name, directly and (up to ``depth``) transitively."""
    symbol = target.symbol or ""
    short = symbol.rsplit(".", 1)[-1]
    excluded = {c.id for c in store.find_symbol(symbol, target.path)}
    dependents: list[Dependent] = []
    frontier = [short]
    for level in range(1, max(1, min(depth, 3)) + 1):
        next_frontier: list[str] = []
        for name in frontier:
            for chunk in store.chunks_referencing(name, MAX_PER_LEVEL):
                inside_target = chunk.path == target.path and span[0] <= chunk.start_line <= span[1]
                if chunk.id in excluded or inside_target:
                    continue
                excluded.add(chunk.id)
                dependents.append(
                    Dependent(chunk=chunk, via=name, level=level, is_test=bool(_TEST_PATH.search(chunk.path)))
                )
                if chunk.symbol and chunk.kind in ("function", "method", "class"):
                    next_frontier.append(chunk.symbol.rsplit(".", 1)[-1])
        frontier = list(dict.fromkeys(next_frontier))
        if not frontier:
            break

    twins = sorted(
        f"{s} ({p})"
        for s, p in store.symbols()
        if s.rsplit(".", 1)[-1] == short and not (s == symbol and p == target.path)
    )
    notes = []
    if twins:
        notes.append(
            f"{len(twins)} other definition(s) share the name {short!r}; some dependents may use those "
            "instead. Check the import or receiver before changing anything."
        )
    if not target.path.endswith((".py", ".pyi")):
        notes.append("Only Python is analysed structurally; for other languages, also run search for the name.")
    notes.append("Found by name, so dynamic calls (getattr, string dispatch, callbacks) aren't included.")
    return ImpactReport(
        symbol=symbol,
        path=target.path,
        start_line=span[0],
        end_line=span[1],
        dependents=dependents,
        same_name_definitions=twins,
        notes=notes,
    )


# ---- find_doc_drift ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    severity: str
    """high: almost certainly wrong. medium: probably wrong. low: worth a look."""
    kind: str
    doc: str
    """Citation of the doc section, path:start-end."""
    claim: str
    detail: str
    suggestion: str | None = None


@dataclass(frozen=True)
class DriftReport:
    findings: list[Finding]
    sections_checked: int
    references_checked: int


def find_doc_drift(
    store: Store, root: Path, path_prefix: str | None = None, check_history: bool = True, limit: int = 50
) -> DriftReport:
    """Documentation claims that no longer match the code: missing files, symbols, settings and flags,
    examples calling functions with parameters they don't take, and docs older than the code they describe."""
    checker = _DriftChecker(store, root)
    for section in store.doc_chunks(path_prefix):
        if not _CHANGELOG.search(section.path):
            checker.check(section)
    if check_history and history.is_git_repo(root):
        checker.check_staleness()
    findings = sorted(checker.findings.values(), key=lambda f: (_SEVERITY_ORDER[f.severity], f.doc))
    return DriftReport(
        findings=findings[:limit], sections_checked=checker.sections, references_checked=checker.references
    )


class _DriftChecker:
    def __init__(self, store: Store, root: Path):
        self.store = store
        self.root = root
        self.files = store.file_paths()
        self.dirs = {"/".join(p.split("/")[:i]) for p in self.files for i in range(1, p.count("/") + 1)}
        self.symbols = store.symbols()
        self.short_names = {s.rsplit(".", 1)[-1] for s, _ in self.symbols}
        self.full_names = {s for s, _ in self.symbols}
        self.packages = {
            (p.split("/")[1] if p.startswith("src/") else p.split("/")[0]).removesuffix(".py")
            for p in self.files
            if p.endswith(".py") and not p.startswith(("tests/", "docs/"))
        }
        self.findings: dict[tuple[str, str, str], Finding] = {}
        self.sections = 0
        self.references = 0
        self._resolved: list[tuple[StoredChunk, str]] = []
        self._mentions: dict[str, bool] = {}

    def check(self, section: StoredChunk) -> None:
        """Markdown (`code`, ``` fences) and reStructuredText (``code``, :func: roles, code-block, doctest)."""
        self.sections += 1
        prose = section.text
        for pattern, language_group, body_group in ((_FENCE, 2, 3), (_RST_BLOCK, 1, 2), (_DOCTEST, None, 1)):
            for match in pattern.finditer(prose):
                language = match.group(language_group).lower() if language_group else "python"
                if language in ("", "python", "py", "pycon", "python3"):
                    self._check_code_block(section, match.group(body_group))
                prose = prose.replace(match.group(0), "\n")
        for match in _RST_ROLE.finditer(prose):
            target = match.group(1)
            target = target.split("<", 1)[1].rstrip(">") if "<" in target else target
            self._check_symbol(section, target.strip().lstrip("~!.").removesuffix("()"), explicit=True)
        prose = _RST_ROLE.sub(" ", prose)
        for match in _DOUBLE.finditer(prose):
            self._check_inline(section, match.group(1).strip())
        for match in _INLINE.finditer(_DOUBLE.sub(" ", prose)):
            self._check_inline(section, match.group(1).strip())

    # -- inline references ---------------------------------------------------------------
    def _check_inline(self, section: StoredChunk, ref: str) -> None:
        if not ref or "://" in ref or ref.startswith(("$", "<", "@")):
            return
        if flag := _FLAG.match(ref):
            self._expect_in_code(section, flag.group(1), "flag", "The command-line flag")
        elif _ENV.match(ref):
            self._expect_in_code(section, ref, "setting", "The setting")
        elif (_PATH.match(ref) and "." in ref.rsplit("/", 1)[-1]) or ref.endswith("/"):
            self._check_path(section, ref)
        elif _SYMBOL.match(ref):
            self._check_symbol(section, ref)

    def _check_path(self, section: StoredChunk, ref: str) -> None:
        self.references += 1
        path = ref.lstrip("./").rstrip("/")
        if path in self.files or path in self.dirs:
            return
        basename = path.rsplit("/", 1)[-1]
        moved = sorted(p for p in self.files if p.rsplit("/", 1)[-1] == basename)
        close = moved or difflib.get_close_matches(path, list(self.files), n=1, cutoff=0.8)
        self._add(
            Finding(
                severity="high" if close else "medium",
                kind="missing_file",
                doc=section.citation,
                claim=f"`{ref}`",
                detail=f"No file or directory {path!r} exists in the repository.",
                suggestion=f"Did it move to {close[0]!r}?" if close else None,
            )
        )

    def _check_symbol(self, section: StoredChunk, ref: str, explicit: bool = False) -> None:
        name = ref.removesuffix("()")
        short = name.rsplit(".", 1)[-1]
        head = name.split(".", 1)[0]
        if "." in name and head not in self.packages and head not in self.short_names:
            return  # another library's API (os.urandom, json.dumps): not this project's to check
        if not explicit and not _looks_like_code(ref):
            return
        self.references += 1
        if name in self.full_names or short in self.short_names:
            self._resolved.append((section, short))
            return
        if self._mentioned(short):
            return  # used in code (a variable, attribute or third-party name); not ours to judge
        close = difflib.get_close_matches(short, sorted(self.short_names), n=1, cutoff=0.75)
        self._add(
            Finding(
                severity="high" if close else "medium",
                kind="missing_symbol",
                doc=section.citation,
                claim=f"`{ref}`",
                detail=f"{short!r} isn't defined or used anywhere in the code.",
                suggestion=f"Was it renamed to {close[0]!r}?" if close else None,
            )
        )

    def _expect_in_code(self, section: StoredChunk, term: str, kind: str, label: str) -> None:
        self.references += 1
        if not self._mentioned(term):
            self._add(
                Finding(
                    severity="medium",
                    kind=f"unknown_{kind}",
                    doc=section.citation,
                    claim=f"`{term}`",
                    detail=f"{label} {term} doesn't appear anywhere in the code.",
                )
            )

    # -- code examples -------------------------------------------------------------------
    def _check_code_block(self, section: StoredChunk, body: str) -> None:
        body = textwrap.dedent(body)
        lines = [
            re.sub(r"^(>>>|\.\.\.) ?", "", line)
            for line in body.splitlines()
            if not body.lstrip().startswith(">>>") or line.startswith((">>>", "..."))
        ]
        try:
            tree = ast.parse(textwrap.dedent("\n".join(lines)))
        except (SyntaxError, ValueError):
            return  # shell commands, pseudo-code, or another language
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.keywords:
                self._check_call(section, node)
            elif isinstance(node, ast.ImportFrom) and node.module and self._is_project_module(node.module):
                for alias in node.names:
                    if alias.name != "*":
                        self._check_import(section, node.module, alias.name)

    def _check_call(self, section: StoredChunk, call: ast.Call) -> None:
        func = call.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        keywords = [k.arg for k in call.keywords if k.arg]
        if not name or not keywords:
            return
        if name[:1].isupper():  # probably a class: its constructor is what the call reaches
            signatures = [s for s in self.store.signatures("__init__") if s[0].endswith(f"{name}.__init__")]
            if not signatures and name in self.short_names:
                return  # a class without its own __init__ (inherited or a dataclass): can't check
        else:
            signatures = self.store.signatures(name)
        if not signatures:
            return
        self.references += 1
        if any(any_kw or set(keywords) <= set(params) for _, _, params, any_kw in signatures):
            return
        symbol, path, params, _ = signatures[0]
        unknown = [k for k in keywords if k not in params]
        suggestion = "; ".join(
            f"{k!r} → {m[0]!r}?" for k in unknown if (m := difflib.get_close_matches(k, params, n=1, cutoff=0.6))
        )
        self._add(
            Finding(
                severity="high",
                kind="wrong_parameters",
                doc=section.citation,
                claim=f"{name}({', '.join(f'{k}=…' for k in keywords)})",
                detail=f"{symbol} in {path} takes ({', '.join(params)}); the example passes "
                f"{', '.join(unknown)}, which it doesn't accept.",
                suggestion=suggestion or None,
            )
        )

    def _check_import(self, section: StoredChunk, module: str, name: str) -> None:
        self.references += 1
        if name in self.short_names or self._mentioned(f"{name} =") or self._mentioned(f"import {name}"):
            return
        close = difflib.get_close_matches(name, sorted(self.short_names), n=1, cutoff=0.75)
        self._add(
            Finding(
                severity="high",
                kind="broken_import",
                doc=section.citation,
                claim=f"from {module} import {name}",
                detail=f"{name!r} isn't defined in the project, so this example fails with ImportError.",
                suggestion=f"Was it renamed to {close[0]!r}?" if close else None,
            )
        )

    def _is_project_module(self, module: str) -> bool:
        top = module.split(".")[0]
        return top in self.dirs or f"src/{top}" in self.dirs or f"{top}.py" in self.files

    # -- staleness -----------------------------------------------------------------------
    def check_staleness(self) -> None:
        """Doc sections describing functions that changed long after the docs were last edited.
        One finding per section, listing what changed."""
        doc_dates: dict[str, datetime | None] = {}
        stale: dict[str, tuple[StoredChunk, list[str]]] = {}
        seen: set[tuple[str, str]] = set()
        checked = 0
        for section, short in self._resolved:
            if checked >= MAX_STALENESS_CHECKS or (section.citation, short) in seen:
                continue
            seen.add((section.citation, short))
            matches = self.store.find_symbol(short)
            if not matches:
                continue
            path, start, end, symbol = definition_span(matches)
            checked += 1
            if section.path not in doc_dates:
                doc_dates[section.path] = self._last_change(section.path)
            doc_date = doc_dates[section.path]
            try:
                changes = history.line_history(self.root, path, start, end, 1)
            except SourceError:
                continue
            if not doc_date or not changes:
                continue
            commit = changes[0].commit
            code_date = datetime.fromisoformat(commit.date)
            if code_date - doc_date > STALE_AFTER:
                entry = stale.setdefault(section.citation, (section, []))
                entry[1].append(
                    f"{symbol} ({path}:{start}-{end}) changed {code_date.date()} in "
                    f"{commit.sha[:10]} ({commit.subject!r})"
                )
        for citation, (section, changed) in stale.items():
            doc_date = doc_dates[section.path]
            if doc_date is None:
                continue
            self._add(
                Finding(
                    severity="low",
                    kind="docs_older_than_code",
                    doc=citation,
                    claim=", ".join(f"`{c.split(' ', 1)[0]}`" for c in changed),
                    detail=f"{section.path} was last edited on {doc_date.date()}, but since then: "
                    + "; ".join(changed),
                    suggestion="Check this section still describes the current behaviour.",
                )
            )

    def _last_change(self, path: str) -> datetime | None:
        try:
            changes = history.file_history(self.root, path, 1)
        except SourceError:
            return None
        return datetime.fromisoformat(changes[0].commit.date) if changes else None

    # -- helpers -------------------------------------------------------------------------
    def _mentioned(self, term: str) -> bool:
        if term not in self._mentions:
            self._mentions[term] = self.store.code_mentions(term)
        return self._mentions[term]

    def _add(self, finding: Finding) -> None:
        self.findings.setdefault((finding.doc, finding.kind, finding.claim), finding)


def _looks_like_code(ref: str) -> bool:
    """Skip plain words in backticks; keep things shaped like identifiers."""
    name = ref.removesuffix("()")
    if name.lower() in _COMMON or len(name) < 3:
        return False
    return ref.endswith("()") or "_" in name or "." in name or bool(re.search(r"[a-z][A-Z]", name))
