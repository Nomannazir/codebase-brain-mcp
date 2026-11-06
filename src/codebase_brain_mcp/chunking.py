"""Split files into chunks that follow the code's own structure.

Python is split by function and class (with exact line numbers, from ``ast``), JavaScript and
TypeScript the same way (with tree-sitter), Markdown by heading, and everything else into overlapping
line windows. Chunks that are still too long are windowed
again but keep their symbol, so a citation always says which function it came from.
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass, replace

from . import chunking_js

WINDOW_LINES = 60
WINDOW_OVERLAP = 10
MAX_CHUNK_LINES = 120
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_IGNORED_NAMES = set(dir(builtins)) | {"self", "cls", "super", "typing", "t", "_"}
DEFINITION_KINDS = ("function", "method", "class", "interface", "type", "enum")
"""Chunk kinds that define a named thing other code can use."""
Signatures = dict[int, tuple[tuple[str, ...], bool]]


@dataclass(frozen=True)
class Chunk:
    path: str
    start_line: int
    """1-based, inclusive."""
    end_line: int
    """1-based, inclusive."""
    kind: str
    """function, class, method, interface, type, enum, module, section, or code."""
    symbol: str | None
    text: str
    refs: tuple[str, ...] = ()
    """Names this chunk uses: calls, attributes and imports (Python, JS and TS). Feeds impact_of_change."""
    params: tuple[str, ...] | None = None
    """Parameter names when the chunk is a function or method; None otherwise. Feeds find_doc_drift."""
    accepts_any_keyword: bool = False
    """True when the function takes ``**kwargs``, so any keyword argument is valid."""

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


def chunk_file(path: str, language: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return []
    if language == "python":
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return _windows(path, lines, 1, len(lines), "code", None)
        chunks = [piece for chunk in _python_chunks(path, lines, tree) for piece in _split_long(chunk, lines)]
        return _with_facts(chunks, *_python_facts(tree))
    if language in ("javascript", "typescript"):
        return _js_chunks(path, language, lines, text)
    if language == "markdown":
        chunks = _markdown_chunks(path, lines)
    else:
        chunks = _windows(path, lines, 1, len(lines), "code", None)
    return [piece for chunk in chunks for piece in _split_long(chunk, lines)]


def _python_chunks(path: str, lines: list[str], tree: ast.Module) -> list[Chunk]:
    chunks: list[Chunk] = []
    covered: list[tuple[int, int]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start, end = _span(node)
        covered.append((start, end))
        if isinstance(node, ast.ClassDef):
            chunks.extend(_class_chunks(path, lines, node, start, end))
        else:
            chunks.append(_make(path, lines, start, end, "function", node.name))
    return _with_module_gaps(path, lines, chunks, covered)


def _js_chunks(path: str, language: str, lines: list[str], text: str) -> list[Chunk]:
    found = chunking_js.outline(path, language, text, MAX_CHUNK_LINES)
    if found is None:
        return _windows(path, lines, 1, len(lines), "code", None)
    chunks = [_make(path, lines, d.start_line, d.end_line, d.kind, d.symbol) for d in found.definitions]
    chunks = _with_module_gaps(path, lines, chunks, found.covered)
    pieces = [piece for chunk in chunks for piece in _split_long(chunk, lines)]
    return _with_facts(pieces, found.uses, found.signatures, chunking_js.IGNORED_NAMES)


def _with_module_gaps(path: str, lines: list[str], chunks: list[Chunk], covered: list[tuple[int, int]]) -> list[Chunk]:
    """Whatever sits between definitions (imports, constants, module docstring) is one "module" chunk each."""
    result = list(chunks)
    cursor = 1
    for start, end in [*sorted(covered), (len(lines) + 1, len(lines) + 1)]:
        gap_start, gap_end = cursor, start - 1
        while gap_start <= gap_end and not lines[gap_start - 1].strip():
            gap_start += 1
        while gap_end >= gap_start and not lines[gap_end - 1].strip():
            gap_end -= 1
        if gap_start <= gap_end:
            result.append(_make(path, lines, gap_start, gap_end, "module", None))
        cursor = max(cursor, end + 1)
    return sorted(result, key=lambda c: c.start_line)


def _python_facts(tree: ast.Module) -> tuple[list[tuple[int, str]], Signatures, set[str]]:
    """The names used on each line, and each function's parameters keyed by its first line."""
    uses: list[tuple[int, str]] = []
    signatures: Signatures = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            uses.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            uses.append((node.lineno, node.attr))
        elif isinstance(node, ast.ImportFrom):
            uses.extend((node.lineno, alias.name) for alias in node.names)
            if node.module:
                uses.extend((node.lineno, part) for part in node.module.split("."))
        elif isinstance(node, ast.Import):
            uses.extend((node.lineno, part) for alias in node.names for part in alias.name.split("."))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
            signatures[_span(node)[0]] = (tuple(n for n in names if n not in ("self", "cls")), args.kwarg is not None)
    return uses, signatures, _IGNORED_NAMES


def _with_facts(
    chunks: list[Chunk], uses: list[tuple[int, str]], signatures: Signatures, ignored: set[str]
) -> list[Chunk]:
    """Attach the names each chunk uses, and the parameters of each function chunk."""
    result = []
    for chunk in chunks:
        own = chunk.symbol.rsplit(".", 1)[-1] if chunk.symbol else None
        refs = sorted(
            {
                name
                for line, name in uses
                if chunk.start_line <= line <= chunk.end_line and name not in ignored and name != own and len(name) > 1
            }
        )
        signature = signatures.get(chunk.start_line) if chunk.kind in ("function", "method") else None
        result.append(
            replace(
                chunk,
                refs=tuple(refs),
                params=signature[0] if signature else None,
                accepts_any_keyword=bool(signature and signature[1]),
            )
        )
    return result


def _class_chunks(path: str, lines: list[str], node: ast.ClassDef, start: int, end: int) -> list[Chunk]:
    if end - start + 1 <= MAX_CHUNK_LINES:
        return [_make(path, lines, start, end, "class", node.name)]
    # A big class: its header (up to the first method) plus one chunk per method.
    methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    header_end = (_span(methods[0])[0] - 1) if methods else end
    chunks = [_make(path, lines, start, header_end, "class", node.name)]
    for method in methods:
        m_start, m_end = _span(method)
        chunks.append(_make(path, lines, m_start, m_end, "method", f"{node.name}.{method.name}"))
    return chunks


def _span(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> tuple[int, int]:
    decorators = node.decorator_list
    start = min([node.lineno, *(d.lineno for d in decorators)])
    return start, node.end_lineno or node.lineno


def _markdown_chunks(path: str, lines: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    trail: list[tuple[int, str]] = []
    """(level, heading) of each open heading; a new heading closes every one at its level or deeper."""
    section_start, section_symbol = 1, None
    in_fence = False
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if not match:
            continue
        if number > section_start:
            chunks.append(_make(path, lines, section_start, number - 1, "section", section_symbol))
        level = len(match.group(1))
        trail = [*(open_ for open_ in trail if open_[0] < level), (level, match.group(2))]
        section_start, section_symbol = number, " > ".join(heading for _, heading in trail)
    chunks.append(_make(path, lines, section_start, len(lines), "section", section_symbol))
    return [c for c in chunks if c.text.strip()]


def _windows(path: str, lines: list[str], start: int, end: int, kind: str, symbol: str | None) -> list[Chunk]:
    chunks = []
    step = WINDOW_LINES - WINDOW_OVERLAP
    cursor = start
    while cursor <= end:
        stop = min(cursor + WINDOW_LINES - 1, end)
        chunk = _make(path, lines, cursor, stop, kind, symbol)
        if chunk.text.strip():
            chunks.append(chunk)
        if stop == end:
            break
        cursor += step
    return chunks


def _split_long(chunk: Chunk, lines: list[str]) -> list[Chunk]:
    if chunk.end_line - chunk.start_line + 1 <= MAX_CHUNK_LINES:
        return [chunk]
    return _windows(chunk.path, lines, chunk.start_line, chunk.end_line, chunk.kind, chunk.symbol)


def _make(path: str, lines: list[str], start: int, end: int, kind: str, symbol: str | None) -> Chunk:
    return Chunk(
        path=path, start_line=start, end_line=end, kind=kind, symbol=symbol, text="\n".join(lines[start - 1 : end])
    )
