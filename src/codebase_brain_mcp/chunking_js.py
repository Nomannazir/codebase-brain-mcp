"""Find the definitions in JavaScript and TypeScript files with tree-sitter.

Returns plain line spans plus the facts ``chunking`` attaches to chunks: which names each line uses
(calls, properties, types, imports) and each function's parameters. Comment blocks directly above a
definition belong to it, so a JSDoc comment is searched together with its function.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

Grammar = Literal["javascript", "typescript", "tsx"]

FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
CLASS_VALUES = {"class", "class_expression"}
CLASS_DECLARATIONS = {"class_declaration", "abstract_class_declaration"}
TYPE_DECLARATIONS = {"interface_declaration": "interface", "type_alias_declaration": "type", "enum_declaration": "enum"}
FUNCTION_DECLARATIONS = {"function_declaration", "generator_function_declaration", "function_signature"}
METHOD_NODES = {"method_definition", "public_field_definition", "field_definition", "abstract_method_signature"}
NAME_NODES = {
    "identifier", "type_identifier", "property_identifier", "private_property_identifier",
    "shorthand_property_identifier",
}  # fmt: skip
DECLARING_PARENTS = {
    *FUNCTION_DECLARATIONS, *CLASS_DECLARATIONS, *TYPE_DECLARATIONS, *METHOD_NODES,
    "variable_declarator", "formal_parameters", "required_parameter", "optional_parameter", "property_signature",
}  # fmt: skip
"""A name directly under one of these (as its name or parameter) is being declared, not used."""
IGNORED_NAMES = {
    "Array", "Boolean", "Date", "Error", "JSON", "Map", "Math", "Number", "Object", "Promise", "RegExp", "Set",
    "String", "Symbol", "console", "exports", "length", "module", "require", "undefined", "window", "document",
}  # fmt: skip


@dataclass(frozen=True)
class Definition:
    start_line: int
    """1-based, inclusive."""
    end_line: int
    """1-based, inclusive."""
    kind: str
    symbol: str


@dataclass(frozen=True)
class Outline:
    definitions: list[Definition]
    covered: list[tuple[int, int]]
    """Line spans owned by top-level definitions; the rest of the file becomes module chunks."""
    uses: list[tuple[int, str]]
    signatures: dict[int, tuple[tuple[str, ...], bool]]
    """Definition start line -> (parameter names, takes a rest parameter)."""


def grammar_for(path: str, language: str) -> Grammar:
    if language == "typescript":
        return "tsx" if path.endswith(".tsx") else "typescript"
    return "javascript"  # this grammar also understands JSX


@cache
def _parser(grammar: Grammar) -> Parser:
    if grammar == "javascript":
        handle = tree_sitter_javascript.language()
    elif grammar == "tsx":
        handle = tree_sitter_typescript.language_tsx()
    else:
        handle = tree_sitter_typescript.language_typescript()
    return Parser(Language(handle))


def outline(path: str, language: str, text: str, max_class_lines: int) -> Outline | None:
    """The file's definitions, or None when it doesn't parse (the caller then falls back to windows)."""
    tree = _parser(grammar_for(path, language)).parse(text.encode())
    root = tree.root_node
    if root.has_error:
        return None
    collector = _Collector(max_class_lines)
    for statement in root.named_children:
        collector.top_level(statement)
    return Outline(
        definitions=collector.definitions,
        covered=collector.covered,
        uses=_uses(root),
        signatures=collector.signatures,
    )


class _Collector:
    def __init__(self, max_class_lines: int) -> None:
        self.max_class_lines = max_class_lines
        self.definitions: list[Definition] = []
        self.covered: list[tuple[int, int]] = []
        self.signatures: dict[int, tuple[tuple[str, ...], bool]] = {}

    def top_level(self, statement: Node) -> None:
        found = _declared(statement)
        if found is None:
            return
        kind, name, node = found
        start, end = _with_leading_comments(statement), statement.end_point.row + 1
        self.covered.append((start, end))
        if kind == "class":
            self._class(node, name, start, end)
            return
        self.definitions.append(Definition(start, end, kind, name))
        if kind == "function":
            self.signatures[start] = _parameters(node)

    def _class(self, node: Node, name: str, start: int, end: int) -> None:
        body = node.child_by_field_name("body")
        methods = [m for m in body.named_children if m.type in METHOD_NODES and _is_function(m)] if body else []
        if end - start + 1 <= self.max_class_lines or not methods:
            self.definitions.append(Definition(start, end, "class", name))
            return
        # A big class: its header (up to the first method) plus one chunk per method, like Python.
        first = _with_leading_comments(methods[0])
        self.definitions.append(Definition(start, max(start, first - 1), "class", name))
        for method in methods:
            m_start = _with_leading_comments(method)
            symbol = f"{name}.{_text(method.child_by_field_name('name')) or 'anonymous'}"
            self.definitions.append(Definition(m_start, method.end_point.row + 1, "method", symbol))
            self.signatures[m_start] = _parameters(_function_of(method))


def _declared(statement: Node) -> tuple[str, str, Node] | None:
    """(kind, name, node) when a top-level statement defines one named thing."""
    if statement.type == "export_statement":
        inner = statement.child_by_field_name("declaration") or statement.child_by_field_name("value")
        if inner is None:
            return None
        if inner.type in FUNCTION_VALUES:
            return "function", "default", inner  # export default function () {}
        if inner.type in CLASS_VALUES:
            return "class", "default", inner
        statement = inner
    kind = statement.type
    name = _text(statement.child_by_field_name("name"))
    if kind in FUNCTION_DECLARATIONS and name:
        return "function", name, statement
    if kind in CLASS_DECLARATIONS and name:
        return "class", name, statement
    if kind in TYPE_DECLARATIONS and name:
        return TYPE_DECLARATIONS[kind], name, statement
    if kind in ("lexical_declaration", "variable_declaration"):
        return _bound_value(statement)
    if kind == "expression_statement":
        return _assigned_value(statement)
    return None


def _bound_value(statement: Node) -> tuple[str, str, Node] | None:
    """``const name = () => ...`` or ``const Name = class {...}``, when it is the only binding."""
    declarators = [d for d in statement.named_children if d.type == "variable_declarator"]
    if len(declarators) != 1:
        return None
    return _named_value(declarators[0].child_by_field_name("name"), declarators[0].child_by_field_name("value"))


def _assigned_value(statement: Node) -> tuple[str, str, Node] | None:
    """CommonJS and prototype style: ``module.exports.name = function () {...}``."""
    expression = statement.named_children[0] if statement.named_children else None
    if expression is None or expression.type != "assignment_expression":
        return None
    left = expression.child_by_field_name("left")
    if left is not None and left.type == "member_expression":
        left = left.child_by_field_name("property")
    return _named_value(left, expression.child_by_field_name("right"))


def _named_value(name_node: Node | None, value: Node | None) -> tuple[str, str, Node] | None:
    name = _text(name_node)
    if not name or value is None or name_node is None or name_node.type not in NAME_NODES:
        return None
    if value.type in FUNCTION_VALUES:
        return "function", name, value
    if value.type in CLASS_VALUES:
        return "class", name, value
    return None


def _is_function(member: Node) -> bool:
    return member.type in ("method_definition", "abstract_method_signature") or _function_of(member) is not member


def _function_of(member: Node) -> Node:
    """The function node holding a class member's parameters (a field may hold an arrow function)."""
    value = member.child_by_field_name("value")
    return value if value is not None and value.type in FUNCTION_VALUES else member


def _parameters(function: Node) -> tuple[tuple[str, ...], bool]:
    single = function.child_by_field_name("parameter")  # x => x
    if single is not None:
        return (_text(single),), False
    params = function.child_by_field_name("parameters")
    names: list[str] = []
    takes_rest = False
    for param in params.named_children if params else []:
        pattern = param.child_by_field_name("pattern") or param
        if pattern.type == "assignment_pattern":
            pattern = pattern.child_by_field_name("left") or pattern
        if pattern.type == "rest_pattern":
            takes_rest = True
            pattern = pattern.named_children[0] if pattern.named_children else pattern
        if pattern.type == "identifier":
            names.append(_text(pattern))
    return tuple(names), takes_rest


def _with_leading_comments(node: Node) -> int:
    """The 1-based start line, moved up over comments that sit directly above the node."""
    start = node.start_point.row
    previous = node.prev_named_sibling
    while previous is not None and previous.type == "comment" and previous.end_point.row >= start - 1:
        start = previous.start_point.row
        previous = previous.prev_named_sibling
    return start + 1


def _uses(root: Node) -> list[tuple[int, str]]:
    """Every name the file mentions, by line: identifiers, properties, types and imported names.

    Walks with a cursor, so a big file doesn't create a Python object for every node at once.
    """
    uses: list[tuple[int, str]] = []
    cursor = root.walk()
    while True:
        node = cursor.node
        if node is not None and node.type in NAME_NODES and not _is_declared(node, cursor.field_name):
            name = _text(node)
            if name not in IGNORED_NAMES:
                uses.append((node.start_point.row + 1, name))
        if cursor.goto_first_child():
            continue
        while not cursor.goto_next_sibling():
            if not cursor.goto_parent():
                return uses


def _is_declared(node: Node, field: str | None) -> bool:
    parent = node.parent
    if parent is None or parent.type not in DECLARING_PARENTS:
        return False
    return field in ("name", "pattern") or parent.type == "formal_parameters"


def _text(node: Node | None) -> str:
    return node.text.decode(errors="replace") if node is not None and node.text is not None else ""
