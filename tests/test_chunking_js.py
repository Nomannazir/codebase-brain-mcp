import textwrap
from pathlib import Path

import pytest

from codebase_brain_mcp.chunking import MAX_CHUNK_LINES, chunk_file

TS_SOURCE = textwrap.dedent("""
    import { request } from "./http";
    import type { Options } from "./types";

    /**
     * Fetch a user by id.
     */
    export async function getUser(id: string, options?: Options): Promise<User> {
        return request(`/users/${id}`, options);
    }

    export const slugify = (text: string) => text.toLowerCase().trim();

    export interface User {
        id: string;
        name: string;
    }

    export type Role = "admin" | "member";

    enum Color {
        Red,
        Green,
    }

    export default class Client {
        constructor(private base: string) {}

        get(path: string, ...rest: unknown[]) {
            return request(this.base + path);
        }
    }
    """).lstrip()


def _by_symbol(chunks):
    """The first chunk of each symbol (a long function is windowed; its first window has the signature)."""
    found = {}
    for chunk in chunks:
        found.setdefault(chunk.symbol, chunk)
    return found


def test_typescript_is_split_by_declaration_with_exact_lines():
    chunks = _by_symbol(chunk_file("src/api.ts", "typescript", TS_SOURCE))

    assert (chunks["getUser"].start_line, chunks["getUser"].end_line) == (4, 9)  # doc comment included
    assert chunks["getUser"].kind == "function"
    assert chunks["slugify"].kind == "function"  # arrow function bound to a const
    assert chunks["User"].kind == "interface"
    assert chunks["Role"].kind == "type"
    assert chunks["Color"].kind == "enum"
    assert chunks["Client"].kind == "class"
    assert chunks[None].kind == "module" and "import { request }" in chunks[None].text


def test_typescript_chunks_know_their_calls_imports_and_parameters():
    chunks = _by_symbol(chunk_file("src/api.ts", "typescript", TS_SOURCE))

    assert "request" in chunks["getUser"].refs
    assert chunks["getUser"].params == ("id", "options")
    assert chunks["slugify"].params == ("text",)
    assert "toLowerCase" in chunks["slugify"].refs
    assert {"request", "Options"} <= set(chunks[None].refs)  # imports


def test_big_js_class_is_split_into_methods_with_rest_parameters_noted():
    body = "\n".join(f"    const x{i} = {i};" for i in range(MAX_CHUNK_LINES))
    source = f"class Big {{\n  small() {{\n    return 1;\n  }}\n\n  large(a, ...more) {{\n{body}\n  }}\n}}\n"
    chunks = chunk_file("big.js", "javascript", source)
    by_symbol = _by_symbol(chunks)

    assert by_symbol["Big"].kind == "class"
    assert by_symbol["Big.small"].kind == "method"
    assert by_symbol["Big.large"].params == ("a", "more")
    assert by_symbol["Big.large"].accepts_any_keyword
    assert all(c.end_line - c.start_line + 1 <= MAX_CHUNK_LINES for c in chunks)


def test_jsx_and_tsx_parse():
    jsx = "export function Button({ label }) {\n  return <button onClick={go}>{label}</button>;\n}\n"
    tsx = "export const Card = ({ title }: { title: string }) => <div>{title}</div>;\n"

    assert _by_symbol(chunk_file("Button.jsx", "javascript", jsx))["Button"].kind == "function"
    assert _by_symbol(chunk_file("Card.tsx", "typescript", tsx))["Card"].kind == "function"


def test_commonjs_exports_are_named():
    source = "function helper() {}\n\nmodule.exports.format = function (value) {\n  return helper(value);\n};\n"
    chunks = _by_symbol(chunk_file("lib.cjs", "javascript", source))

    assert chunks["format"].kind == "function" and "helper" in chunks["format"].refs


def test_broken_javascript_falls_back_to_windows():
    chunks = chunk_file("broken.js", "javascript", "function oops( {\n  return 1\n")

    assert len(chunks) == 1 and chunks[0].kind == "code"


@pytest.fixture
def web_shop(tmp_path: Path) -> Path:
    files = {
        "src/price.ts": "export function discount(price: number): number {\n  return price * 0.9;\n}\n",
        "src/cart.ts": (
            'import { discount } from "./price";\n\n'
            "export class Cart {\n  total(items: number[]) {\n"
            "    return items.map(discount).reduce((a, b) => a + b, 0);\n  }\n}\n"
        ),
        "src/checkout.js": (
            'const { Cart } = require("./cart");\n\n'
            "function checkout(items) {\n  return new Cart().total(items);\n}\n\nmodule.exports = { checkout };\n"
        ),
        "src/price.test.ts": (
            'import { discount } from "./price";\n\ntest("discount", () => {\n  expect(discount(10)).toBe(9);\n});\n'
        ),
    }
    root = tmp_path / "web-shop"
    for rel, body in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(body)
    return root


def test_impact_works_across_javascript_and_typescript_files(brain, web_shop):
    brain.index(str(web_shop))

    report = brain.impact_of_change(None, "discount")

    direct = {d.chunk.symbol for d in report.dependents if d.level == 1}
    indirect = {d.chunk.symbol for d in report.dependents if d.level == 2}
    assert "Cart" in direct
    assert "checkout" in indirect
    assert any(d.is_test and d.chunk.path == "src/price.test.ts" for d in report.dependents)
    assert not any("Only Python" in note for note in report.notes)


def test_declaration_names_are_not_uses():
    source = "class Store {\n  constructor(size) {\n    this.size = size;\n  }\n\n  get(key) {\n    return lookup(key);\n  }\n}\n"
    (chunk,) = chunk_file("store.js", "javascript", source)

    assert "lookup" in chunk.refs
    assert not {"constructor", "get"} & set(chunk.refs)


def test_impact_follows_a_constructor_to_the_code_that_builds_the_class(brain, tmp_path):
    filler = "\n".join(f"    this.f{i} = {i};" for i in range(MAX_CHUNK_LINES))
    files = {
        "src/price.ts": "export function discount(price: number) {\n  return price * 0.9;\n}\n",
        "src/order.ts": (
            'import { discount } from "./price";\n\n'
            f"export class Order {{\n  constructor(total: number) {{\n    this.total = discount(total);\n{filler}\n  }}\n\n"
            "  describe() {\n    return String(this.total);\n  }\n}\n"
        ),
        "src/shop.ts": 'import { Order } from "./order";\n\nexport function buy(total: number) {\n  return new Order(total);\n}\n',
        "src/errors.ts": "export class ShopError extends Error {\n  constructor(message: string) {\n    super(message);\n  }\n}\n",
    }
    root = tmp_path / "orders"
    for rel, body in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(body)
    brain.index(str(root))

    report = brain.impact_of_change(None, "discount")

    reached = {d.chunk.symbol for d in report.dependents}
    assert "Order.constructor" in reached
    assert "buy" in reached  # builds an Order, so it is affected
    assert "ShopError" not in reached  # merely has a constructor of its own


def test_short_class_names_and_private_methods_are_uses():
    (chunk,) = chunk_file("index.js", "javascript", "export function make(input) {\n  return Ky.create(input);\n}\n")
    private = "class Ky {\n  #prepare() {}\n\n  run() {\n    return this.#prepare();\n  }\n}\n"
    (klass,) = chunk_file("ky.js", "javascript", private)

    assert "Ky" in chunk.refs
    assert "#prepare" in klass.refs and "run" not in klass.refs
