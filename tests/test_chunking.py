import textwrap

from codebase_brain_mcp.chunking import MAX_CHUNK_LINES, chunk_file


def test_python_is_split_by_definition_with_exact_lines():
    source = textwrap.dedent("""
        import os

        @decorator
        def first():
            return 1


        class Thing:
            def method(self):
                return 2
        """).lstrip()
    chunks = chunk_file("mod.py", "python", source)
    by_symbol = {c.symbol: c for c in chunks}

    assert by_symbol[None].kind == "module" and by_symbol[None].text == "import os"
    assert (by_symbol["first"].start_line, by_symbol["first"].end_line) == (3, 5)  # decorator included
    assert by_symbol["first"].kind == "function"
    assert by_symbol["Thing"].kind == "class"
    assert by_symbol["first"].citation == "mod.py:3-5"


def test_big_class_is_split_into_methods():
    body = "\n".join(f"        x{i} = {i}" for i in range(MAX_CHUNK_LINES))
    source = f"class Big:\n    def small(self):\n        return 1\n\n    def large(self):\n{body}\n"
    symbols = [c.symbol for c in chunk_file("big.py", "python", source)]

    assert "Big.small" in symbols and "Big.large" in symbols


def test_markdown_sections_keep_their_heading_path():
    source = "# Guide\n\nIntro.\n\n## Install\n\nRun it.\n\n```\n# not a heading\n```\n"
    symbols = [c.symbol for c in chunk_file("README.md", "markdown", source)]

    assert symbols == ["Guide", "Guide > Install"]


def test_invalid_python_falls_back_to_windows():
    chunks = chunk_file("broken.py", "python", "def oops(:\n    pass\n")

    assert len(chunks) == 1 and chunks[0].kind == "code"


def test_empty_file_has_no_chunks():
    assert chunk_file("empty.py", "python", "\n\n") == []
