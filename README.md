# Project Brain MCP

An [MCP](https://modelcontextprotocol.io) server that gives AI agents a **cited, searchable memory of a codebase**.
Point it at a local folder or a public git URL. Claude, Cursor or any MCP client can then ask questions about the
code and get answers backed by exact `path:start-end` citations.

## Why it's different

- **Code-aware chunking.** Python is split by function, class and method using the real syntax tree, Markdown by
  heading. Each chunk knows its symbol (`TimestampSigner.unsign`) and exact lines, so citations point at real code.
- **Hybrid retrieval.** SQLite FTS5 BM25 (with `camelCase` / `snake_case` identifier splitting) plus local
  embeddings, merged with reciprocal-rank fusion. Exact names and fuzzy questions both work.
- **Local and free.** Embeddings run on your machine (`fastembed`, ONNX). No API keys, and no code leaves your
  computer.
- **Incremental.** Re-indexing only re-embeds files whose content changed. On a 42-file repo, a re-index with no
  changes takes about 2 seconds.
- **Agent-friendly output.** Structured tool results, read-only tool annotations, progress notifications while
  indexing, and clear error messages the model can act on.

## Tools

| Tool | What it does |
|---|---|
| `index_repo(source, name?)` | Index a local path or an `https://` git URL (shallow clone). Reports progress. |
| `ask_codebase(question, repo?, k?)` | The most relevant code and docs as numbered, cited sources, with instructions to answer only from them. |
| `search(query, repo?, k?, path_prefix?, kinds?)` | Raw hybrid search, filterable by path and chunk kind. |
| `read_lines(path, start, end, repo?)` | Open the exact lines behind a citation. Refuses paths outside the repo. |
| `list_repos()` | What's indexed, at which commit, and in which languages. |

Also included: the resource `brain://repos`, and the prompt `answer_with_citations`.

## Quick start

```bash
uv sync
uv run project-brain index https://github.com/pallets/itsdangerous
uv run project-brain search "how is the signature timestamp checked for expiry"
```

```
0.0313  src/itsdangerous/timed.py:72-158  method TimestampSigner.unsign
0.0311  docs/timed.rst:1-28  code
0.0301  src/itsdangerous/timed.py:170-228  class TimedSerializer
```

### Use it from Claude Code

```bash
claude mcp add project-brain -- uv run --directory /path/to/project-brain-mcp project-brain
```

### Configuration

| Variable | Default | |
|---|---|---|
| `PROJECT_BRAIN_HOME` | `~/.project-brain` | Indexes and clones |
| `PROJECT_BRAIN_EMBEDDER` | `fastembed` | `hash` gives an offline, deterministic embedder (used in tests) |
| `PROJECT_BRAIN_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Any fastembed text model |
| `PROJECT_BRAIN_MAX_FILE_BYTES` | `512000` | Larger files are skipped |

## How it works

```
source ─► resolve (local dir / shallow https clone) ─► list files (.gitignore-aware, skips binaries and lockfiles)
       ─► chunk (ast / headings / line windows) ─► embed (fastembed) ─► SQLite: chunks + vectors + FTS5
query  ─► BM25 top 50 ┐
       ─► vector top 50 ┴─► reciprocal-rank fusion ─► filters ─► cited sources
```

## Security

- Only `https://` clone URLs are accepted. Git runs with `GIT_ALLOW_PROTOCOL=https:file` and no terminal prompts.
- `read_lines` resolves paths and refuses anything outside the indexed repository.
- Repo names are validated, so they can't escape the index directory.
- All query tools are marked read-only.

## Development

```bash
uv run pytest     # 17 tests, offline (hash embedder, in-process MCP client)
```

## Roadmap

- [ ] Git history and PR discussions: `explain_history` ("why is it built this way?")
- [ ] Knowledge graph of symbols, files and commits: `impact_of_change`
- [ ] `find_doc_drift`: documentation claims that no longer match the code
- [ ] tree-sitter chunking for more languages
- [ ] Streamable HTTP deployment with auth, retrieval evals, and OpenTelemetry tracing
