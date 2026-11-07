---
name: codebase-brain
description: Answer questions about a codebase with exact file:line citations, explain why code looks the way it does from git history, check what a change will break before editing, and find out-of-date docs. Use when the user asks how something works, where something is, why code was written a certain way, what depends on a function, or whether docs are current, and the codebase-brain MCP server is connected.
---

# Codebase Brain

The `codebase-brain` MCP server keeps a local, searchable index of a repository: its code, its docs and its git
history. Use it instead of guessing or reading files at random, and back every claim with a citation.

## Setup

If the tools aren't available, the server isn't connected. Tell the user to run:

```bash
claude mcp add codebase-brain -- uvx codebase-brain-mcp
```

For other clients the server command is `uvx codebase-brain-mcp` (stdio).

## Before anything else

1. Call `list_repos`. If the repo the user means isn't listed, call `index_repo` with its local path or https git
   URL. Re-indexing is cheap: only changed files are re-embedded, so re-run it if the code changed since the
   `commit` it was indexed at.
2. When more than one repo is indexed, pass `repo` on every call.

## Pick the tool by the question

| The user asks | Call |
|---|---|
| "How does X work?", "Where is Y handled?" | `ask_codebase(question)`, then answer from its sources only |
| A narrower lookup, or results filtered by folder or kind | `search(query, path_prefix?, kinds?)` |
| "Why is it written like this?", "Can I remove this check?" | `explain_history(symbol=...)` or `explain_history(path=..., start_line=..., end_line=...)` |
| "When did we change Z?", "Which commit fixed the timezone bug?" | `search_history(query)` |
| "What breaks if I change X?", or before you edit a function yourself | `impact_of_change(symbol)` |
| "Are the docs up to date?", "Is the README correct?" | `find_doc_drift(path_prefix?)` |
| You need the full text behind a citation | `read_lines(path, start, end)` |

`kinds` accepts: function, class, method, interface, type, enum, module, section, code.

## Rules

- **Cite every claim** as `path:start-end`, exactly as the tool returned it. Don't invent line numbers, and don't
  cite a file you haven't seen in a result.
- **Answer only from the sources you got.** If they don't answer the question, say so, then search again with
  different words: the identifier name, or a plain description of the behaviour. Don't fill gaps from memory.
- **Search both ways.** Exact names (`TimestampSigner.unsign`, `normalizeRetryOptions`) find code fastest. Plain
  words ("how are expired tokens rejected") find code when you don't know the names. If one misses, try the other.
- **Check history before removing odd-looking code.** A strange `try`/`except` or special case often exists
  because of a past bug. `explain_history` shows the commit and pull request that added it, so quote that reason.
- **Run `impact_of_change` before editing a shared function.** List what it reports (direct users, indirect users,
  tests) and update or run those tests as part of the change.
- **Open before you quote.** Search results are chunks. Call `read_lines` for the full function before you explain
  details or propose an edit.

## Know the limits, and say them when they matter

- `impact_of_change` matches names, so it can over-report when several things share a name. It notes these
  same-named definitions, and you should mention them. It misses dynamic calls such as `getattr`, string dispatch
  and callbacks. Python, JavaScript and TypeScript are analysed structurally; for other languages, also `search`
  for the name.
- `find_doc_drift` checks code examples in docs against Python signatures only. File, symbol, setting and flag
  checks work for any language.
- Search can rank tests above source when tests describe the same behaviour in plain words. If the top results
  are all tests, narrow with `path_prefix` (for example `src/`) or `kinds=["function", "method", "class"]`.
- History shows commit authors by name only. Pull-request titles and descriptions come from GitHub, for repos
  hosted there.

## Example

User: "Why does `unsign` catch `OverflowError`, and is it safe to remove?"

1. `explain_history(symbol="unsign")` finds commit `3edfbbb`, "handle OverflowError in timed unsign on 32-bit
   systems", and the pull request behind it.
2. `impact_of_change(symbol="unsign")` lists the callers and the tests that cover it.
3. Answer: it was added to fix a crash on 32-bit systems (cite the commit and `src/itsdangerous/timed.py:72-158`).
   Removing it would bring that crash back, and these tests would need checking: …
