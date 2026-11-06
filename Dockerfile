# syntax=docker/dockerfile:1

# ---- build: install dependencies and pre-download the embedding model ---------------------------
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never FASTEMBED_CACHE_PATH=/models
WORKDIR /app

# Dependencies first, so code changes don't re-download them.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev

# Bake the model into the image: fast cold starts, and no download at runtime.
RUN /app/.venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# ---- runtime: small image, non-root user ------------------------------------------------------
FROM python:3.12-slim
LABEL io.modelcontextprotocol.server.name="io.github.Nomannazir/codebase-brain-mcp" \
      org.opencontainers.image.source="https://github.com/Nomannazir/codebase-brain-mcp" \
      org.opencontainers.image.description="MCP server: a cited, searchable memory of a codebase" \
      org.opencontainers.image.licenses="MIT"
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 brain \
    && mkdir /data && chown brain /data

COPY --from=build /app /app
COPY --from=build /models /models
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models \
    CODEBASE_BRAIN_HOME=/data \
    PORT=8765

USER brain
VOLUME ["/data"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=3)"

# CODEBASE_BRAIN_TOKEN must be set at run time; the server refuses to start publicly without it.
CMD ["sh", "-c", "exec codebase-brain serve --transport streamable-http --host 0.0.0.0 --port \"$PORT\""]
