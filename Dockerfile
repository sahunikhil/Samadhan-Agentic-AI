# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------------------
# One image, every service:  samadhan serve api | commerce | helpdesk | knowledge
# One artifact to build, scan, sign and roll out - the command selects the role.
# ---------------------------------------------------------------------------------------

FROM python:3.12-slim-bookworm AS builder
# Pin uv by copying the static binary from its official image (reproducible builds).
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# 1) Dependencies only - this layer is cached until pyproject.toml / uv.lock change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --extra observability

# 2) The project itself.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra observability

# 3) Bake the ONNX models (dense, BM25, reranker ~150 MB) into the image so a
#    scale-from-zero container doesn't download them on its first request.
ARG PRELOAD_MODELS=true
RUN mkdir -p /app/models && if [ "$PRELOAD_MODELS" = "true" ]; then \
      /app/.venv/bin/python -c "from samadhan.config import RetrievalSettings as R; from samadhan.rag.embeddings import EmbeddingModels as E; E(R(model_cache_dir='/app/models')).warmup()"; \
    fi

# ---------------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime
LABEL org.opencontainers.image.title="samadhan" \
      org.opencontainers.image.description="Multi-agent customer support: LangGraph + MCP + hybrid RAG" \
      org.opencontainers.image.licenses="MIT"

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/models /app/models
COPY --chown=app:app LICENSE THIRD_PARTY_NOTICES.md ./
COPY --chown=app:app data ./data
COPY --chown=app:app evals/datasets ./evals/datasets

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAMADHAN_RETRIEVAL__MODEL_CACHE_DIR=/app/models \
    SAMADHAN_OBSERVABILITY__LOG_JSON=true \
    HF_HUB_OFFLINE=1

# Dev/Compose: the auto-generated service-token key pair lives on a shared volume mounted here.
RUN mkdir -p /app/.data/keys && chown -R app:app /app/.data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(os.environ.get('HEALTHCHECK_URL','http://127.0.0.1:8000/healthz'), timeout=4)" || exit 1

CMD ["samadhan", "serve", "api"]
