# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for the Sicurre-ML inference API.
#
# Stage 1 (builder): installs Python dependencies with uv into a virtual env.
# Stage 2 (runtime): copies only the venv and source — no build tooling.
#
# Build:  docker build -t sicurre-ml .
# Run:    docker run -p 8000:8000 --env-file .env sicurre-ml
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency install ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Pin the build tool and avoid a floating cross-registry build stage.
RUN python -m pip install --no-cache-dir uv==0.11.14

WORKDIR /app

# Copy only the dependency manifest first so Docker layer cache is reused
# when source files change but dependencies have not.
COPY pyproject.toml uv.lock ./

# Install base + inference deps only.
# --frozen: fail if uv.lock is out of date (prevents silent drift).
# --no-dev: excludes the dev group (pytest, ruff, mypy — not needed at runtime).
RUN uv sync --group inference --no-dev --frozen

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ARG SERVICE_VERSION=0.1.0

# Non-root user for security — never run production containers as root.
RUN groupadd --system sicurre && useradd --system --gid sicurre --no-create-home sicurre

WORKDIR /app

# Copy the pre-built virtual env from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Copy application source.
COPY src/ src/

# The venv is at /app/.venv; add its bin to PATH so uv run resolves correctly.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVICE_VERSION=${SERVICE_VERSION}

# ONNX model cache is written here at runtime; map a named volume in compose.
ENV ONNX_MODEL_CACHE_DIR=/tmp/sicurre_onnx

RUN mkdir -p /tmp/sicurre_onnx && chown sicurre:sicurre /tmp/sicurre_onnx

USER sicurre

EXPOSE 8000

# Single worker: the ONNX session is held in lru_cache; multiple workers would
# each load their own 425 MB model. Scale horizontally with multiple containers
# behind a load balancer instead of multiple in-process workers.
CMD ["uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--access-log"]
