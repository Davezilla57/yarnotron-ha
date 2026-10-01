# Yarnotron backend, headless: no microphone, no speaker, no voice extra.
# Home Assistant supplies the ears and the voice; this image is the story brain.
#
# NOT built or run in the environment this was written in (no Docker daemon).
# The same install and start steps were exercised outside Docker; see
# docs/yarnotron-ha-guide.html, "Verification".

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Stories, sessions and preferences live here; mount a volume on it.
    YARNOTRON_DATA_DIR=/data

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
# README.md is the package readme named in pyproject.toml, so the build needs it.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# Base dependencies only: no `voice`, no `dev`, no `postgres`. --no-editable
# installs the package into the venv so /app/src is not needed at runtime.
RUN uv sync --frozen --no-dev --no-editable

RUN useradd --system --uid 10001 --home-dir /data yarnotron \
    && mkdir -p /data \
    && chown yarnotron /data
USER yarnotron
VOLUME /data
EXPOSE 8989

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8989/health', timeout=4)"]

# 0.0.0.0 inside the container; publish the port only to the network you trust,
# and set YARNOTRON_API_TOKEN (docker-compose.yml refuses to start without it).
CMD ["/app/.venv/bin/uvicorn", "yarnotron.main:app", "--host", "0.0.0.0", "--port", "8989"]
