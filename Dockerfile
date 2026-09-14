# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Python 3.12 specifically: shazamio-core ships wheels for it, and building that
# Rust extension from source is not something to inflict on an Unraid box.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    HTTP_PORT=8080

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl supervisor tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies are installed before the source is copied, so editing the
# application does not reinstall them on every rebuild.
RUN pip install --no-cache-dir \
      "httpx>=0.27" "fastapi>=0.115" "uvicorn[standard]>=0.32" \
      "numpy>=1.26" "scipy>=1.12" shazamio

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY docker ./docker
COPY scripts ./scripts

VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${HTTP_PORT}/api/health" || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
