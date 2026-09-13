# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Python 3.12 specifically: shazamio-core ships wheels for it, and building that
# Rust extension from source is not something to inflict on an Unraid box.

ARG INCLUDE_SEGMENTER=true

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    HTTP_PORT=8080

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl supervisor tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir . \
 && pip install --no-cache-dir "numpy>=1.26" "scipy>=1.12" shazamio

# The speech/music CNN is what separates a DJ talking from an unrecognised song.
# It drags in TensorFlow (~1GB), so it can be left out for a constrained host --
# set SEGMENTER=light to match, and expect the classifier to abstain more often.
RUN if [ "$INCLUDE_SEGMENTER" = "true" ]; then \
      pip install --no-cache-dir inaSpeechSegmenter \
      && python -c "from inaSpeechSegmenter import Segmenter; Segmenter(vad_engine='smn', detect_gender=False)" ; \
    fi

COPY docker ./docker
COPY scripts ./scripts

VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${HTTP_PORT}/api/health" || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
