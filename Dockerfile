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

# Dependencies are installed before the source is copied, so editing the
# application does not invalidate the TensorFlow layer and force a ~2 GB
# reinstall on every rebuild.
RUN pip install --no-cache-dir \
      "httpx>=0.27" "fastapi>=0.115" "uvicorn[standard]>=0.32" \
      "numpy>=1.26" "scipy>=1.12" shazamio

# The speech/music CNN is what separates a DJ talking from an unrecognised song.
# It can be left out for a constrained host -- set SEGMENTER=light to match, and
# expect the classifier to abstain more often.
#
# inaSpeechSegmenter names `tensorflow[and-cuda]` and `onnxruntime-gpu` as
# requirements, so a plain install drags in the whole CUDA stack -- cuDNN,
# cuBLAS, NCCL, nvcc -- taking the image past 8 GB for hardware a headless
# Unraid box does not have. Installing tensorflow-cpu first does not help: it is
# a different distribution name, so pip pulls `tensorflow` anyway.
#
# So the segmenter goes in with --no-deps and its requirements are listed here
# explicitly, with the CPU builds substituted. pip reports the substitution as a
# dependency conflict, which is exactly what it is; the import checks below are
# what prove the swap works and guard this hand-written list against drift.
RUN if [ "$INCLUDE_SEGMENTER" = "true" ]; then \
      pip install --no-cache-dir "tensorflow-cpu>=2.16" onnxruntime \
      && pip install --no-cache-dir --no-deps inaSpeechSegmenter pyannote.core \
      && pip install --no-cache-dir \
           pandas scikit-image soundfile matplotlib Pyro4 pytextgrid sortedcontainers \
      && python -c "import tensorflow; print('tensorflow', tensorflow.__version__)" \
      && python -c "from inaSpeechSegmenter import Segmenter; Segmenter(vad_engine='smn', detect_gender=False); print('segmenter ok')" ; \
    fi

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
