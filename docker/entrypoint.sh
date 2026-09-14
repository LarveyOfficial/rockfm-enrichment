#!/usr/bin/env bash
set -euo pipefail

echo "RockFM Enrichment starting"
echo "  source     : ${SOURCE_URL:-https://rockfm-cope.flumotion.com/playlist.m3u8}"
echo "  timezones  : ${SOURCE_TZ:-Europe/Madrid} -> ${LOCAL_TZ:-America/New_York}"
echo "  buffer     : ${BUFFER_HOURS:-24}h in ${DATA_DIR:-/data}"
echo "  recognizer : ${RECOGNIZER:-shazamio}"

mkdir -p "${DATA_DIR:-/data}"/{segments,art}

# Seeding the fingerprint index from RockFM's rotation catalog takes ~20 minutes
# on a cold start, so it runs in the background rather than delaying playout.
if [ "${SEED_ON_START:-true}" = "true" ]; then
  echo "  seeding fingerprint index in the background"
  python -m rockfm.seed >/proc/1/fd/1 2>&1 &
fi

exec supervisord -c /app/docker/supervisord.conf
