#!/usr/bin/env bash
set -euo pipefail

echo "RockFM Enrichment starting"
echo "  source     : ${SOURCE_URL:-https://rockfm-cope.flumotion.com/playlist.m3u8}"
echo "  timezones  : ${SOURCE_TZ:-Europe/Madrid} -> ${LOCAL_TZ:-America/New_York}"
echo "  buffer     : ${BUFFER_HOURS:-24}h in ${DATA_DIR:-/data}"
echo "  recognizer : ${RECOGNIZER:-shazamio}"

mkdir -p "${DATA_DIR:-/data}"/{segments,art}

exec supervisord -c /app/docker/supervisord.conf
