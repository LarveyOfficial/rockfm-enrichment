"""Tell adverts and jingles apart from live talk, by noticing what repeats.

Adverts and station idents air again and again; a presenter talking never does.
So every stretch we could not name as a song is fingerprinted into a second,
separate index. Audio we have heard before is an advert or a jingle. Audio that
is genuinely new is someone talking.

This costs nothing and needs no external service, and it sharpens every day the
container runs -- which is also its weakness: on a cold index everything looks
novel. Until a cluster has been seen twice the caller should fall back to naming
the programme rather than guessing "advert".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..fingerprint import FingerprintIndex, compute

log = logging.getLogger("rockfm.classify.repetition")

KIND = "nonmusic"
# Speech is far less distinctive than music, so demand a stronger match here
# than for songs; a false cluster would label live talk as an advert.
MIN_VOTES = 24
MIN_SCORE = 0.08
# Two sightings closer together than this are the same airing seen twice, not a
# repeat -- which is what re-analysing already-scanned audio produces.
SAME_AIRING_MS = 120_000


@dataclass(frozen=True)
class Cluster:
    track_id: int
    key: str
    occurrences: int

    @property
    def is_repeat(self) -> bool:
        return self.occurrences >= 2


class RepetitionIndex:
    def __init__(self, conn) -> None:
        self.conn = conn
        self.index = FingerprintIndex(conn)

    def _next_key(self) -> str:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM fp_tracks WHERE kind = ?", (KIND,)
        ).fetchone()
        return f"cluster:{int(row['n']) + 1}"

    def observe(self, samples: np.ndarray, at_ms: int) -> Cluster | None:
        """Match this audio against what we have heard before, or record it."""
        hashes = compute(samples)
        if not hashes:
            return None

        match = self.index.match(hashes, kind=KIND, min_votes=MIN_VOTES)
        if match is not None and match.score >= MIN_SCORE:
            occurrences = self.index.record_sighting(match.track_id, at_ms, SAME_AIRING_MS)
            log.debug("heard %s again (airing %d)", match.key, occurrences)
            return Cluster(match.track_id, match.key, occurrences)

        key = self._next_key()
        track_id = self.index.add(
            kind=KIND, key=key, hashes=hashes, source="repetition", anchor_ms=at_ms
        )
        self.conn.execute(
            "UPDATE fp_tracks SET last_seen_ms = ? WHERE id = ?", (at_ms, track_id)
        )
        return Cluster(track_id, key, 1)
