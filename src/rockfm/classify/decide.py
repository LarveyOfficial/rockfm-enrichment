"""Decide what a non-song stretch actually was, and name it in Spanish.

Four signals, fused:

  repetition  heard before -> advert; genuinely new -> live talk
  speech/music  keeps an unrecognised *song* from being labelled as talk
  schedule    which programme is on air, its presenters and artwork
  clock       the overnight block is advertised as running without breaks

When the signals do not agree, prefer the vaguer label that is still true: name
the programme rather than guess "Publicidad". A listener seeing the right show
name during an advert is a much smaller error than seeing "Publicidad" while the
presenter is talking.

Very short stretches are left alone entirely. Station jingles run about two
seconds between tracks, and no useful verdict fits them -- they are not adverts
and nobody is talking. Anything below `MIN_NONMUSIC_SECONDS` keeps the station
name rather than being forced into a category it does not belong in.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from .. import db
from .. import strings_es as S
from ..buffer import BufferReader
from ..config import Config
from ..rockfm_api import Programme
from ..schedule import Schedule
from .repetition import Cluster, RepetitionIndex
from .segmenter import MUSIC, SPEECH
from .segmenter import build as build_segmenter

log = logging.getLogger("rockfm.classify")

CHUNK_MS = 15_000
RATE = 16_000
IDENT_MAX_MS = 20_000        # anything repeating and this short is a jingle


@dataclass
class Chunk:
    start_ms: int
    end_ms: int
    kind: str
    cluster_id: int | None
    confidence: float


class Classifier:
    def __init__(
        self,
        config: Config,
        conn,
        schedule: Schedule | None = None,
        segmenter=None,
    ) -> None:
        self.config = config
        self.conn = conn
        self.reader = BufferReader(conn, config)
        self.schedule = schedule or Schedule()
        self.segmenter = segmenter if segmenter is not None else build_segmenter(config.segmenter)
        self.repetition = RepetitionIndex(conn)
        self.min_nonmusic_ms = int(config.min_nonmusic_seconds * 1000)

    # --- helpers ---

    def _madrid(self, at_ms: int) -> datetime:
        return datetime.fromtimestamp(at_ms / 1000, tz=UTC).astimezone(
            self.config.source_tz
        )

    def _programme(self, at_ms: int) -> Programme | None:
        return self.schedule.at(self._madrid(at_ms))

    # --- the decision ---

    def _decide(self, at_ms: int, audio_kind: str, cluster: Cluster | None) -> tuple[str, float]:
        moment = self._madrid(at_ms)
        no_break = self.schedule.is_no_break_block(moment)

        if cluster is not None and cluster.is_repeat:
            # Heard before. During a block the station bills as running without
            # adverts, a repeat is more likely a promo for one of its own
            # programmes, so name the show instead of crying advert.
            if no_break:
                return S.KIND_PROGRAMA, 0.5
            return S.KIND_PUBLICIDAD, 0.75

        if audio_kind == MUSIC:
            # Music we could not name: an obscure track, not someone talking.
            return S.KIND_DESCONOCIDO, 0.4

        if audio_kind == SPEECH:
            return S.KIND_PROGRAMA, 0.6

        # No usable speech/music reading and nothing familiar: name the show.
        return S.KIND_PROGRAMA, 0.3

    def _chunks(self, start_ms: int, end_ms: int) -> list[Chunk]:
        chunks: list[Chunk] = []
        position = start_ms
        while position < end_ms:
            finish = min(position + CHUNK_MS, end_ms)
            if finish - position < self.min_nonmusic_ms:
                break  # trailing sliver: too short to judge
            samples = self.reader.read(position, finish - position, rate=RATE)
            if samples.size == 0:
                position = finish
                continue
            audio_kind = self.segmenter.classify(samples, RATE)
            cluster = None
            if audio_kind != MUSIC:
                cluster = self.repetition.observe(samples, position)
            kind, confidence = self._decide(position, audio_kind, cluster)
            chunks.append(
                Chunk(
                    start_ms=position,
                    end_ms=finish,
                    kind=kind,
                    cluster_id=cluster.track_id if cluster else None,
                    confidence=confidence,
                )
            )
            position = finish
        return chunks

    @staticmethod
    def _merge(chunks: list[Chunk]) -> list[Chunk]:
        merged: list[Chunk] = []
        for chunk in chunks:
            if merged and merged[-1].kind == chunk.kind:
                previous = merged[-1]
                previous.end_ms = chunk.end_ms
                previous.confidence = max(previous.confidence, chunk.confidence)
                previous.cluster_id = previous.cluster_id or chunk.cluster_id
            else:
                merged.append(chunk)
        return merged

    def _commit(self, chunk: Chunk) -> None:
        programme = self._programme(chunk.start_ms)
        db.upsert_timeline(
            self.conn,
            {
                "start_ms": chunk.start_ms,
                "end_ms": chunk.end_ms,
                "kind": chunk.kind,
                "title": None,
                "artist": None,
                "album": None,
                "year": None,
                "art_url": programme.image_url if programme else None,
                "show_title": programme.title if programme else None,
                "show_lead": programme.lead if programme else None,
                "show_image": programme.image_url if programme else None,
                "confidence": chunk.confidence,
                "source": "classifier",
                "cluster_id": chunk.cluster_id,
            },
            int(time.time() * 1000),
        )

    def _commit_short(self, row) -> None:
        """Retire a sliver without labelling it as anything in particular."""
        programme = self._programme(row["start_ms"])
        db.upsert_timeline(
            self.conn,
            {
                "start_ms": row["start_ms"],
                "end_ms": row["end_ms"],
                "kind": S.KIND_DESCONOCIDO,
                "art_url": programme.image_url if programme else None,
                "show_title": programme.title if programme else None,
                "show_lead": programme.lead if programme else None,
                "show_image": programme.image_url if programme else None,
                "confidence": 0.0,
                "source": "too-short",
            },
            int(time.time() * 1000),
        )

    def _retro_label(self, cluster_id: int) -> int:
        """Relabel earlier airings once a cluster turns out to repeat.

        The first time an advert airs it looks novel, so it gets filed as
        programme talk. When it comes round again we know better, and the
        buffered timeline is still there to correct.
        """
        updated = self.conn.execute(
            "UPDATE timeline SET kind = ?, source = 'classifier:retro'"
            " WHERE cluster_id = ? AND kind = ?",
            (S.KIND_PUBLICIDAD, cluster_id, S.KIND_PROGRAMA),
        ).rowcount
        if updated:
            log.info("relabelled %d earlier airing(s) of cluster %d as adverts", updated, cluster_id)
        return updated

    # --- entry point ---

    def run_once(self, limit: int = 40) -> int:
        rows = list(
            self.conn.execute(
                "SELECT * FROM timeline WHERE kind = ? AND source = 'unresolved'"
                " ORDER BY start_ms LIMIT ?",
                (S.KIND_DESCONOCIDO, limit),
            )
        )
        handled = 0
        for row in rows:
            if row["end_ms"] - row["start_ms"] < self.min_nonmusic_ms:
                # A jingle or a gap between tracks. Mark it done so it is not
                # rescanned every pass, but pass no judgement on it.
                self._commit_short(row)
                handled += 1
                continue
            chunks = self._merge(self._chunks(row["start_ms"], row["end_ms"]))
            if not chunks:
                continue
            for chunk in chunks:
                self._commit(chunk)
                if chunk.cluster_id and chunk.kind == S.KIND_PUBLICIDAD:
                    self._retro_label(chunk.cluster_id)
            handled += 1
        return handled


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = Config()
    config.ensure_dirs()
    classifier = Classifier(config, db.connect(config.db_path))
    while True:
        if classifier.run_once() == 0:
            time.sleep(30)


if __name__ == "__main__":
    main()
