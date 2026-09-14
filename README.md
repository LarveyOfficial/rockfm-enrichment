# RockFM Enrichment

Records Spain's [RockFM](https://www.rockfm.fm/) and replays it on a wall-clock
delay, so the Madrid morning show lands on your morning. While the audio waits in
the buffer it is identified by audio fingerprinting and enriched with title,
artist, album, year, artwork, duration and elapsed time. Everything that is not a
song — adverts, DJ talk, news, idents — is labelled in Spanish with the real
programme name and presenters.

It serves a delayed HLS stream and a now-playing API, and can push both the audio
and the corrected metadata into AzuraCast.

## Why

RockFM's stream carries its own metadata, but it runs hours to days behind — the
tag will happily claim a Joey Ramone track while Blondie is playing. Nothing here
reads it. Song identity comes from the audio alone.

## How it works

```
RockFM HLS ──► ingest ──► 24h buffer on disk (segments stored byte-identical)
                              │
                              ▼   (runs ~6h ahead of airtime)
                          analyzer ──► local fingerprint index
                              │             └── miss ──► external recogniser ──► learn
                              ▼
                          classifier ──► repetition · speech/music · schedule
                              │
                              ▼
                          timeline ──► playout (HLS + now-playing API)
                                   └──► AzuraCast (live source + metadata)
```

The six-hour delay is the whole trick: there is time to probe the audio, bisect
song boundaries and look things up long before any of it goes out.

**Recognition.** A local landmark fingerprint index answers most lookups for
free. Only audio it has never heard goes to an external recogniser, and the
answer is then learned from the broadcast itself — so a song costs about one
external call the first time it airs and nothing afterwards. Measured on live
radio: 16 calls on a cold index, 0 on a warm one over the same audio.

**Adverts vs DJ talk.** Adverts repeat; a presenter talking never does. Non-song audio is fingerprinted into a second index, so anything heard
before is an advert or an ident and anything genuinely new is live talk. A
trained speech/music model keeps an unrecognised *song* from being mistaken for
talk, and the station's own schedule supplies the programme name and artwork.
When the signals disagree, the vaguer true label wins: naming the show beats
guessing "Publicidad". Stretches shorter than `MIN_NONMUSIC_SECONDS` are left
alone entirely -- station jingles run about two seconds and no verdict fits
them, so they keep the station name rather than being forced into a category.

**The delay is not a constant.** Spain and the United States change DST on
different dates, so Madrid→New York is six hours for most of the year and five
for a few weeks each spring and autumn. The offset is recomputed from the two
wall clocks, and the hour-long jump is deferred to a non-music boundary so it
never cuts a song in half.

## Running it

```bash
docker compose up -d --build
```

Pushes to `master` rebuild and publish the image automatically via GitHub
Actions, using the workflow's own token -- no registry credentials are stored in
the repository. Because dependencies are installed above `COPY src`, a
code-only change reuses the heavy layers and finishes in about a minute.

Building locally with podman needs `--format docker`, otherwise the image is
written in OCI format and the `HEALTHCHECK` is silently dropped:

```bash
podman build --format docker -t rockfm-enrichment:latest .
```

Note that rootless podman without systemd never actually runs the healthcheck --
the container sits at `starting` forever. That is a podman behaviour, not a
problem with the image; `curl localhost:8080/api/health` reports the same thing
directly, and Docker runs the check normally.

Then point a player at `http://<host>:8080/hls/playlist.m3u8`, or open
`http://<host>:8080/` for a small web player. On Unraid, use
`unraid-template.xml`.

### The dashboard

`http://<host>:8080/dashboard` shows what the pipeline is actually doing: the
recorded buffer as a colour-coded timeline, every boundary the analyzer chose,
the metadata attached to each item, and how it was identified -- local
fingerprint or external recogniser, with its confidence.

Crucially it plays any of it back **immediately**, without waiting out the delay.
Click a song to hear it from its start, or use *hear the start/end boundary* to
play the fifteen seconds either side of a transition. A playhead tracks across
the timeline while audio plays, so you can see whether a boundary lands where it
sounds like it should. That is the intended way to judge how well recognition is
working before trusting it.

It is a read-only view on `/api/status`, `/api/timeline` and `/replay.m3u8`; it
changes nothing.

The first run seeds the fingerprint index from RockFM's 951-track rotation
catalog. That takes around twenty minutes in the background and happens once;
playout works throughout.

### On Unraid

The image is published to GHCR, so there is nothing to build:

```
ghcr.io/larveyofficial/rockfm-enrichment:latest
```

Copy `unraid-template.xml` to
`/boot/config/plugins/dockerMan/templates-user/` and add the container from
*Docker → Add Container → Template: user-defined*. The only setting that must be
right is **Your Timezone**; the defaults cover everything else, and the AzuraCast
fields can stay blank until you want them.

Only the full image is published. If the server is short on space, build locally
with `--build-arg INCLUDE_SEGMENTER=false` and set `SEGMENTER=light` -- that
drops the speech/music model and takes the image from ~2.7 GB to ~0.9 GB, at the
cost of the classifier abstaining on speech rather than deciding.

**Nothing plays for the first six hours.** That is the buffer filling to match
the delay, not a fault: the dashboard shows a countdown and the health check
stays green throughout. Recognition starts working within minutes, so the
dashboard is useful long before the audio is.

### Key settings

| Variable | Default | Notes |
|---|---|---|
| `LOCAL_TZ` | `America/New_York` | Delay is derived from this vs `Europe/Madrid` |
| `BUFFER_HOURS` | `24` | Must exceed the delay. ~29 MB per hour |
| `RECOGNIZER` | `shazamio` | or `audd`, `acrcloud` |
| `SEGMENTER` | `ina` | or `light` for a constrained host |
| `DISPLAY_LANGUAGE` | `es` | or `en` |
| `MIN_NONMUSIC_SECONDS` | `5` | Below this, a non-song stretch gets no verdict |
| `PUBLIC_URL` | — | Makes artwork URLs absolute for AzuraCast |

`DELAY_SECONDS` forces a fixed delay, which is useful for testing — set it to
something small and the stream becomes near-live.

### AzuraCast

Set `AZURACAST_BASE_URL`, `AZURACAST_STATION_ID` and `AZURACAST_API_KEY` (the key
needs the station's *Broadcasting* permission), plus `AZURACAST_DJ_URL` and
`AZURACAST_DJ_PASSWORD` to push the audio as a live source.

The station must use the **Liquidsoap backend**. Metadata updates go through
`/api/station/{id}/nowplaying/update`, which resolves to Liquidsoap's
`custom_metadata.insert` — a relay-only station has no backend adapter and the
call fails.

Artwork is sent along with the metadata, but whether AzuraCast keeps our URL or
substitutes its own Last.fm/MusicBrainz lookup depends on the version.
`MetadataClient.probe_art_support()` reports which. Either way the container's
own now-playing API carries the official RockFM artwork.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev,recognize]'
.venv/bin/python -m pytest
```

`shazamio` needs Python 3.12 — its Rust extension has no wheels for 3.13+.

Useful scripts:

| Script | What it does |
|---|---|
| `scripts/demo_recognition.py` | Sweeps live radio and prints what each probe really is |
| `scripts/demo_timeline.py` | Records, analyzes, classifies, prints the timeline |
| `scripts/validate_fingerprint.py` | Recall/precision of the fingerprinter against real audio |

## A note on Shazam

`shazamio` must be called with `SearchParams(segment_duration_seconds=12)`.
Shazam's signature format is built around a twelve-second sample and its backend
rejects signatures made from any other length — without it every lookup silently
returns no match, including for tracks the Shazam app identifies instantly. With
it, a live sweep identified 20 of 21 probes. It looks like a tuning knob. It is
not.

`shazamio` is reverse-engineered and unofficial, which is why the local index
does the heavy lifting and this is only consulted for genuinely new audio. AudD
and ACRCloud adapters are included for anyone who would rather pay for a
supported API.

## Known limitations

- Album selection comes from free search APIs and occasionally lands on a
  compilation rather than the original release. Years are reliable; album titles
  are best-effort.
- Advert detection needs history. Until a cluster has been heard twice it is
  labelled with the programme name rather than guessed as an advert, so expect
  the first day to under-report adverts.
- The classifier has been exercised against music and idents; a validation run
  across a daytime block with real advert breaks is still outstanding.
- The speech/music CNN makes the image large. Build with
  `--build-arg INCLUDE_SEGMENTER=false` and set `SEGMENTER=light` for a much
  smaller image, at the cost of the classifier abstaining more often.
