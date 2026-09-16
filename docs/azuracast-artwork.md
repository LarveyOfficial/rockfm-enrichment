# Pushing artwork to AzuraCast

Everything below was read out of AzuraCast's own source, not inferred from
behaviour. Paths are relative to `AzuraCast/AzuraCast` on `main`.

## The problem

We push `title`, `artist`, `album` and `art` to
`POST /api/station/{id}/nowplaying/update`. AzuraCast shows the right title and
artist, an empty album, and the wrong picture — either the station's default
album art or a cover it looked up itself.

## Why `art` can never arrive that way

`UpdateMetadataAction` forwards every request parameter untouched:

```php
$backend->updateMetadata($station, $request->getParams());
```

`Liquidsoap::updateMetadata` then hands them to
`ConfigWriter::annotateArray`, which filters:

```php
$values = array_filter($values, fn($val, $key) =>
    $val !== null && in_array($key, AnnotateNextSong::ALLOWED_ANNOTATIONS, true),
    ARRAY_FILTER_USE_BOTH);
```

`AnnotateNextSong::ALLOWED_ANNOTATIONS` is:

```
title, artist, duration, song_id, media_id, playlist_id, jingle_mode,
request_id, sq_id, liq_amplify, azuracast_autocue, azuracast_cache_key,
autocue_cue_in, autocue_cue_out, autocue_fade_in, autocue_fade_out,
autocue_start_next
```

No `art`. No `album`. Both are discarded before the annotation exists. This is
why the album field has been empty in every reading we took — same cause, and it
was visible from the first dump.

No station setting, mount point, stream format or ID3 frame changes this. The
parameter has nowhere to go.

## How AzuraCast actually decides artwork

`SongApiGenerator::getAlbumArtUrl` tries four things in order:

1. **Station media library** — only when the song object is a `StationMedia`
2. **Remote lookup** — if `allowRemoteArt` and the external-art setting is on
3. **Streamer artwork** — if live and that streamer has custom art uploaded
4. **Station default album art**

Nothing reads the audio source. A relayed Icecast/Shoutcast stream cannot supply
art no matter what is embedded in it.

## The way in: `media_id`

With a Liquidsoap backend, `NowPlayingApiGenerator` sets

```php
$updateSongFromNowPlaying = !$station->backend_type->isEnabled();   // false for us
```

so the current song is **not** read from the stream text. It comes from the
history/queue records, and `FeedbackCommand::getSongHistory` builds those:

```php
if (empty($payload['media_id'])) {
    $newSong = Song::createFromArray(['artist' => ..., 'title' => ...]);
    return new SongHistory($station, $newSong);      // bare Song → no branch 1
}
$media = $this->em->find(StationMedia::class, $payload['media_id']);
...                                                   // StationMedia → branch 1
```

So a history row carrying `media_id` **is** a `StationMedia`, branch 1 fires, and
art is served from `api:stations:media:art` for that record — a record we can
write to. And `media_id` survives the annotation filter, unlike `art`.

## Plan

Keep one placeholder media record in the station library: a short silent file,
never played, purely a metadata carrier. On each boundary:

1. `POST /api/station/{id}/art/{media_id}` — upload the current artwork
   (documented as "Sets the album art for a track")
2. `PUT /api/station/{id}/file/{media_id}` — set its title and artist, because
   with `media_id` present AzuraCast takes the song from the media record rather
   than from our payload
3. `POST /api/station/{id}/nowplaying/update?media_id=…` — point now-playing at it

Three calls per song change, roughly one change every three minutes. The artwork
is then exactly the image we send: no external lookup, no default.

### Settled: the annotation path does carry `media_id`

`FeedbackCommand::doRun` begins:

```php
if (!$asAutoDj) {
    return false;
}
```

so it was an open question whether our `custom_metadata.insert` path reaches it
carrying `media_id`, or whether only the AutoDJ callback does. It does. Read
back from a live station while the three calls were running:

```
art: https://radio.luisvervaet.dev/api/station/rock_fm/art/2eba06a6c6dff045b57e5537-….jpg
```

That is the `api:stations:media:art` route -- branch 1, the station media
library -- not the station default under `/static/uploads/…/album_art.*` and not
a remote lookup. The carrier record is what AzuraCast serves. No fallback to
driving the queue directly is needed.

Note the album field stays empty, exactly as the filter predicts: `album` is not
in `ALLOWED_ANNOTATIONS` either, and the carrier's album is not what
`SongApiGenerator` reads for that field.

### Variants

- **One shared record, rewritten each time** (above). Fewest records; needs the
  title/artist update on every change. Watch `isDifferentFromCurrentSong` — the
  song hash must change or AzuraCast may treat it as the same track.
- **One record per distinct song.** More uploads up front, but each record is
  written once and reused on later airings, and no per-boundary title update.
  Better fit for a station with a repeating rotation.

## Related

Getting here also required the station's remote-URL playlist to be typed
**Stream**, not **Playlist**. Typed as a playlist, AzuraCast downloads the
`.m3u8`, reads it as an M3U track list, takes its one non-comment line and hands
that to Liquidsoap without resolving it against the base URL:

```
Response (200): {"uri":"annotate:playlist_id="17":chunks.m3u8"}
[request:3] Nonexistent file or ill-formed URI "chunks.m3u8"!
```

Liquidsoap then retries once a second forever and falls back to `error.mp3`,
which surfaces as "Station Offline" with the queue stuck on "Remote Playlist
URL". Any HLS master playlist fails this way, RockFM's own included -- the
relative child reference is correct HLS, and AzuraCast is simply reading it as
something it is not.
