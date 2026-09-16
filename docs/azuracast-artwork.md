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

### Open question to settle first

`FeedbackCommand::doRun` begins:

```php
if (!$asAutoDj) {
    return false;
}
```

It needs confirming that our `custom_metadata.insert` path reaches this carrying
`media_id`, rather than only the AutoDJ callback doing so. If it does not, the
fallback is the same three calls driving the queue directly instead of the
annotation — more work, same result.

Cheapest test: upload one media record, set a distinctive image on it, push
`media_id` with the metadata, and read `/api/nowplaying/{id}` to see whether
`now_playing.song.art` becomes the media art URL.

### Variants

- **One shared record, rewritten each time** (above). Fewest records; needs the
  title/artist update on every change. Watch `isDifferentFromCurrentSong` — the
  song hash must change or AzuraCast may treat it as the same track.
- **One record per distinct song.** More uploads up front, but each record is
  written once and reused on later airings, and no per-boundary title update.
  Better fit for a station with a repeating rotation.

## Related

`MetadataClient.probe_art_support()` exists to detect whether AzuraCast kept our
artwork. It has never been called from anywhere. Now that the answer is known to
be a flat no for the `art` parameter, it should be deleted rather than wired up.
