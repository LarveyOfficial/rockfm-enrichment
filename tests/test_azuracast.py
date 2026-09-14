from rockfm.azuracast import metadata_for

SONG = {
    "kind": "cancion", "title": "Denis", "artist": "Blondie",
    "album": "Plastic Letters", "year": 1977, "art_url": "/art/abc.jpg",
    "show_title": None, "show_lead": None,
}
# A stretch between songs that the schedule had nothing to say about.
UNNAMED = {"kind": "programa", "art_url": None, "show_title": None, "show_lead": None}
SHOW = {
    "kind": "programa", "art_url": "/art/show.jpg",
    "show_title": "El Pirata y su banda",
    "show_lead": "El Pirata, Sayago, Álex Clavero y Raquel Piqueras",
}


def test_song_maps_to_real_fields():
    meta = metadata_for(SONG, "es")
    assert meta.title == "Denis"
    assert meta.artist == "Blondie"
    assert meta.album == "Plastic Letters"


def test_an_unnamed_stretch_falls_back_to_the_station():
    """No schedule entry, so the honest label is the station itself."""
    meta = metadata_for(UNNAMED, "es")
    assert meta.title == "RockFM"


def test_programme_shows_presenters_as_the_artist():
    meta = metadata_for(SHOW, "es")
    assert meta.title == "El Pirata y su banda"
    assert "Álex Clavero" in meta.artist


def test_relative_art_becomes_absolute_when_a_public_url_is_known():
    assert metadata_for(SONG, "es", "https://radio.example.com/").art == (
        "https://radio.example.com/art/abc.jpg"
    )


def test_relative_art_is_left_alone_without_a_public_url():
    assert metadata_for(SONG, "es").art == "/art/abc.jpg"


def test_params_omit_empty_fields():
    params = metadata_for(UNNAMED, "es").as_params()
    assert "album" not in params
    assert "art" not in params
    assert params["title"] == "RockFM"
