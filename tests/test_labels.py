from rockfm import labels


def test_song_uses_the_conventional_artist_dash_title():
    row = {"kind": "cancion", "artist": "Blondie", "title": "Denis", "album": "Plastic Letters"}
    assert labels.stream_title(row) == "Blondie - Denis"


def test_song_missing_an_artist_still_shows_the_title():
    assert labels.stream_title({"kind": "cancion", "title": "Denis"}) == "Denis"


def test_advert_uses_its_spanish_label():
    assert labels.stream_title({"kind": "publicidad"}) == "Publicidad - Volvemos enseguida"


def test_programme_shows_the_show_and_presenters():
    row = {"kind": "programa", "show_title": "RockFM Motel", "show_lead": "Rodrigo Contreras"}
    assert labels.stream_title(row) == "RockFM Motel - Rodrigo Contreras"


def test_nothing_playing_falls_back_to_the_station():
    assert labels.stream_title({}) == "RockFM"


def test_unknown_stretch_names_the_station_and_show():
    row = {"kind": "desconocido", "show_title": "RockFM noche"}
    assert labels.stream_title(row) == "RockFM - RockFM noche"


def test_render_keeps_spanish_accents_intact():
    row = {"kind": "programa", "show_title": "El Pirata y su banda",
           "show_lead": "El Pirata, Sayago, Álex Clavero y Raquel Piqueras"}
    primary, secondary = labels.render(row)
    assert primary == "El Pirata y su banda"
    assert "Álex Clavero" in secondary
