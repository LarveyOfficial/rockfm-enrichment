"""Ranking tests for enrichment. No network: candidates are supplied directly."""

from rockfm.enrich import _rank, normalise, similarity


def itunes(artist, track, album, year, collection_artist=None):
    return {
        "artistName": artist,
        "trackName": track,
        "collectionName": album,
        "releaseDate": f"{year}-01-01T00:00:00Z" if year else None,
        "collectionArtistName": collection_artist,
    }


READ = lambda item: (  # noqa: E731
    item.get("artistName", ""),
    item.get("trackName", ""),
    item.get("collectionName"),
    int(item["releaseDate"][:4]) if item.get("releaseDate") else None,
)
OWNER = lambda item: item.get("collectionArtistName")  # noqa: E731


def rank(artist, title, items):
    return _rank(artist, title, items, READ, collection_artist=OWNER)


def test_normalise_strips_punctuation_and_case():
    assert normalise("AC/DC — Thunderstruck!") == "ac dc thunderstruck"


def test_similarity_is_case_and_punctuation_insensitive():
    assert similarity("Pretty Fly (For A White Guy)", "pretty fly for a white guy") > 0.95


def test_nothing_matches_when_the_artist_is_wrong():
    assert rank("Blondie", "Denis", [itunes("Some Other Band", "Denis", "X", 1980)]) is None


def test_nothing_matches_when_the_title_is_wrong():
    assert rank("Blondie", "Denis", [itunes("Blondie", "Atomic", "X", 1980)]) is None


def test_tribute_and_karaoke_records_lose_to_the_real_thing():
    best = rank(
        "Green Day",
        "American Idiot",
        [
            itunes("Green Day", "American Idiot", "Lullaby Renditions of Green Day", 2007),
            itunes("Green Day", "American Idiot", "American Idiot", 2004),
        ],
    )
    assert best["collectionName"] == "American Idiot"


def test_various_artists_compilations_lose_even_with_an_innocuous_name():
    # The title gives nothing away; only the collection credit reveals it,
    # and it drags a nonsense year along with it.
    best = rank(
        "Billy Joel",
        "We Didn't Start the Fire",
        [
            itunes("Billy Joel", "We Didn't Start the Fire",
                   "Pop Music: The Modern Era", 1966, collection_artist="Various Artists"),
            itunes("Billy Joel", "We Didn't Start the Fire", "Storm Front", 1989,
                   collection_artist="Billy Joel"),
        ],
    )
    assert best["collectionName"] == "Storm Front"


def test_live_albums_lose_to_studio_releases():
    best = rank(
        "Scorpions",
        "Big City Nights",
        [
            itunes("Scorpions", "Big City Nights", "Tokyo Tapes (Live)", 1978),
            itunes("Scorpions", "Big City Nights", "Love at First Sting", 1984),
        ],
    )
    assert best["collectionName"] == "Love at First Sting"


def test_earliest_release_wins_among_equally_good_studio_albums():
    best = rank(
        "Queen",
        "Bohemian Rhapsody",
        [
            itunes("Queen", "Bohemian Rhapsody", "A Night at the Opera (2011 Mix)", 2011),
            itunes("Queen", "Bohemian Rhapsody", "A Night at the Opera", 1975),
        ],
    )
    assert best["releaseDate"].startswith("1975")


def test_a_compilation_is_still_returned_when_it_is_all_there_is():
    best = rank(
        "The Police",
        "Every Breath You Take",
        [itunes("The Police", "Every Breath You Take", "The Classics", 1983,
                collection_artist="Various Artists")],
    )
    assert best is not None
    assert best["releaseDate"].startswith("1983")


def test_empty_candidate_list():
    assert rank("Blondie", "Denis", []) is None
