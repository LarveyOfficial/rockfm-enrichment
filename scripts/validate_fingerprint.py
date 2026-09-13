from pathlib import Path
import sys, json, subprocess, urllib.request, urllib.parse, tempfile, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import numpy as np

from rockfm import fingerprint, audio, db

SONGS = ["blondie denis", "acdc highway to hell", "queen bohemian rhapsody",
         "nirvana smells like teen spirit", "the police roxanne", "dire straits sultans of swing"]
CACHE = Path(os.environ.get('FP_CACHE', '/tmp/rockfm-fp-previews'))
CACHE.mkdir(exist_ok=True)

def preview(term):
    dest = CACHE / (term.replace(' ','_') + '.m4a')
    if dest.exists(): return dest
    url = "https://itunes.apple.com/search?" + urllib.parse.urlencode(
        {"term": term, "entity": "song", "limit": 1})
    d = json.load(urllib.request.urlopen(url, timeout=20))
    r = d['results'][0]
    urllib.request.urlretrieve(r['previewUrl'], dest)
    print(f"  fetched {r['artistName']} - {r['trackName']}")
    return dest

def broadcastify(samples):
    """Simulate the radio chain: 48k HE-AAC mono round-trip + light noise."""
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td,'a.raw'); enc = os.path.join(td,'b.aac')
        (samples*32767).astype('<i2').tofile(raw)
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','s16le','-ar','8000','-ac','1',
                        '-i',raw,'-c:a','aac','-b:a','48k',enc], check=True)
        out = audio.decode_path(enc)
    out = out + np.random.default_rng(0).normal(0, 0.002, out.shape).astype('float32')
    return out

print("fetching previews...")
paths = {s: preview(s) for s in SONGS}
conn = db.connect(Path(os.environ.get('FP_DB', '/tmp/rockfm-fp.db')))
conn.execute("DELETE FROM fp_hashes"); conn.execute("DELETE FROM fp_tracks")
idx = fingerprint.FingerprintIndex(conn)

REF = SONGS[:5]; HELDOUT = SONGS[5]
print("\nindexing references (30s previews):")
pcm = {}
for s in SONGS:
    pcm[s] = audio.decode_path(paths[s])
for s in REF:
    h = fingerprint.compute(pcm[s])
    idx.add(kind='music', key=s, hashes=h, title=s, source='preview')
    print(f"  {s:<32} {len(pcm[s])/8000:5.1f}s  {len(h):6d} hashes")

def probe(samples, start, dur=10.0):
    a = int(start*8000); b = a + int(dur*8000)
    return samples[a:b]

print("\n--- POSITIVE: clean 10s excerpts at several offsets")
ok = tot = 0
for s in REF:
    for start in (2, 8, 15):
        m = idx.match(fingerprint.compute(probe(pcm[s], start)), kind='music')
        hit = m is not None and m.key == s
        ok += hit; tot += 1
        print(f"  {s:<32} @{start:>2}s -> {'HIT ' if hit else 'MISS'} "
              f"{f'votes={m.votes:4d} score={m.score:.3f} margin={m.margin:4.1f} offset={m.offset_seconds:+.2f}s' if m else ''}")
print(f"  => {ok}/{tot} correct")

print("\n--- ROBUSTNESS: same excerpts through a 48k HE-AAC + noise round-trip")
ok2 = tot2 = 0
for s in REF:
    deg = broadcastify(pcm[s])
    for start in (2, 8, 15):
        m = idx.match(fingerprint.compute(probe(deg, start)), kind='music')
        hit = m is not None and m.key == s
        ok2 += hit; tot2 += 1
        print(f"  {s:<32} @{start:>2}s -> {'HIT ' if hit else 'MISS'} "
              f"{f'votes={m.votes:4d} score={m.score:.3f} margin={m.margin:4.1f}' if m else ''}")
print(f"  => {ok2}/{tot2} correct")

print("\n--- NEGATIVE: a song that is NOT in the index (must not match)")
fp = 0
for start in (2, 8, 15):
    m = idx.match(fingerprint.compute(probe(pcm[HELDOUT], start)), kind='music')
    print(f"  {HELDOUT:<32} @{start:>2}s -> {'FALSE POSITIVE '+m.key if m else 'correctly unmatched'}"
          + (f" votes={m.votes} score={m.score:.3f}" if m else ""))
    fp += m is not None
print(f"  => {fp} false positives")

print("\n--- NEGATIVE: pure noise (must not match)")
rng = np.random.default_rng(1)
m = idx.match(fingerprint.compute(rng.normal(0,0.1,80000).astype('float32')), kind='music')
print("  noise ->", f"FALSE POSITIVE {m.key} votes={m.votes}" if m else "correctly unmatched")
