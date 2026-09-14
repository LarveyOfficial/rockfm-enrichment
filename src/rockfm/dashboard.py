"""A private dashboard for auditing what the pipeline decided.

Shows the recorded buffer as a timeline, what the analyzer made of each stretch,
and -- the point of it -- lets any of it be played back immediately rather than
waiting out the six-hour delay. Playing a song and watching the playhead cross
its block is how you check whether the boundaries are actually right.
"""

from __future__ import annotations

DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RockFM · pipeline</title>
<style>
:root{
  color-scheme:dark;
  --bg:#0b0b0e; --panel:#141419; --line:#26262e; --fg:#f2f2f5; --dim:#8a8a95;
  --cancion:#3d7dd8; --programa:#2f9e8f;
  --desconocido:#3f3f46;
  --ok:#3fb950; --warn:#d29922; --bad:#f85149;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:20px}
h1{font-size:1.05rem;margin:0;font-weight:600}
h1 small{color:var(--dim);font-weight:400;margin-left:10px;font-size:.8rem}
header{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin-bottom:14px}

.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  padding:7px 11px;font-size:.78rem;display:flex;gap:7px;align-items:center}
.chip b{font-weight:600;font-variant-numeric:tabular-nums}
.chip span{color:var(--dim)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--ok);flex:none}
.dot.warn{background:var(--warn)} .dot.bad{background:var(--bad)}

.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:14px;margin-bottom:16px}
.panel h2{font-size:.76rem;text-transform:uppercase;letter-spacing:.09em;
  color:var(--dim);margin:0 0 10px;font-weight:600}

.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
button{background:#1e1e26;color:var(--fg);border:1px solid var(--line);border-radius:7px;
  padding:6px 12px;font-size:.78rem;cursor:pointer;font-family:inherit}
button:hover{background:#26262f}
button.on{background:var(--cancion);border-color:var(--cancion);color:#fff}
button.danger{border-color:#7a3a30;color:#e08a75}
button.danger:hover{background:#2e1f1c;border-color:#a04a3c}
button:disabled{opacity:.4;cursor:default}

.track{position:relative;height:56px;background:#0e0e12;border:1px solid var(--line);
  border-radius:7px;overflow:hidden;cursor:crosshair}
.blk{position:absolute;top:0;bottom:0;border-right:1px solid rgba(0,0,0,.55);
  overflow:hidden;padding:5px 6px;font-size:.66rem;line-height:1.25;white-space:nowrap;
  text-overflow:ellipsis;color:#fff;opacity:.92}
.blk:hover{opacity:1;filter:brightness(1.25)}
.blk.sel{outline:2px solid #fff;outline-offset:-2px;z-index:3}
.blk small{display:block;opacity:.75;font-size:.62rem}
.gap{position:absolute;top:0;bottom:0;background:repeating-linear-gradient(45deg,
  var(--bad),var(--bad) 3px,transparent 3px,transparent 6px);z-index:4;min-width:2px}
.head{position:absolute;top:0;bottom:0;width:2px;background:#fff;z-index:5;
  box-shadow:0 0 6px #fff;pointer-events:none;display:none}
.ruler{position:relative;height:18px;margin-top:4px;color:var(--dim);font-size:.66rem}
.ruler span{position:absolute;transform:translateX(-50%);white-space:nowrap;
  font-variant-numeric:tabular-nums}

.legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;font-size:.7rem;color:var(--dim)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}

.detail{display:grid;grid-template-columns:132px 1fr;gap:16px}
.detail img{width:132px;height:132px;object-fit:cover;border-radius:8px;background:#0e0e12}
.detail dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:3px 14px;
  font-size:.8rem;align-content:start}
.detail dt{color:var(--dim)} .detail dd{margin:0;font-variant-numeric:tabular-nums}
.detail .now{font-size:1.02rem;font-weight:600;margin:0 0 2px}
.detail .sub{color:var(--dim);margin:0 0 10px;font-size:.83rem}
.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.note{margin-top:10px;font-size:.76rem;color:var(--warn)}

table{width:100%;border-collapse:collapse;font-size:.76rem}
th{text-align:left;color:var(--dim);font-weight:600;padding:5px 8px;
  border-bottom:1px solid var(--line);text-transform:uppercase;
  font-size:.66rem;letter-spacing:.07em}
td{padding:5px 8px;border-bottom:1px solid #1c1c22;font-variant-numeric:tabular-nums}
tr{cursor:pointer} tr:hover td{background:#1a1a21}
tr.sel td{background:#1f2733}
.tag{display:inline-block;padding:1px 7px;border-radius:999px;font-size:.64rem;color:#fff}
.muted{color:var(--dim)}
.empty{color:var(--dim);padding:18px;text-align:center;font-size:.82rem}
.look{display:grid;grid-template-columns:110px 1fr 1fr 1.4fr 44px;gap:8px 10px;align-items:center}
.look .hdr{color:var(--dim);font-size:.66rem;text-transform:uppercase;letter-spacing:.07em}
.look input{background:#0e0e12;border:1px solid var(--line);border-radius:6px;color:var(--fg);
  padding:6px 9px;font:inherit;font-size:.78rem;width:100%}
.look input:focus{outline:1px solid var(--cancion);border-color:var(--cancion)}
.look .thumb{width:40px;height:40px;border-radius:5px;object-fit:cover;background:#0e0e12}
.saved{color:var(--ok);font-size:.76rem;margin-left:10px}
.cfg{display:grid;grid-template-columns:190px 1fr;gap:8px 12px;align-items:center;max-width:720px}
.cfg label{color:var(--dim);font-size:.78rem}
.cfg input[type=text]{background:#0e0e12;border:1px solid var(--line);border-radius:6px;
  color:var(--fg);padding:6px 9px;font:inherit;font-size:.78rem;width:100%}
.cfg input[type=text]:focus{outline:1px solid var(--cancion);border-color:var(--cancion)}
.cfg .toggle{display:flex;align-items:center;gap:8px;font-size:.8rem}
.cfg .group{grid-column:1/-1;color:var(--dim);font-size:.66rem;text-transform:uppercase;
  letter-spacing:.08em;margin-top:10px;border-top:1px solid var(--line);padding-top:10px}
</style></head><body>
<div class="wrap">
  <header>
    <h1>RockFM pipeline <small id="srcclock"></small></h1>
    <div><button id="auto" class="on">auto-refresh</button></div>
  </header>

  <div class="chips" id="chips"></div>

  <div class="panel">
    <h2>Buffer timeline</h2>
    <div class="controls">
      <span class="muted" style="font-size:.76rem">window</span>
      <button data-h="0.5">30 min</button>
      <button data-h="2" class="on">2 h</button>
      <button data-h="6">6 h</button>
      <button data-h="24">24 h</button>
      <span style="flex:1"></span>
      <button id="reanalyze" title="Re-run the analyzer over everything still buffered. Keeps the fingerprint index.">re-analyze buffer</button>
      <button id="reset" class="danger" title="Throw away the timeline and every reference learned from air, then re-seed. Keeps the recorded audio, the catalogue and your settings.">reset analysis</button>
      <button id="stop" disabled>stop audio</button>
    </div>
    <div class="track" id="track"><div class="head" id="head"></div></div>
    <div class="ruler" id="ruler"></div>
    <div class="legend" id="legend"></div>
  </div>

  <div class="panel">
    <h2>Selected</h2>
    <div id="detail"><div class="empty">Click anything on the timeline to inspect and play it.</div></div>
  </div>

  <div class="panel">
    <h2>Settings</h2>
    <p class="muted" style="margin:-4px 0 12px;font-size:.76rem">
      Saved straight to the database and picked up within a few seconds &mdash; no
      restart, so changing something never costs a hole in the recording.
      Secrets are never sent back to this page; leave one blank to keep it.
    </p>
    <div id="settings"><div class="empty">loading&hellip;</div></div>
  </div>

  <div class="panel">
    <h2>Appearance</h2>
    <p class="muted" style="margin:-4px 0 12px;font-size:.76rem">
      What players show during advert breaks. Songs describe themselves, and
      presenter talk already takes the real programme name and artwork from
      RockFM's schedule. Leave a field blank for the built-in default.
    </p>
    <div id="appearance"><div class="empty">loading&hellip;</div></div>
  </div>

  <div class="panel">
    <h2>Items</h2>
    <div style="max-height:340px;overflow:auto">
      <table><thead><tr>
        <th>Madrid</th><th>Dur</th><th>Kind</th><th>What</th><th>Source</th><th>Conf</th>
      </tr></thead><tbody id="rows"></tbody></table>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/hls.js/1.5.17/hls.min.js"></script>
<script>
const KINDS = ['cancion','programa','desconocido'];
const colour = k => getComputedStyle(document.documentElement)
  .getPropertyValue('--' + (KINDS.includes(k) ? k : 'desconocido')).trim();

let hours = 2, auto = true, status = null, items = [], range = null;
let selected = null, replayStart = null, tz = 'Europe/Madrid';

const audio = new Audio();
let hls = null;

const el = id => document.getElementById(id);
const pad = n => String(n).padStart(2, '0');
const clock = ms => new Date(ms).toLocaleTimeString('en-GB',
  {hour12:false, timeZone:tz, hour:'2-digit', minute:'2-digit', second:'2-digit'});
const dur = s => s == null ? '–' : Math.floor(s/60) + ':' + pad(Math.floor(s%60));

function hhmm(seconds) {
  if (seconds == null) return '–';
  const m = Math.round(seconds / 60);
  return m >= 60 ? `${Math.floor(m/60)}h ${m%60}m` : `${m}m`;
}

// --- data ---------------------------------------------------------------

async function load() {
  try {
    const [s, t] = await Promise.all([
      fetch('/api/status', {cache:'no-store'}).then(r => r.json()),
      fetch(`/api/timeline?hours=${hours}`, {cache:'no-store'}).then(r => r.json()),
    ]);
    status = s; items = t.items; range = t.range; tz = s.source_timezone || tz;
    renderChips(); renderTrack(); renderRows();
    el('srcclock').textContent =
      `${clock(s.now)} live · delay ${(s.delay_seconds/3600).toFixed(2)} h`;
  } catch (err) {
    el('chips').innerHTML = '<div class="chip"><i class="dot bad"></i>server unreachable</div>';
  }
}

// --- status chips -------------------------------------------------------

function chip(label, value, level) {
  const dot = level ? `<i class="dot ${level}"></i>` : '';
  return `<div class="chip">${dot}<span>${label}</span><b>${value}</b></div>`;
}

function renderChips() {
  const b = status.buffer, a = status.analyzer, c = status.counts;
  const lag = b.ingest_lag_seconds;
  const lagLevel = lag == null ? 'bad' : lag < 30 ? '' : lag < 120 ? 'warn' : 'bad';
  const behind = a.behind_live_seconds;
  const out = [
    chip('ingest lag', lag == null ? '–' : lag.toFixed(0) + 's', lagLevel),
    chip('buffer', hhmm(b.seconds)),
    chip('analyzer', a.state === 'scanning'
           ? `scanning ${a.progress || ''}`.trim()
         : a.state === 'waiting' ? 'waiting for audio'
         : behind == null ? 'starting…' : 'behind ' + hhmm(behind),
         // Scanning writes a heartbeat every probe, so a stale one now really
         // does mean stuck rather than merely slow.
         a.last_active_seconds != null && a.last_active_seconds > 300 ? 'bad'
         : behind != null && behind > 900 ? 'warn' : ''),
    chip('segments', c.segments),
    chip('songs found', c.songs),
    chip('learned', c.learned_songs),
    chip('gaps', status.gaps.length, status.gaps.length ? 'warn' : ''),
  ];
  if (b.state !== 'ready') {
    const text = b.state === 'filling'
      ? `filling · ready in ${hhmm(b.seconds_until_ready)}`
      : b.state === 'gap'
        ? `recording gap · resumes in ${hhmm(b.seconds_until_ready)}`
        : b.state;
    out.unshift(chip('playout', text, b.state === 'gap' ? 'bad' : 'warn'));
  }
  el('chips').innerHTML = out.join('');
}

// --- timeline -----------------------------------------------------------

function renderTrack() {
  const track = el('track');
  [...track.querySelectorAll('.blk,.gap')].forEach(n => n.remove());
  const span = range.end - range.start;
  if (span <= 0) return;
  const pct = ms => ((ms - range.start) / span) * 100;

  for (const it of items) {
    const left = Math.max(0, pct(it.start));
    const width = Math.min(100 - left, pct(it.end) - pct(it.start));
    if (width <= 0) continue;
    const node = document.createElement('div');
    node.className = 'blk' + (selected && selected.start === it.start ? ' sel' : '');
    node.style.cssText = `left:${left}%;width:${width}%;background:${colour(it.kind)}`;
    node.title = `${clock(it.start)} – ${clock(it.end)} (${dur(it.duration)})\\n${it.primary}`
      + (it.secondary ? `\\n${it.secondary}` : '') + `\\nvia ${it.source}`;
    if (width > 2.5) {
      node.innerHTML = `${it.primary}<small>${clock(it.start)} · ${dur(it.duration)}</small>`;
    }
    node.onclick = () => select(it);
    track.appendChild(node);
  }

  for (const g of status.gaps) {
    if (g.before < range.start || g.after > range.end) continue;
    const node = document.createElement('div');
    const left = Math.max(0, pct(g.after));
    node.className = 'gap';
    node.style.cssText = `left:${left}%;width:${Math.max(0.2, pct(g.before) - left)}%`;
    node.title = `gap: ${g.missing_seconds.toFixed(0)}s of audio missing`;
    track.appendChild(node);
  }

  const ruler = el('ruler');
  ruler.innerHTML = '';
  for (let i = 0; i <= 6; i++) {
    const mark = document.createElement('span');
    mark.style.left = (i / 6 * 100) + '%';
    mark.textContent = clock(range.start + span * i / 6);
    ruler.appendChild(mark);
  }

  el('legend').innerHTML = KINDS.map(k =>
    `<span><i style="background:${colour(k)}"></i>${k}</span>`).join('')
    + '<span><i style="background:var(--bad)"></i>gap</span>';
}

// --- item list ----------------------------------------------------------

function renderRows() {
  const body = el('rows');
  if (!items.length) {
    body.innerHTML = '<tr><td colspan="6" class="empty">Nothing analyzed in this window yet.</td></tr>';
    return;
  }
  body.innerHTML = items.slice().reverse().map(it => `
    <tr data-start="${it.start}" class="${selected && selected.start === it.start ? 'sel' : ''}">
      <td>${clock(it.start)}</td>
      <td>${dur(it.duration)}</td>
      <td><span class="tag" style="background:${colour(it.kind)}">${it.kind}</span></td>
      <td>${escape(it.primary)}${it.secondary ? ` <span class="muted">· ${escape(it.secondary)}</span>` : ''}</td>
      <td class="muted">${it.source || '–'}</td>
      <td class="muted">${it.confidence == null ? '–' : it.confidence.toFixed(2)}</td>
    </tr>`).join('');
  [...body.querySelectorAll('tr[data-start]')].forEach(tr => {
    tr.onclick = () => select(items.find(i => i.start === Number(tr.dataset.start)));
  });
}

function escape(text) {
  const d = document.createElement('div');
  d.textContent = text == null ? '' : text;
  return d.innerHTML;
}

// --- detail + playback --------------------------------------------------

function select(it) {
  if (!it) return;
  selected = it;
  const art = it.art
    ? `<img src="${it.art}" alt="">`
    : `<div style="width:132px;height:132px;border-radius:8px;background:#0e0e12"></div>`;
  el('detail').innerHTML = `
    <div class="detail">
      ${art}
      <div>
        <p class="now">${escape(it.primary)}</p>
        <p class="sub">${escape(it.secondary || '')}</p>
        <dl>
          <dt>kind</dt><dd><span class="tag" style="background:${colour(it.kind)}">${it.kind}</span></dd>
          <dt>starts</dt><dd>${clock(it.start)}</dd>
          <dt>ends</dt><dd>${clock(it.end)}</dd>
          <dt>duration</dt><dd>${dur(it.duration)}</dd>
          ${it.album ? `<dt>album</dt><dd>${escape(it.album)}${it.year ? ', ' + it.year : ''}</dd>` : ''}
          ${it.show ? `<dt>programme</dt><dd>${escape(it.show)}</dd>` : ''}
          <dt>identified by</dt><dd>${it.source || '–'}</dd>
          <dt>confidence</dt><dd>${it.confidence == null ? '–' : it.confidence.toFixed(2)}</dd>
        </dl>
        <div class="actions">
          <button onclick="playFrom(${it.start}, ${Math.ceil(it.duration) + 20})">play from start</button>
          <button onclick="playFrom(${it.start} - 15000, 45)">hear the start boundary</button>
          <button onclick="playFrom(${it.end} - 15000, 45)">hear the end boundary</button>
        </div>
        <div id="note" class="note" style="display:none"></div>
        <div id="progress" class="muted" style="margin-top:8px;font-size:.76rem"></div>
      </div>
    </div>`;
  renderTrack(); renderRows();
}

function playFrom(startMs, seconds) {
  stopAudio();
  replayStart = startMs;
  const url = `/replay.m3u8?start=${Math.round(startMs)}&duration=${Math.round(seconds)}`;
  if (window.Hls && Hls.isSupported()) {
    hls = new Hls();
    hls.loadSource(url);
    hls.attachMedia(audio);
  } else {
    audio.src = url;
  }
  // play() must be called straight from the click. Deferring it to hls.js's
  // MANIFEST_PARSED callback loses the user-gesture context that autoplay
  // policy requires, and playback silently never starts even though every
  // segment downloads.
  audio.play().then(() => setNote('')).catch(() => {
    setNote('Your browser blocked playback — press play again.');
  });
  el('stop').disabled = false;
  el('head').style.display = 'block';
}

function setNote(text) {
  const note = el('note');
  if (note) { note.textContent = text; note.style.display = text ? 'block' : 'none'; }
}

function stopAudio() {
  const progress = el('progress');
  if (progress) progress.textContent = '';
  audio.pause();
  if (hls) { hls.destroy(); hls = null; }
  audio.removeAttribute('src');
  el('stop').disabled = true;
  el('head').style.display = 'none';
  replayStart = null;
}

// The playhead is the whole point: watching it cross a boundary is how you
// tell whether the analyzer put the boundary in the right place.
audio.addEventListener('timeupdate', () => {
  if (replayStart == null || !range) return;
  const at = replayStart + audio.currentTime * 1000;
  const progress = el('progress');
  if (progress) progress.textContent = `playing · ${clock(at)} (+${dur(audio.currentTime)})`;
  const span = range.end - range.start;
  const head = el('head');
  if (at < range.start || at > range.end || span <= 0) { head.style.display = 'none'; return; }
  head.style.display = 'block';
  head.style.left = ((at - range.start) / span * 100) + '%';
});
audio.addEventListener('ended', stopAudio);

// --- settings -----------------------------------------------------------

const SETTING_LABELS = {
  public_url: 'Public URL',
  display_language: 'Display language',
  min_nonmusic_seconds: 'Min non-music seconds',
  max_seam_seconds: 'Max crossfade seam seconds',
  azuracast_enabled: 'Enable AzuraCast',
  azuracast_base_url: 'AzuraCast base URL',
  azuracast_station_id: 'Station ID',
  azuracast_api_key: 'API key',
  azuracast_dj_url: 'DJ URL',
  azuracast_dj_password: 'DJ password',
};
const AZURACAST_FIELDS = ['azuracast_enabled', 'azuracast_base_url', 'azuracast_station_id',
                          'azuracast_api_key', 'azuracast_dj_url', 'azuracast_dj_password'];

async function loadSettings() {
  const data = await (await fetch('/api/settings', {cache:'no-store'})).json();
  const row = name => {
    const spec = data.fields[name], value = data.current[name];
    const label = `<label for="set-${name}">${SETTING_LABELS[name] || name}</label>`;
    if (spec.type === 'bool') {
      return label + `<span class="toggle"><input type="checkbox" id="set-${name}"
        data-name="${name}" ${value ? 'checked' : ''}></span>`;
    }
    const placeholder = spec.type === 'secret' ? 'unchanged' : String(spec.default ?? '');
    return label + `<input type="text" id="set-${name}" data-name="${name}"
      value="${escape(spec.type === 'secret' ? '' : value)}"
      placeholder="${escape(placeholder)}">`;
  };
  const general = Object.keys(data.fields).filter(n => !AZURACAST_FIELDS.includes(n));
  el('settings').innerHTML = `
    <div class="cfg">
      ${general.map(row).join('')}
      <div class="group">AzuraCast</div>
      ${AZURACAST_FIELDS.map(row).join('')}
    </div>
    <div class="actions">
      <button id="save-settings">save</button>
      <span id="settings-note" class="saved"></span>
    </div>`;
  el('save-settings').onclick = saveSettings;
}

async function saveSettings() {
  const payload = {};
  document.querySelectorAll('#settings [data-name]').forEach(input => {
    payload[input.dataset.name] = input.type === 'checkbox' ? input.checked : input.value;
  });
  const btn = el('save-settings');
  btn.disabled = true;
  try {
    const res = await fetch('/api/settings', {
      method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(res.status);
    el('settings-note').textContent = 'saved';
    setTimeout(() => { el('settings-note').textContent = ''; }, 2500);
    loadSettings();
  } catch (err) {
    el('settings-note').textContent = 'could not save';
  }
  btn.disabled = false;
}

// --- appearance ---------------------------------------------------------

let look = null;

async function loadAppearance() {
  look = await (await fetch('/api/appearance', {cache:'no-store'})).json();
  const rows = look.configurable.map(kind => {
    const cur = look.current[kind] || {}, def = look.defaults[kind] || {};
    const field = (name, placeholder) =>
      `<input data-kind="${kind}" data-field="${name}" value="${escape(cur[name] || '')}"
              placeholder="${escape(placeholder)}">`;
    return `
      <span><span class="tag" style="background:${colour(kind)}">${kind}</span></span>
      ${field('title', def.title || 'programme name')}
      ${field('artist', def.artist || 'presenters')}
      ${field('art', 'artwork URL')}
      <img class="thumb" data-thumb="${kind}" src="${escape(cur.art || '')}" alt="">`;
  }).join('');
  el('appearance').innerHTML = `
    <div class="look">
      <span class="hdr">kind</span><span class="hdr">title</span>
      <span class="hdr">artist</span><span class="hdr">artwork URL</span><span></span>
      ${rows}
    </div>
    <div class="actions">
      <button id="save-look">save</button>
      <span id="saved-note" class="saved"></span>
    </div>`;
  el('save-look').onclick = saveAppearance;
  document.querySelectorAll('[data-field="art"]').forEach(input => {
    input.oninput = () => {
      const thumb = document.querySelector(`[data-thumb="${input.dataset.kind}"]`);
      if (thumb) thumb.src = input.value;
    };
  });
}

async function saveAppearance() {
  const payload = {};
  document.querySelectorAll('#appearance input').forEach(input => {
    (payload[input.dataset.kind] ||= {})[input.dataset.field] = input.value;
  });
  const btn = el('save-look');
  btn.disabled = true;
  try {
    const res = await fetch('/api/appearance', {
      method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(res.status);
    look.current = (await res.json()).current;
    el('saved-note').textContent = 'saved';
    setTimeout(() => { el('saved-note').textContent = ''; }, 2500);
    load();
  } catch (err) {
    el('saved-note').textContent = 'could not save';
  }
  btn.disabled = false;
}

// --- controls -----------------------------------------------------------

[...document.querySelectorAll('[data-h]')].forEach(btn => {
  btn.onclick = () => {
    hours = Number(btn.dataset.h);
    [...document.querySelectorAll('[data-h]')].forEach(b => b.classList.toggle('on', b === btn));
    load();
  };
});
el('auto').onclick = () => {
  auto = !auto;
  el('auto').classList.toggle('on', auto);
};
el('stop').onclick = stopAudio;
el('reset').onclick = async () => {
  const ok = confirm(
`Throw away the timeline and everything learned from the air, then re-seed?

Kept: the recorded audio, the catalogue references, your settings.
Dropped: every song and gap on the timeline, and any reference a broadcast overwrote.

The analyzer restarts from the beginning of the buffer. Nothing stops playing.`
  );
  if (!ok) return;
  const btn = el('reset');
  btn.disabled = true;
  const previous = btn.textContent;
  try {
    const res = await fetch('/api/reset', {method: 'POST'});
    const body = res.ok ? await res.json() : null;
    btn.textContent = body
      ? `dropped ${body.dropped_references}, kept ${body.kept_references}`
      : 'failed';
  } catch (err) {
    btn.textContent = 'failed';
  }
  setTimeout(() => { btn.textContent = previous; btn.disabled = false; }, 6000);
  load();
};
el('reanalyze').onclick = async () => {
  const btn = el('reanalyze');
  btn.disabled = true;
  const previous = btn.textContent;
  try {
    const res = await fetch('/api/reanalyze', {method: 'POST'});
    btn.textContent = res.ok ? 're-analyzing…' : 'failed';
  } catch (err) {
    btn.textContent = 'failed';
  }
  setTimeout(() => { btn.textContent = previous; btn.disabled = false; }, 4000);
  load();
};

load();
loadSettings();
loadAppearance();
setInterval(() => { if (auto) load(); }, 5000);
</script></body></html>
"""
