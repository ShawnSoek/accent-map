"""The web page: logic.js runs in QuickJS, every script compiles, HTML and JS agree, the server serves it.

There is no browser here; the microphone and the look of the page are checked by hand (README).
"""

import json
import re

import pytest
import quickjs
from fastapi.testclient import TestClient

from fakes import fake_config
from server.app import WEB_DIR, Engine, create_app
from server.model_api import load_model
from server.session import SessionSettings

SCRIPTS = ["logic.js", "map.js", "app.js", "mic-worklet.js"]


@pytest.fixture(scope="module")
def js():
    ctx = quickjs.Context()
    ctx.eval((WEB_DIR / "logic.js").read_text())

    def call(expression):
        return json.loads(ctx.eval(f"JSON.stringify({expression})"))
    return call


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_compiles(name):
    # Wrapped in a function that is never called: QuickJS parses it without running browser code.
    quickjs.Context().eval("(function () {\n" + (WEB_DIR / name).read_text() + "\n})")


def test_projection_maps_box_corners(js):
    out = js("(function(){ var p = AccentLogic.projection([-10, 30, 10, 50]);"
             " return [p.width, p.height, p.point(-10, 50), p.point(10, 30)]; })()")
    k = 0.766044  # cos(40 degrees)
    assert out[0] == pytest.approx(20 * k, abs=1e-5) and out[1] == 20
    assert out[2] == [0, 0]
    assert out[3] == [pytest.approx(20 * k, abs=1e-5), 20]


def test_geometry_path_handles_holes_and_multipolygons(js):
    square = [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]
    hole = [[1, 1], [2, 1], [2, 2], [1, 1]]
    geometry = {"type": "MultiPolygon", "coordinates": [[square, hole], [square]]}
    d = js(f"AccentLogic.geometryPath({json.dumps(geometry)}, AccentLogic.projection([0, -10, 10, 10]))")
    assert d.count("M") == 3 and d.count("Z") == 3
    assert d.startswith("M0.00 10.00L")


def test_zone_probabilities_add_classes_sharing_a_zone(js):
    classes = [{"id": "a", "zone": "z1"}, {"id": "b", "zone": "z1"}, {"id": "c", "zone": "z2"}]
    assert js(f"AccentLogic.zoneProbabilities({json.dumps(classes)}, {{a: 0.2, b: 0.3, c: 0.5}})") == \
        {"z1": 0.5, "z2": 0.5}
    assert js(f"AccentLogic.zoneProbabilities({json.dumps(classes)}, null)") == {"z1": 0, "z2": 0}


def test_zone_opacity(js):
    values = js("[0, 0.05, 0.2, 0.5, 0.9, 1].map(function (p) { return AccentLogic.zoneOpacity(p, 'confident'); })")
    assert values == sorted(values) and values[0] > 0 and values[-1] == 1
    assert js("AccentLogic.zoneOpacity(0.9, 'listening')") < js("AccentLogic.zoneOpacity(0.9, 'uncertain')")


@pytest.mark.parametrize("time, expected", [(0.5, None), (1.0, 1.0), (1.7, 1.5), (3.0, 3.0), (99, 3.0)])
def test_event_at_follows_playback_time(js, time, expected):
    events = [{"t": t} for t in (1.0, 1.5, 2.0, 2.5, 3.0)]
    found = js(f"AccentLogic.eventAt({json.dumps(events)}, {time})")
    assert (found and found["t"]) == expected


def test_event_at_with_no_events(js):
    assert js("AccentLogic.eventAt([], 5)") is None


def test_insert_event_keeps_order_and_skips_duplicates(js):
    got = js("""(function () {
        var ev = [], added = [];
        [{index: 5, t: 6.0}, {index: 0, t: 1.0}, {index: 3, t: 4.0}, {index: 3, t: 4.0}, {index: 9, t: 10.0}]
          .forEach(function (e) { added.push(AccentLogic.insertEvent(ev, e)); });
        return [ev.map(function (e) { return e.index; }), added];
    })()""")
    assert got == [[0, 3, 5, 9], [True, True, True, False, True]]


def test_fresh_event_ignores_results_from_skipped_parts(js):
    events = json.dumps([{"index": 0, "t": 2.0}, {"index": 1, "t": 4.0}])
    assert js(f"AccentLogic.freshEventAt({events}, 4.5, 2.0)")["t"] == 4.0
    assert js(f"AccentLogic.freshEventAt({events}, 6.2, 2.0)")["t"] == 4.0      # within one update + 0.3 s
    assert js(f"AccentLogic.freshEventAt({events}, 30, 2.0)") is None           # skipped far ahead
    assert js(f"AccentLogic.freshEventAt({events}, 1.0, 2.0)") is None          # before the first window


FILE = {"first_t": 2.0, "hop": 2.0, "duration": 200.0}


def a_pass(**changes):
    out = {"start": 0, "origin": 0, "paused": False, "due": 0, "got": True, "first": 2, "position": 2,
           "lastAt": 0, "done": False}
    return {**out, **changes}


def a_view(time, playing=True, now=10.0, infer=1.8, seeked=False):
    return {"time": time, "playing": playing, "now": now, "infer": infer, "seeked": seeked}


def windows(*ts):
    return [{"index": round(t / 2) - 1, "t": t} for t in ts]


@pytest.mark.parametrize("events, stream, view, expected", [
    (windows(2), a_pass(), a_view(3.0), False),                                   # analyzed
    ([], None, a_view(10.0), True),                                               # nothing asked yet
    (windows(60, 62), a_pass(origin=60, start=64, first=60, position=62), a_view(30.0, seeked=True), True),
    # waiting for the first result of a pass asked ahead: playback drifting past it is no reason
    (windows(2), a_pass(origin=40, start=45.6, got=False, first=None, position=45.6), a_view(48.5), False),
    (windows(2), a_pass(origin=40, start=45.6, got=False, first=None, position=45.6),
     a_view(60.0, seeked=True), True),                                            # but a skip is
    # asked while paused, then play: replace it only if its first result would come late
    (windows(2), a_pass(origin=40, start=40, paused=True, due=13.0, got=False, first=None, position=40),
     a_view(40.0, seeked=True), True),
    ([], a_pass(origin=0, start=0, paused=True, due=11.2, got=False, first=None, position=0),
     a_view(0.5, seeked=True), False),                                            # file start, short window
    (windows(48, 50), a_pass(origin=40, start=46, first=46, position=50, lastAt=9.0), a_view(52.5), False),
    (windows(48, 50), a_pass(origin=40, start=46, first=46, position=50, lastAt=9.0), a_view(60.0), True),
    (windows(48, 50), a_pass(origin=40, start=46, first=46, position=50), a_view(60.0, playing=False), True),
    (windows(46, 48), a_pass(origin=40, start=46, first=46, position=200, done=True), a_view(44.0), False),
    (windows(2, 4, 6), a_pass(position=6, done=True), a_view(20.0), True),       # pass ended early
    # Playing an analyzed stretch (54..64): look ahead to the window after it (66).
    (windows(*range(54, 65, 2)), a_pass(position=64, lastAt=9.5), a_view(60.0), False),   # next one comes
    (windows(*range(54, 65, 2)), a_pass(position=34, lastAt=9.5), a_view(60.0), True),    # pass far behind
    (windows(*range(54, 65, 2)), a_pass(position=34, lastAt=9.5), a_view(60.0, playing=False), False),
    (windows(*range(54, 65, 2)), a_pass(position=80, first=70), a_view(60.0), True),       # pass started later
    (windows(*range(54, 65, 2)), a_pass(origin=40, first=70, position=200, done=True), a_view(60.0), True),
    (windows(*range(54, 65, 2)), a_pass(origin=40, first=54, position=64, done=True), a_view(60.0), True),
    # A pass that left a gap because it could not be ready in time: a new one would only be later.
    (windows(26, 28, 30, 34), a_pass(origin=30, start=34.3, first=34, position=34, lastAt=10.0),
     a_view(32.14), False),
    (windows(196, 198, 200), a_pass(position=34), a_view(199.0), False),          # analyzed to the end
])
def test_needs_restart(js, events, stream, view, expected):
    got = js(f"AccentLogic.needsRestart({json.dumps(events)}, {json.dumps(FILE)}, {json.dumps(stream)},"
             f" {json.dumps(view)})")
    assert got is expected


def test_analyzed_until_follows_the_unbroken_run(js):
    events = json.dumps(windows(2, 4, 6, 10, 12) + [{"index": 99, "t": 13.5}])
    assert [js(f"AccentLogic.analyzedUntil({events}, {t}, 2.0)") for t in (1.0, 3.0, 8.5, 9.0, 11.0)] == \
        [None, 6, None, None, 13.5]


def test_new_pass_starts_ahead_while_playing(js):
    def new(previous, view, events=()):
        return js(f"AccentLogic.newPass({json.dumps(FILE)}, {json.dumps(previous)}, {json.dumps(view)},"
                  f" {json.dumps(list(events))})")
    paused = new(None, a_view(40.0, playing=False))
    assert paused["start"] == 40.0 and paused["paused"] is True and paused["due"] == pytest.approx(10 + 1.95)
    assert new(None, a_view(40.0))["start"] == pytest.approx(40 + 1.95 + 1.99)   # server idle: one inference
    busy = new(a_pass(lastAt=9.5), a_view(40.0))                                 # 1.3 s left of the running one
    assert busy["start"] == pytest.approx(40 + 1.3 + 1.8 + 0.15 + 1.99) and busy["paused"] is False
    assert new(a_pass(got=False), a_view(40.0))["due"] == pytest.approx(10 + 1.8 + 1.8 + 0.15)
    assert new(None, a_view(199.0))["start"] == 200.0                             # never past the end
    # In an analyzed stretch: right after it when the first result can be there in time ...
    assert new(None, a_view(60.0), windows(*range(54, 65, 2)))["start"] == pytest.approx(66.01)
    # ... else one update past when it can arrive (62.5 + 1.95 > 64).
    assert new(None, a_view(62.5), windows(*range(54, 63, 2)))["start"] == pytest.approx(62.5 + 1.95 + 1.99)


def test_note_inference_follows_the_longest_window(js):
    got = js("""(function () {
        var est = null, out = [];
        [[2, 400], [10, 1800], [4, 700], [10, null], [10, 1700]].forEach(function (w) {
            est = AccentLogic.noteInference(est, {window_seconds: w[0], infer_ms: w[1]});
            out.push(est.seconds);
        });
        return out;
    })()""")
    assert got == [0.4, 1.8, 1.8, 1.8, 1.7]


# Playback against a simulated server: one inference at a time, `infer` seconds each, and an
# inference that has started always finishes (its result is dropped if its pass was abandoned).
# The page part mirrors web/app.js: check on seek and play, and every 0.25 s while playing.
SIMULATE = r"""
function simulate(hop, infer, duration, script, until) {
  var L = AccentLogic, dt = 0.01;
  var ends = [];
  for (var k = 1; k * hop <= duration + 1e-9; k++) ends.push(Math.round(k * hop * 1000) / 1000);
  if (ends[ends.length - 1] < duration - 1e-9) ends.push(duration);
  var file = {first_t: ends[0], hop: hop, duration: duration};
  var server = {job: null, pass: null};
  var page = {events: [], stream: null, inference: null, time: 0, playing: false, lastCheck: -1, restarts: 0};
  var now = 0, log = [];
  function startIndex(t) { var i = 0; while (i + 1 < ends.length && ends[i + 1] <= t + 1e-9) i++; return i; }
  function view(seeked) {
    return {time: page.time, playing: page.playing, now: now, seeked: seeked,
            infer: page.inference ? page.inference.seconds : hop / 1.2};
  }
  function startStream() {
    var p = L.newPass(file, page.stream, view(false), page.events);
    if (server.pass) server.pass.aborted = true;
    server.pass = {next: startIndex(p.start), aborted: false, page: p};
    page.stream = p;
    page.restarts++;
  }
  function check(seeked) {
    if (L.needsRestart(page.events, file, page.stream, view(seeked))) startStream();
  }
  for (var step = 0; now < until; step++) {
    now = Math.round(step * dt * 1000) / 1000;
    while (script.length && script[0][0] <= now + 1e-9) {
      var a = script.shift();
      if (a[1] === "upload") startStream();
      else if (a[1] === "seek") { page.time = a[2]; check(true); }
      else if (a[1] === "play") { page.playing = true; check(true); }
      else if (a[1] === "pause") page.playing = false;
    }
    if (server.job && now >= server.job.until - 1e-9) {
      var j = server.job;
      server.job = null;
      if (!j.pass.aborted) {
        var e = {index: j.index, t: ends[j.index], infer_ms: infer * 1000, window_seconds: Math.min(10, ends[j.index])};
        L.insertEvent(page.events, e);
        L.passResult(j.pass.page, e, now);
        page.inference = L.noteInference(page.inference, e);
        if (j.pass.next >= ends.length) j.pass.page.done = true;
      }
    }
    if (!server.job && server.pass && !server.pass.aborted && server.pass.next < ends.length) {
      server.job = {pass: server.pass, index: server.pass.next++, until: now + infer};
    }
    if (page.playing && now - page.lastCheck >= 0.25 - 1e-9) { page.lastCheck = now; check(false); }
    var shown = L.freshEventAt(page.events, page.time, hop);
    log.push([now, page.time, shown ? shown.t : null, page.restarts]);
    if (page.playing) page.time = Math.min(page.time + dt, duration);
  }
  return log;
}
"""

PLAYBACK = [
    (0.0, "upload"), (0.3, "play"),
    (20.0, "seek", 60.0),          # skip ahead while playing
    (35.0, "seek", 30.0),          # back into a part not analyzed yet
    (50.0, "seek", 10.0),          # back into an analyzed part
    (58.0, "seek", 70.0),          # ahead into a part the skip at 20 s analyzed (to ~80 s), play past its end
    (72.0, "pause"), (73.0, "seek", 120.0), (80.0, "play"),     # skip while paused, wait, play
    (92.0, "pause"), (92.5, "seek", 160.0), (92.6, "play"),     # skip while paused, play at once
]


@pytest.mark.parametrize("hop, infer", [(2.0, 1.8), (2.25, 1.8), (0.5, 0.35)])
def test_map_follows_playback_after_skips(hop, infer):
    ctx = quickjs.Context()
    ctx.eval((WEB_DIR / "logic.js").read_text())
    ctx.eval(SIMULATE)
    log = json.loads(ctx.eval(f"JSON.stringify(simulate({hop}, {infer}, 200, {json.dumps(PLAYBACK)}, 110))"))

    def between(a, b):
        return [row for row in log if a <= row[0] < b - 1e-9]

    times = [a[0] for a in PLAYBACK] + [110.0]
    worst_wait = 2 * infer + 0.15 + hop + 0.05          # running inference + one more + one update
    for (at, action, *_), next_at in zip(PLAYBACK, times[1:]):
        if action not in ("seek", "play") or (action == "seek" and at in (73.0, 92.5)):
            continue
        rows = between(at, next_at)
        first = next(i for i, row in enumerate(rows) if row[2] is not None)
        wait = rows[first][0] - at
        assert wait <= worst_wait, (action, at, wait)
        # From then on the map shows a result for the moment being heard, without gaps.
        assert all(row[2] is not None and row[1] - row[2] <= hop + 0.3 + 1e-6 for row in rows[first:]), (at, action)
        restarts = rows[-1][3] - between(at - 0.02, at)[0][3]
        assert restarts <= 1 + (at == 92.6), (at, restarts)
    assert between(10.0, 20.0)[0][3] == 1                # the first pass started at the upload, kept
    assert between(50.0, 50.02)[0][2] is not None        # analyzed part: shown at once
    paused = between(73.0, 80.0)                         # the moment skipped to shows while paused
    assert paused[-1][2] == pytest.approx(120.0 - (120.0 % hop))
    assert next(row[0] for row in paused if row[2] is not None) - 73.0 <= 2 * infer + 0.2


def test_clock(js):
    assert js("[0, 5.9, 65, 600].map(AccentLogic.clock)") == ["0:00", "0:05", "1:05", "10:00"]


def test_ndjson_parser_handles_split_lines(js):
    got = js("""(function () {
        var p = AccentLogic.ndjsonParser(), out = [];
        out = out.concat(p.push('{"a":1}\\n{"b"'));
        out = out.concat(p.push(':2}\\n\\n{"c":3}\\n{"d"'));
        out = out.concat(p.push(':4}'));
        return out.concat(p.end());
    })()""")
    assert got == [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]


def test_percent(js):
    assert js("[0.923, 0.004, 0, 1].map(AccentLogic.percent)") == ["92 %", "<1 %", "0 %", "100 %"]


BY_ID = {"mx": {"label": "Mexico"}, "car": {"label": "Caribbean"}, "es": {"label": "Southern Spain"}}
SESSION = {"listen_speech_seconds": 2.0, "top_k": 3}


def describe(js, event):
    return js(f"AccentLogic.describe({json.dumps(event)}, {json.dumps(BY_ID)}, {json.dumps(SESSION)})")


def test_describe_each_status(js):
    top = [{"id": "car", "p": 0.61}, {"id": "mx", "p": 0.19}, {"id": "es", "p": 0.1}]
    assert describe(js, None)["status"] == "idle"
    listening = describe(js, {"status": "listening", "speech_seconds": 1.26, "top": top})
    assert listening["badge"] == "Listening…" and "1.3 of 2.0 s" in listening["text"]
    assert describe(js, {"status": "no_speech", "speech_seconds": 0, "top": []})["text"] == "Waiting for speech…"
    assert "last result" in describe(js, {"status": "no_speech", "speech_seconds": 3, "top": top})["text"]
    assert describe(js, {"status": "uncertain", "speech_seconds": 3, "top": top})["text"] == \
        "No clear winner: Caribbean 61 % vs Mexico 19 %."
    confident = describe(js, {"status": "confident", "speech_seconds": 3, "top": top})
    assert confident["badge"] == "Confident" and confident["text"] == "Most likely Caribbean (61 %)."


FAKE_DOM = """
function FakeEl(tag) {
  var classes = {};
  this.tag = tag; this.attrs = {}; this.children = []; this.style = {}; this.listeners = {};
  this.classList = { toggle: function (c, on) { if (on) classes[c] = true; else delete classes[c]; },
                     contains: function (c) { return !!classes[c]; } };
}
FakeEl.prototype.setAttribute = function (k, v) { this.attrs[k] = String(v); };
FakeEl.prototype.appendChild = function (c) { this.children.push(c); return c; };
FakeEl.prototype.addEventListener = function (t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); };
var document = { createElementNS: function (ns, tag) { return new FakeEl(tag); },
                 createElement: function (tag) { return new FakeEl(tag); } };
function walk(node, out) { out.push(node); node.children.forEach(function (c) { walk(c, out); }); return out; }
"""


@pytest.fixture(scope="module")
def drawn_map():
    """map.js run against a fake DOM with the real stage-1 zones and panels."""
    from conftest import REAL_MODEL_DIR
    from server.config import load_config
    config = load_config(REAL_MODEL_DIR)
    ctx = quickjs.Context()
    ctx.eval(FAKE_DOM)
    ctx.eval((WEB_DIR / "logic.js").read_text())
    ctx.eval((WEB_DIR / "map.js").read_text())
    panels = json.dumps([p.model_dump() for p in config.map.panels])
    zones = (config.model_dir / config.map.zones_file).read_text()
    ctx.eval(f"var container = new FakeEl('div'); var m = AccentMap.create(container, {panels}, {zones}, {{}});"
             "function overlays() { return walk(container, []).filter(function (n) {"
             " return n.attrs['class'] === 'zone-fill'; }); }")
    def call(expression):
        result = ctx.eval(f"JSON.stringify({expression})")
        return None if result is None else json.loads(result)   # undefined -> None
    return call, config


def test_map_draws_every_zone_in_its_panels(drawn_map):
    js, config = drawn_map
    assert js("container.children.map(function (f) { return f.className; })") == ["panel main", "panel inset"]
    zones = js("overlays().map(function (n) { return n.attrs['data-zone']; })")
    assert sorted(set(zones)) == sorted({c.zone for c in config.classes})
    assert js("walk(container, []).filter(function (n) { return n.attrs['class'] === 'uncovered'; }).length") == 2
    assert js("walk(container, []).filter(function (n) { return n.attrs['class'] === 'land'; }).length") == 2


def test_map_update_and_highlight(drawn_map):
    js, _ = drawn_map
    js("m.update({es_south: 0.9, mexico: 0.05}, 'confident')")
    opacity = js("(function () { var o = {}; overlays().forEach(function (n) {"
                 " o[n.attrs['data-zone']] = n.style.fillOpacity; }); return o; })()")
    assert opacity["es_south"] == js("AccentLogic.zoneOpacity(0.9, 'confident')")
    assert opacity["mexico"] == js("AccentLogic.zoneOpacity(0.05, 'confident')")
    assert opacity["chilean"] == js("AccentLogic.zoneOpacity(0, 'confident')")   # missing -> 0
    js("m.update(null, 'idle')")
    assert set(js("overlays().map(function (n) { return n.style.fillOpacity; })")) == {0}
    js("m.highlight('mexico')")
    assert js("overlays().filter(function (n) { return n.classList.contains('hl'); })"
              ".map(function (n) { return n.attrs['data-zone']; })") == ["mexico"]


def test_html_has_every_element_the_scripts_use():
    html = (WEB_DIR / "index.html").read_text()
    ids = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r'\$\("([^"]+)"\)', (WEB_DIR / "app.js").read_text()))
    assert used and used <= ids, f"missing in index.html: {sorted(used - ids)}"
    for script in ("logic.js", "map.js", "app.js"):
        assert f'src="{script}"' in html
    assert html.index('src="logic.js"') < html.index('src="map.js"') < html.index('src="app.js"')
    assert '"mic-worklet.js"' in (WEB_DIR / "app.js").read_text()


def test_server_serves_the_page():
    model = load_model(fake_config(3))
    client = TestClient(create_app(engine=Engine(model=model, settings=SessionSettings())))
    page = client.get("/")
    assert page.status_code == 200 and "<title>Accent Map</title>" in page.text
    for name in SCRIPTS + ["style.css"]:
        r = client.get(f"/{name}")
        assert r.status_code == 200 and r.text == (WEB_DIR / name).read_text()
    assert client.get("/api/model").status_code == 200       # API routes are not shadowed
