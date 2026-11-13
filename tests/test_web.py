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
