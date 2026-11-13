/* Pure display logic for the Accent Map page. No DOM access, so tests/test_web.py can run it
   in QuickJS. Loaded as a classic script; exposes the global AccentLogic. */
var AccentLogic = (function () {
  "use strict";

  function clamp01(x) {
    return x < 0 ? 0 : x > 1 ? 1 : x;
  }

  // Equirectangular projection for one map panel. x is scaled by cos(middle latitude) so
  // shapes are not stretched sideways. Units are degrees of latitude.
  function projection(bbox) {
    var west = bbox[0], south = bbox[1], east = bbox[2], north = bbox[3];
    var k = Math.cos(((south + north) / 2) * Math.PI / 180);
    return {
      width: (east - west) * k,
      height: north - south,
      point: function (lon, lat) { return [(lon - west) * k, north - lat]; }
    };
  }

  function ringPath(ring, proj) {
    var out = "";
    for (var i = 0; i < ring.length; i++) {
      var xy = proj.point(ring[i][0], ring[i][1]);
      out += (i === 0 ? "M" : "L") + xy[0].toFixed(2) + " " + xy[1].toFixed(2);
    }
    return out + "Z";
  }

  // GeoJSON Polygon or MultiPolygon -> SVG path data. Draw with fill-rule="evenodd" for holes.
  function geometryPath(geometry, proj) {
    var polygons = geometry.type === "Polygon" ? [geometry.coordinates]
      : geometry.type === "MultiPolygon" ? geometry.coordinates : [];
    var out = "";
    polygons.forEach(function (rings) {
      rings.forEach(function (ring) { out += ringPath(ring, proj); });
    });
    return out;
  }

  // Probability per map zone. Several classes may share a zone; their probabilities add up.
  function zoneProbabilities(classes, probs) {
    var out = {};
    classes.forEach(function (c) {
      out[c.zone] = (out[c.zone] || 0) + (probs ? (probs[c.id] || 0) : 0);
    });
    return out;
  }

  // How strongly a zone is coloured (0..1). Faint while listening or without speech.
  var DIMMED = { listening: 0.45, no_speech: 0.6 };
  function zoneOpacity(p, status) {
    var o = 0.04 + 0.96 * Math.pow(clamp01(p), 0.75);
    return Math.round(o * (DIMMED[status] || 1) * 1000) / 1000;
  }

  // The latest event with t <= time, or null. `events` is sorted by t (the server sends them so).
  function eventAt(events, time) {
    var lo = 0, hi = events.length - 1, found = null;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (events[mid].t <= time + 1e-6) { found = events[mid]; lo = mid + 1; } else { hi = mid - 1; }
    }
    return found;
  }

  // Newline-delimited JSON arriving in arbitrary text pieces.
  function ndjsonParser() {
    var buffer = "";
    function take(final) {
      var lines = buffer.split("\n");
      buffer = final ? "" : lines.pop();
      return lines.filter(function (l) { return l.trim() !== ""; }).map(function (l) { return JSON.parse(l); });
    }
    return {
      push: function (text) { buffer += text; return take(false); },
      end: function () { return take(true); }
    };
  }

  function percent(p) {
    var v = Math.round(p * 100);
    return (v < 1 && p > 0 ? "<1" : String(v)) + " %";
  }

  function topEntries(event, byId, k) {
    return (event && event.top ? event.top : []).slice(0, k).map(function (e) {
      return { id: e.id, label: byId[e.id] ? byId[e.id].label : e.id, p: e.p };
    });
  }

  // Status badge and sentence for an event (null = nothing analyzed yet).
  function describe(event, byId, session) {
    if (!event) {
      return { status: "idle", badge: "Ready", text: "Start the microphone or choose an audio file." };
    }
    var top = topEntries(event, byId, 2);
    if (event.status === "listening") {
      return {
        status: "listening", badge: "Listening…",
        text: "Heard " + event.speech_seconds.toFixed(1) + " of " + session.listen_speech_seconds.toFixed(1) +
          " s of speech. Early guesses are shown faintly."
      };
    }
    if (event.status === "no_speech") {
      return {
        status: "no_speech", badge: "No speech",
        text: top.length ? "Nothing loud enough in the last few seconds. Showing the last result."
          : "Waiting for speech…"
      };
    }
    if (event.status === "uncertain") {
      return {
        status: "uncertain", badge: "Uncertain",
        text: "No clear winner: " + top.map(function (e) { return e.label + " " + percent(e.p); }).join(" vs ") + "."
      };
    }
    if (event.status === "confident") {
      return { status: "confident", badge: "Confident", text: "Most likely " + top[0].label + " (" + percent(top[0].p) + ")." };
    }
    return { status: "idle", badge: String(event.status), text: "" };
  }

  return {
    projection: projection,
    geometryPath: geometryPath,
    zoneProbabilities: zoneProbabilities,
    zoneOpacity: zoneOpacity,
    eventAt: eventAt,
    ndjsonParser: ndjsonParser,
    percent: percent,
    topEntries: topEntries,
    describe: describe
  };
})();
