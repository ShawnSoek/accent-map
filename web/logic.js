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

  // Index of the latest event with t <= time, or -1. `events` is sorted by t.
  function indexAt(events, time) {
    var lo = 0, hi = events.length - 1, found = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (events[mid].t <= time + 1e-6) { found = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    return found;
  }

  // The latest event with t <= time, or null.
  function eventAt(events, time) {
    var i = indexAt(events, time);
    return i < 0 ? null : events[i];
  }

  // Add an event to a list sorted by t. Returns false (and changes nothing) if an event with the
  // same index is already there: windows are numbered by the server, the same in every pass.
  function insertEvent(events, event) {
    var lo = 0, hi = events.length;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (events[mid].t < event.t) lo = mid + 1; else hi = mid;
    }
    if (lo < events.length && events[lo].index === event.index) return false;
    events.splice(lo, 0, event);
    return true;
  }

  // The event to show at playback time `time`: the latest one at or before it, if recent enough
  // (windows end every `hop` seconds, so an older one belongs to a part that was skipped).
  function freshEventAt(events, time, hop) {
    var e = eventAt(events, time);
    return e && time - e.t <= hop + 0.3 ? e : null;
  }

  // t of the last result in the unbroken run of results that covers `time`, or null.
  function analyzedUntil(events, time, hop) {
    var i = indexAt(events, time);
    if (i < 0 || time - events[i].t > hop + 0.3) return null;
    while (i + 1 < events.length && events[i + 1].t - events[i].t <= hop + 1e-6) i++;
    return events[i].t;
  }

  // ---- File mode: which part of the file the server analyzes ----
  // A pass is one request: the server analyzes forward from the last window ending at or before
  // `start`, one window per inference, and cannot stop an inference halfway (it starts the next
  // one as soon as it has sent a result). Results must reach the page before playback gets to
  // them: one that arrives late is already older than one update and is not shown.
  //   file = {first_t, hop, duration}
  //   view = {time (playhead, s), playing, now (clock, s), infer (s per full window),
  //           seeked (true when the listener just moved the playhead or pressed play)}
  //   pass = {start, origin (playhead when asked), paused (asked while paused), due (clock time
  //           its first result is expected), got (any result yet), first (t of its first
  //           result), position (t of its newest result, or start), lastAt (clock time of its
  //           newest result, or of the request), done}

  // Seconds of model time per full window: the newest result among the longest windows seen.
  function noteInference(estimate, event) {
    if (event.infer_ms == null) return estimate;                  // no speech: model not run
    if (estimate && event.window_seconds < estimate.window - 1e-6) return estimate;
    return { window: event.window_seconds, seconds: event.infer_ms / 1000 };
  }

  // Seconds until a pass asked for now sends its first result: the rest of the inference the
  // server is running for the current pass, then one more.
  function firstResultDelay(pass, view) {
    var busy = !pass || pass.done ? 0
      : pass.got ? Math.max(0, view.infer - (view.now - pass.lastAt)) : view.infer;
    return busy + view.infer + 0.15;
  }

  // A new pass. Paused: start at the playhead, so the result for this moment shows while paused.
  // Playing: start right after the analyzed stretch the playhead is in, if the first result can
  // be there in time; otherwise one update past the moment it can arrive, so the first window
  // ends after where the playhead will be then. Later results come faster than playback (one
  // inference per update), so they arrive in time too.
  function newPass(file, previous, view, events) {
    var delay = firstResultDelay(previous, view);
    var start = view.time;
    if (view.playing) {
      var earliest = view.time + delay;                 // a window ending before this comes late
      var last = analyzedUntil(events, Math.max(view.time, file.first_t), file.hop);
      var next = last === null ? null : Math.min(last + file.hop, file.duration);
      start = next !== null && next >= earliest - 0.01 ? next + 0.01 : earliest + file.hop - 0.01;
    }
    start = Math.min(Math.max(start, 0), file.duration);
    return { start: start, origin: view.time, paused: !view.playing, due: view.now + delay,
      got: false, first: null, position: start, lastAt: view.now, done: false };
  }

  function passResult(pass, event, now) {
    if (!pass.got) { pass.got = true; pass.first = event.t; }
    pass.position = event.t;
    pass.lastAt = now;
  }

  // Should the page abandon `pass` (it may be null) and ask for a new one?
  function needsRestart(events, file, pass, view) {
    var target = Math.max(view.time, file.first_t);              // no window ends before the first
    var last = analyzedUntil(events, target, file.hop);           // null: this moment is not analyzed
    if (last !== null && (!view.playing || last >= file.duration - 1e-6)) return false;
    if (!pass) return true;
    if (target < pass.origin - 0.25) return true;                 // skipped back before this pass
    if (!pass.got) {
      // Waiting for its first result, which is for a window ending at or after `low`. Playback
      // drifting past it is not a reason to give up: a new pass would wait just as long.
      var low = Math.max(pass.start - file.hop, file.first_t);
      if (pass.paused && view.playing) {                          // asked while paused, now playing
        return view.time + Math.max(0, pass.due - view.now) - low > 0.3;
      }
      return !!view.seeked && target > pass.start + file.hop + 0.3;  // skipped well past its start
    }
    if (last !== null) {
      // Playing an analyzed stretch: will a result for the window right after it come in time?
      var next = Math.min(last + file.hop, file.duration);
      if (pass.position < next - 1e-6) {
        if (pass.done) return true;                               // the pass ended before it
        var steps = Math.ceil((next - pass.position) / file.hop - 1e-6);
        var at = Math.max(pass.lastAt + view.infer, view.now) + (steps - 1) * view.infer;
        return at > view.now + (next - view.time) + 0.3;
      }
      // The pass started after that window (maybe because it could not be ready in time): replace
      // it only if a new pass would fill the gap sooner.
      var delay = firstResultDelay(pass, view);
      var resume = next >= view.time + delay - 0.01 ? next : view.time + delay + file.hop;
      return resume < pass.first - 0.01;
    }
    if (pass.done) return target >= pass.first;                   // ended early; else in its lead gap
    // How far the playhead will be past the pass's next result when that result arrives.
    var busy = view.playing ? Math.max(0, view.infer - (view.now - pass.lastAt)) : 0;
    return target + busy - (pass.position + file.hop) > file.hop + 0.3;
  }

  function clock(seconds) {
    var s = Math.max(0, Math.floor(seconds));
    var rest = s % 60;
    return Math.floor(s / 60) + ":" + (rest < 10 ? "0" : "") + rest;
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
    insertEvent: insertEvent,
    freshEventAt: freshEventAt,
    analyzedUntil: analyzedUntil,
    noteInference: noteInference,
    firstResultDelay: firstResultDelay,
    newPass: newPass,
    passResult: passResult,
    needsRestart: needsRestart,
    clock: clock,
    ndjsonParser: ndjsonParser,
    percent: percent,
    topEntries: topEntries,
    describe: describe
  };
})();
