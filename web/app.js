/* Accent Map page. Loads the model info and zones, runs the microphone or a file through the
   server, and shows every event with the same render(). Testable logic is in logic.js,
   drawing in map.js. Live and file mode receive identical events from the server; file mode
   shows the newest event whose time is <= the playback position. */
(function () {
  "use strict";
  var L = AccentLogic;
  var $ = function (id) { return document.getElementById(id); };

  var info = null;      // GET /api/model
  var classes = [];
  var byId = {};
  var map = null;       // AccentMap, or null if the model has no zones
  var live = null;      // the running microphone session
  var job = null;       // the running file analysis
  var shown;            // the event on screen (undefined = nothing rendered yet)

  init().catch(function (err) { showError("Could not load the model: " + err.message); });

  async function init() {
    info = await getJson("/api/model");
    classes = info.classes;
    classes.forEach(function (c) { byId[c.id] = c; });
    $("model-name").textContent = info.display_name;
    var a = info.audio;
    $("update-info").textContent = "Uses the last " + a.window_seconds + " s of audio · updates every " +
      a.update_seconds + " s" + (a.update_seconds > a.target_update_seconds
        ? " on this computer (" + a.target_update_seconds + " s on faster hardware)" : "");
    document.documentElement.style.setProperty("--fade", (a.update_seconds * 0.9).toFixed(2) + "s");
    info.caveats.forEach(function (text) {
      var li = document.createElement("li");
      li.textContent = text;
      $("caveats").appendChild(li);
    });
    buildAllList();
    if (info.map.zones_url) {
      var zones = await getJson(info.map.zones_url);
      map = AccentMap.create($("map"), info.map.panels, zones, { onHover: showTooltip });
      $("attribution").textContent = zones.attribution || "";
    } else {
      $("map-area").hidden = true;
    }
    render(null);
    $("mic-button").addEventListener("click", function () { if (live) stopLive(); else startLive(); });
    $("player").addEventListener("seeked", onPlayerMoved);
    $("player").addEventListener("play", onPlayerMoved);
    $("file-input").addEventListener("change", function (e) {
      var f = e.target.files[0];
      e.target.value = "";          // allow choosing the same file again
      if (f) startFile(f);
    });
    setControls();
  }

  async function getJson(url) {
    var res = await fetch(url);
    if (!res.ok) throw new Error(url + ": " + res.status + " " + res.statusText);
    return res.json();
  }

  // ---- Rendering (shared by live and file mode) --------------------------------------

  function render(event, hint) {
    shown = event;
    var d = L.describe(event, byId, info.session);
    $("status").className = "status " + d.status;
    $("status-badge").textContent = d.badge;
    $("status-text").textContent = hint || d.text;
    var probs = event ? event.smoothed : null;
    if (map) map.update(probs ? L.zoneProbabilities(classes, probs) : null, event ? event.status : "idle");
    renderTop(event);
    renderAll(probs);
  }

  function renderTop(event) {
    var list = $("top-list");
    list.textContent = "";
    list.classList.toggle("tentative", !!event && event.status === "listening");
    var entries = L.topEntries(event, byId, info.session.top_k);
    if (!entries.length) {
      var empty = document.createElement("li");
      empty.className = "empty";
      empty.textContent = "No result yet.";
      list.appendChild(empty);
      return;
    }
    entries.forEach(function (e) {
      var li = document.createElement("li");
      li.innerHTML = '<div class="row"><span class="name"></span><span class="pct"></span></div>' +
        '<div class="bar"><div class="fill"></div></div>';
      li.querySelector(".name").textContent = e.label;
      li.querySelector(".pct").textContent = L.percent(e.p);
      li.querySelector(".fill").style.width = (e.p * 100).toFixed(1) + "%";
      hoverHighlights(li, byId[e.id] && byId[e.id].zone);
      list.appendChild(li);
    });
  }

  function buildAllList() {
    classes.forEach(function (c) {
      var li = document.createElement("li");
      li.dataset.id = c.id;
      li.innerHTML = '<div class="row"><span class="name"></span><span class="pct">–</span></div>' +
        '<div class="bar"><div class="fill"></div></div><p class="desc"></p>';
      li.querySelector(".name").textContent = c.label;
      li.querySelector(".desc").textContent = c.description;
      hoverHighlights(li, c.zone);
      $("all-list").appendChild(li);
    });
  }

  function renderAll(probs) {
    Array.prototype.forEach.call($("all-list").children, function (li) {
      var p = probs ? probs[li.dataset.id] || 0 : null;
      li.querySelector(".pct").textContent = p === null ? "–" : L.percent(p);
      li.querySelector(".fill").style.width = p === null ? "0" : (p * 100).toFixed(1) + "%";
    });
  }

  function hoverHighlights(element, zone) {
    if (!zone) return;
    element.addEventListener("mouseenter", function () { if (map) map.highlight(zone); });
    element.addEventListener("mouseleave", function () { if (map) map.highlight(null); });
  }

  function showTooltip(target, e) {
    var tip = $("tooltip");
    if (!target) { tip.hidden = true; return; }
    var title, text;
    if (target.zone) {
      var members = classes.filter(function (c) { return c.zone === target.zone; });
      var probs = shown ? shown.smoothed : null;
      title = members.map(function (c) { return c.label; }).join(" / ");
      text = members.map(function (c) { return c.description; }).join(" ");
      if (probs) {
        var p = members.reduce(function (s, c) { return s + (probs[c.id] || 0); }, 0);
        title += " · " + L.percent(p);
      }
    } else {
      title = target.label;
      text = target.text;
    }
    tip.querySelector("strong").textContent = title;
    tip.querySelector("span").textContent = text;
    tip.hidden = false;
    var x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    var y = Math.min(e.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }

  function showError(message) {
    $("error").textContent = message;
    $("error").hidden = !message;
  }

  function setDetails(text) {
    $("details").textContent = text;
  }

  function setControls() {
    var button = $("mic-button");
    button.disabled = !info || (live && live.stopping);
    button.textContent = !live ? "Start microphone" : live.stopping ? "Stopping…" : "Stop microphone";
    button.classList.toggle("recording", !!live && !live.stopping);
    $("file-input").disabled = !info || !!live;
  }

  // ---- Live microphone ---------------------------------------------------------------

  async function startLive() {
    stopFile();
    showError("");
    if (!navigator.mediaDevices || !window.AudioWorkletNode) {
      showError("This browser can't record here. Use a current Chrome, Edge or Firefox on http://localhost.");
      return;
    }
    live = { ready: false, queue: [], stopping: false, events: 0 };
    var session = live;
    setControls();
    render(null, "Allow microphone access when the browser asks.");
    try {
      session.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
      });
      session.ctx = new AudioContext();
      await session.ctx.audioWorklet.addModule("mic-worklet.js");
      var source = session.ctx.createMediaStreamSource(session.stream);
      session.node = new AudioWorkletNode(session.ctx, "capture", { processorOptions: { chunkSeconds: 0.1 } });
      var mute = session.ctx.createGain();
      mute.gain.value = 0;            // keep the graph running without playing the mic back
      source.connect(session.node);
      session.node.connect(mute);
      mute.connect(session.ctx.destination);
      session.node.port.onmessage = function (m) { onChunk(session, m.data); };

      var ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/live");
      ws.binaryType = "arraybuffer";
      session.ws = ws;
      ws.onopen = function () { ws.send(JSON.stringify({ type: "start", sample_rate: session.ctx.sampleRate })); };
      ws.onmessage = function (m) { onLiveMessage(session, JSON.parse(m.data)); };
      ws.onclose = function () {
        if (live === session && !session.stopping) showError("The connection to the server was lost.");
        finishLive(session);
      };
      render(null, "Speak now. The first result appears after about " + info.session.listen_speech_seconds +
        " s of speech.");
    } catch (err) {
      showError(micErrorText(err));
      finishLive(session);
    }
  }

  function onChunk(session, data) {
    var db = 20 * Math.log10(Math.max(data.rms, 1e-6));
    $("level-fill").style.width = Math.max(0, Math.min(100, (db + 60) / 60 * 100)).toFixed(0) + "%";
    if (session.stopping) return;
    if (session.ready && session.ws.readyState === WebSocket.OPEN) {
      session.ws.send(data.samples.buffer);
    } else if (session.queue.length < 50) {
      session.queue.push(data.samples.buffer);    // until the server says ready (max ~5 s)
    }
  }

  function onLiveMessage(session, msg) {
    if (msg.type === "ready") {
      session.ready = true;
      session.queue.forEach(function (buf) { session.ws.send(buf); });
      session.queue = [];
    } else if (msg.type === "event") {
      session.events += 1;
      render(msg);
      setDetails("Last update: model " + (msg.infer_ms ? Math.round(msg.infer_ms) + " ms" : "skipped (no speech)") +
        " · server delay " + (msg.latency_ms / 1000).toFixed(2) + " s · windows skipped " + msg.dropped);
    } else if (msg.type === "done") {
      session.ws.close();
    } else if (msg.type === "error") {
      showError("Server: " + msg.message);
    }
  }

  function stopLive() {
    var session = live;
    if (!session || session.stopping) return;
    session.stopping = true;
    releaseAudio(session);
    setControls();
    if (session.ws && session.ws.readyState === WebSocket.OPEN && session.ready) {
      session.ws.send(JSON.stringify({ type: "stop" }));   // server analyzes the rest, then "done"
      setTimeout(function () { finishLive(session); }, 10000);
    } else {
      finishLive(session);
    }
  }

  function releaseAudio(session) {
    if (session.node) session.node.port.onmessage = null;
    if (session.stream) session.stream.getTracks().forEach(function (t) { t.stop(); });
    if (session.ctx && session.ctx.state !== "closed") session.ctx.close();
    $("level-fill").style.width = "0";
  }

  function finishLive(session) {
    if (live !== session) return;
    releaseAudio(session);
    if (session.ws && session.ws.readyState <= WebSocket.OPEN) session.ws.close();
    live = null;
    setControls();
  }

  function micErrorText(err) {
    if (err.name === "NotAllowedError") return "Microphone access was blocked. Allow it in the address bar and try again.";
    if (err.name === "NotFoundError") return "No microphone was found.";
    return "Could not start the microphone: " + err.message;
  }

  // ---- File: upload once, analyze from the playhead, show results in sync with playback ---
  // The server analyzes forward from a start position. Results are cached by window, so parts
  // already analyzed show at once. When the listener skips to a part that is not analyzed and
  // the running pass will not get there in time, the page abandons that pass (the server then
  // stops it) and asks for a new one, a little ahead of the playhead while playing, so results
  // are ready when playback reaches them (logic.js newPass, needsRestart).
  // All this needs one inference per update to be faster than playback. When the computer is
  // slower than the server measured at startup, the page asks the server to cut the file again
  // with a longer interval (recut) and keeps showing the earlier results until new ones come.

  async function startFile(f) {
    stopFile();
    showError("");
    var player = $("player");
    var current = { id: null, name: f.name, url: URL.createObjectURL(f), file: null, events: [],
      stream: null, inference: null, recent: [], recutting: false, previous: null, frame: 0, hint: undefined,
      lastCheck: 0 };
    job = current;
    player.src = current.url;
    player.hidden = false;
    render(null, "Uploading " + f.name + "…");
    try {
      var body = new FormData();
      body.append("file", f, f.name);
      var res = await fetch("/api/files", { method: "POST", body: body });
      if (!res.ok) throw new Error(await errorDetail(res));
      var meta = await res.json();
      if (job !== current) return;                  // another file was chosen meanwhile
      current.id = meta.id;
      current.file = { first_t: meta.first_t, hop: meta.update_seconds, duration: meta.duration,
        windows: meta.windows };
      startStream(current);
      current.frame = requestAnimationFrame(tick);
    } catch (err) {
      if (job === current) {
        showError(err.message);
        render(null);
        setDetails("");
      }
    }
  }

  // What logic.js needs to know about playback right now.
  function view(current, seeked) {
    var player = $("player");
    return { time: player.currentTime, playing: !player.paused, now: performance.now() / 1000,
      infer: current.inference ? current.inference.seconds : current.file.hop / 1.2, seeked: !!seeked };
  }

  function startStream(current) {
    var stream = L.newPass(current.file, current.stream, view(current), current.events);
    if (current.stream) current.stream.controller.abort();
    stream.controller = new AbortController();
    current.stream = stream;
    updateFileDetails(current);
    readStream(current, stream).catch(function (err) {
      if (err.name !== "AbortError" && job === current) showError(err.message);
    });
  }

  async function readStream(current, stream) {
    var fileId = current.id;
    var url = "/api/files/" + encodeURIComponent(fileId) + "/events?start=" + stream.start.toFixed(2);
    var res = await fetch(url, { signal: stream.controller.signal });
    if (!res.ok) throw new Error(await errorDetail(res));
    var reader = res.body.getReader();
    var decoder = new TextDecoder();
    var parser = L.ndjsonParser();
    for (;;) {
      var chunk = await reader.read();
      var messages = chunk.done ? parser.end() : parser.push(decoder.decode(chunk.value, { stream: true }));
      messages.forEach(function (m) {
        if (current.id !== fileId) return;                // the file was cut again: other windows
        if (m.type === "event") {
          L.insertEvent(current.events, m);
          L.passResult(stream, m, performance.now() / 1000);
          current.inference = L.noteInference(current.inference, m);
          current.recent = L.noteRecent(current.recent, m, info.audio.window_seconds);
        } else if (m.type === "done") {
          stream.done = true;
        }
      });
      if (messages.length) updateFileDetails(current);
      if (chunk.done) break;
    }
    stream.done = true;
    updateFileDetails(current);
  }

  function updateFileDetails(current) {
    if (job !== current || !current.file) return;
    var s = current.stream;
    setDetails(current.name + ": " + current.events.length + " of " + current.file.windows + " updates ready · " +
      (s && !s.done ? "analyzing from " + L.clock(s.got ? s.first : s.start) : "analysis idle") +
      " · one update every " + current.file.hop + " s");
  }

  function checkCoverage(current, seeked) {
    if (!current.file || current.recutting) return;
    if (L.slowerThanPlayback(current.recent, current.file.hop)) { recut(current); return; }
    if (L.needsRestart(current.events, current.file, current.stream, view(current, seeked))) startStream(current);
  }

  // Full windows take about as long as the file's interval: results would keep falling behind
  // playback. Ask the server to cut the file again with the interval it sustains now, then
  // continue from the playhead. If it keeps the interval, ask again only after 3 new results.
  async function recut(current) {
    current.recutting = true;
    current.recent = [];
    try {
      var res = await fetch("/api/files/" + encodeURIComponent(current.id) + "/recut", { method: "POST" });
      if (!res.ok) throw new Error(await errorDetail(res));
      var meta = await res.json();
      if (job !== current || meta.id === current.id) return;
      current.previous = { events: current.events, hop: current.file.hop };   // shown until new results come
      current.id = meta.id;
      current.file = { first_t: meta.first_t, hop: meta.update_seconds, duration: meta.duration,
        windows: meta.windows };
      current.events = [];
      startStream(current);                              // aborts the old pass
    } catch (err) {
      if (job === current) showError(err.message);
    } finally {
      current.recutting = false;
    }
  }

  function onPlayerMoved() {
    if (job) checkCoverage(job, true);
  }

  function tick() {
    var current = job;
    if (!current) return;
    var player = $("player");
    var now = performance.now();
    if (!player.paused && now - current.lastCheck > 250) {
      current.lastCheck = now;
      checkCoverage(current);
    }
    var t = player.currentTime;
    var event = L.freshEventAt(current.events, t, current.file.hop) ||
      (current.previous ? L.freshEventAt(current.previous.events, t, current.previous.hop) : null);
    var hint = null;
    if (!event) {
      var s = current.stream;
      hint = t < current.file.first_t
        ? (player.paused ? "Press play. The map follows the audio as it plays." : "Listening…")
        : "Analyzing this part…" + (s && s.got && s.first > t ? " The map continues at " + L.clock(s.first) + "." : "");
    }
    if (event !== shown || hint !== current.hint) {
      current.hint = hint;
      render(event, hint);
    }
    current.frame = requestAnimationFrame(tick);
  }

  function stopFile() {
    var current = job;
    if (!current) return;
    job = null;
    if (current.stream) current.stream.controller.abort();
    cancelAnimationFrame(current.frame);
    var player = $("player");
    player.pause();
    player.removeAttribute("src");
    player.load();
    player.hidden = true;
    URL.revokeObjectURL(current.url);
    setDetails("");
  }

  async function errorDetail(res) {
    try {
      var body = await res.json();
      if (body && body.detail) return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (e) { /* not JSON */ }
    return "Upload failed: " + res.status + " " + res.statusText;
  }
})();
