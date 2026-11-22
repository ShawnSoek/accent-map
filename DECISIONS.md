# Decisions

Short notes on each choice and why. Newest at the bottom.

## 2026-11-13: Stage 1 setup

- **Run inference on CPU.** This machine has an AMD Ryzen 7 7730U with integrated Radeon
  graphics and no NVIDIA driver, so CUDA isn't available. ROCm doesn't support this iGPU.
  torch-directml was rejected because it needs an old torch and may not work with SpeechBrain.
  Latency is measured on the CPU.
- **New environment on Python 3.12.** The old `.venv` (Python 3.10) had a CUDA torch build, a
  torchaudio that didn't match torch, and no `transformers`. Current torch, transformers and
  FastAPI require Python 3.11 or newer. The old env is kept as `.venv-old`, not deleted.
- **Package pins.** `requirements.in` lists the top-level packages, and `uv pip compile`
  generates the fully pinned `requirements.txt`.
  - torch 2.11.0+cpu with torchaudio 2.11.0+cpu. 2.11.0 is the newest torchaudio on the
    PyTorch CPU index, and torchaudio has to match torch's minor version.
  - speechbrain 1.1.2.
  - transformers 4.57.6. This is the last 4.x; SpeechBrain's HF wrapper was written against 4.x.
  - Install with `uv pip install -r requirements.txt --torch-backend cpu`. The flag sends
    torch/torchaudio to the CPU index and everything else to PyPI.
- **Pin both model revisions.**
  - `Jzuluaga/accent-id-commonaccent_xlsr-es-spanish` @ `0eff6f06bdbbcd75cafe3260c00a93985ca36d2d`
  - `facebook/wav2vec2-large-xlsr-53` @ `c3f9d884181a224a6ac87bf8885c84d1cff3384f`. SpeechBrain's
    wav2vec2 wrapper builds the architecture from this base model, and then the fine-tuned
    checkpoint overwrites the weights.
  - The model's `hyperparams.yaml` uses `pretrained_path: <repo name>`, which would load HEAD.
    We override it to a local snapshot of the pinned revision.
- **Check weights strictly after loading.** SpeechBrain's `torch_parameter_transfer` loads
  checkpoints with `strict=False` and only logs a warning. A renamed key would give a model that
  runs but is wrong. We compare checkpoint keys with the module keys ourselves and fail on any
  mismatch.
- **Map class indices by label string.** Each class in `config.yaml` has a `model_label`. The
  adapter reads index→label from the model's label encoder and maps by that string. The class order
  in `config.yaml` doesn't matter, and loading fails if the label sets differ.
- **Plain HTML/JS frontend, no build step.** Node isn't installed, and a build step isn't needed
  for one page. Everything that needs tests (windowing, smoothing, status) is in Python and tested
  with pytest. The browser only renders.
- **Load the model with a local `hyperparams.yaml` (option B1+, approved).**
  `models/commonaccent_es6/hyperparams.yaml` is a copy of the pinned file with two changes,
  marked `CHANGED` in the file:
  1. The wav2vec2 class path. The model card's original fails on SpeechBrain 1.1.2 with
     `ImportError: There is no such class as speechbrain.lobes.models.huggingface_wav2vec.HuggingFaceWav2Vec2`.
     The class is now `speechbrain.integrations.huggingface.wav2vec2.Wav2Vec2`, with the same arguments.
  2. Added `preout_mlp`. **The published config is broken.** It leaves out the two-layer
     head the checkpoint was trained with, so none of the `model.ckpt` keys match. SpeechBrain
     only warns and keeps a randomly initialised output layer. On the 4 sample clips, the published
     head gives near-uniform probabilities (0.2–0.3) that change with the random seed. The
     rebuilt head gives 0.86–0.92.
     - The head is copied from the training config `train_w2v2.yaml` in
       JuanPZuluaga/accent-recog-slt2022: VanillaNN, GELU, 2 × 1024. All 5 checkpoint tensors load
       strictly.
     - GELU is the activation in all 11 versions of that config. We can't prove which exact version
       trained this model.
     - `custom_interface.py`'s `classify_*` methods also skip `preout_mlp`. Our adapter only uses its
       `encode_batch`, then runs `preout_mlp → output_mlp → softmax` itself.
  - Rejected: B2, the 2023 stack (speechbrain 0.5.16). It needs Python ≤ 3.11 and a torch older
    than 2.6, which has the unsafe model-loading flaw. Also rejected: B3, our own loader, which
    has more ways to get subtle details wrong.
  - Verified: zero SpeechBrain transfer warnings. A strict reload (`wav2vec2` 422 keys, `model` 5
    keys) finds nothing missing or unexpected and changes no tensors. Outputs are probabilities
    (sum 1.000000, SpeechBrain `Softmax` without log). Load takes about 6–8 s and peaks at 2.7 GB,
    or 3.9 GB with the extra verification reload.
  - Sample clips: `mexico` → Mexico 0.92, `spain` → Southern Spain 0.92, `caribe-cuba-colombia` → Caribbean 0.86,
    `andino` → **Caribbean 0.91**. File names are hints, not verified labels.
- **Compare model outputs with a tolerance, not exact equality.** All modules are in eval mode,
  so dropout and LayerDrop are off. Repeated runs still differ by up to 1.2e-7 in probabilities,
  even with 1 thread. That's float32 rounding in the CPU kernels.
- **Map shapes from Natural Earth (public domain).** A build script merges admin-1 regions into
  dialect zones and writes a small `zones.geojson`. Only that output is committed. The map is drawn
  as inline SVG with no tiles or map library, so it works offline. Zones are approximations of
  dialect areas, not borders.

## 2026-11-13: Model plug-in (step 3)

- **What `config.yaml` contains.** Besides the fields in CLAUDE.md (display name, source,
  pinned revision, classes with id/label/description/zone), it also has:
  - `model_label` on each class.
  - `backend` and `backend_options`.
  - `audio` (sample rate, window, hop), so a later model can use a different window.
  - `caveats`, which the UI will show.
  - Unknown keys are errors, so typos fail loudly. `source` is optional only so the fake
    backend can work without one; the SpeechBrain backend requires it.
- **No downloads when the server starts.** `scripts/fetch_model.py <model dir>` downloads
  the pinned revisions. The backend only reads the local cache (`local_files_only`) and tells
  you to run the fetch script if a file is missing.
- **Strict loading replaces SpeechBrain's `Pretrainer`.** The adapter loads each checkpoint
  listed in `backend_options.checkpoints` with `strict=True`, and the label encoder with
  `CategoricalEncoder.load`.
  - `mean_var_norm_emb` isn't loaded: `encode_batch` never uses it, and its checkpoint is
    empty (count 0).
  - Checked by hand: a config without the rebuilt head now fails to load with "Missing key(s):
    0.w.weight" instead of quietly using a random head.
- **The contract is enforced in the base class.** Every backend's `predict()` call is checked:
  - Input must be 1-D float, finite, and at least 0.5 s long (`MIN_AUDIO_SECONDS`).
  - Output must use exactly the config ids, each value must be in [0, 1], and the values must
    sum to 1 (±1e-4).
- **Fake backend for tests and demos.** It works with any class count and has no torch import.
  A sine tone at `base_hz + i·step_hz` selects class i, and silence gives a uniform result. The
  fast test suite uses it, and so will the UI check with a different class count.
- **The regression values use the app's soxr resampling.** These differ from the step-2 values,
  which used SpeechBrain/torchaudio resampling, by up to 0.013, with the same top-1 class. The
  test tolerance is 1e-3: floating-point noise is about 1e-7, so any real change in the model
  or pipeline will fail it.

## 2026-11-13: Speed (step 4) and option 1

- **Measured on the CPU** (`scripts/bench_latency.py`; median of 30 runs, laptop on mains power):
  - A 5 s window takes 805 ms with torch's default 8 threads. With 4 threads it takes 990 ms;
    with 16 it takes 706 ms but is less stable and leaves no CPU for the browser.
  - A 3 s window takes 493 ms, a 1 s window 240 ms.
  - Measuring the first setting again at the end gave 259 ms vs 254 ms, so the CPU didn't slow down.
- **Decision: keep the 5 s window and update every 1.0 s on this machine (option 1).**
  - The model config still asks for 0.5 s. At startup the server measures one full-window
    inference and picks the smallest multiple of 0.25 s that is at least 1.2 × that time
    (`server/pacing.py`). That gives 0.5 s on fast hardware and 1.0 s here, where the startup
    measurement was 777 ms.
  - At 1.0 s, file analysis runs about 1.25× faster than playback, so files stay in sync without
    pre-buffering.
  - Rejected:
    - keeping 0.5 s and skipping windows: irregular updates, and file mode can't keep up
    - a 3 s window: no headroom, and probably less accurate, which we can't measure honestly
    - int8 quantization: possible later as a separate, verified experiment

## 2026-11-13: Audio, windowing, smoothing, session (step 5)

- **Resampling uses soxr, streaming for the mic.** Streamed output equals whole-signal output
  exactly in tests at 44.1 and 48 kHz.
- **Windows end on fixed boundaries** (k × interval from the start of the stream). This makes
  them independent of how the browser splits audio into chunks, so the same audio gives the same
  windows in live and file mode. The window grows until it reaches 5 s; the first one needs 1 s
  of audio.
  - At the end of a file, one extra window covers the leftover audio after the last boundary.
- **Live mode analyzes only the newest window when the model is busy.** The server (step 6)
  implements this. Skipped windows still count toward "speech heard", because that total is
  computed when audio arrives, not when a window is analyzed.
- **Silence is detected with a loudness threshold, not a speech detector.** A 20 ms frame counts
  as sound if it is louder than −45 dBFS. Windows with less than 0.75 s of sound aren't sent to
  the model; the status is `no_speech` and the map keeps the last result. Background noise
  louder than −45 dBFS counts as sound, which is a known limitation.
- **Smoothing is a time-based moving average with τ = 1.5 s.** The weight is
  `alpha = 1 − exp(−dt/τ)`, so the feel doesn't depend on the update rate, and after a long pause
  it restarts from the new prediction. In tests, a raw output flipping between 0.3 and 0.7 on every
  update becomes a wobble of under 0.1.
- **Statuses** (the thresholds are hand-set and never tuned on `data/own_tests/`):
  - `listening` until 2 s of sound has been heard in total
  - then `confident` if the smoothed top class is at least 0.5 and leads the second by at least 0.15
  - otherwise `uncertain`
- **Observation on the sample clips with the real model.** Early, short windows can be
  misleading: `caribe-cuba-colombia.wav` shows Mexico (0.69, `confident`) at 4 s and switches to
  Caribbean once the window is 5 s long. We're not tuning on 4 clips; this is a candidate for
  raising `listen_speech_seconds` if real use confirms it.

## 2026-11-13: Server (step 6)

- **FastAPI + uvicorn, one model, one inference thread.** All sessions share one model
  instance, and inferences run one at a time on a single worker thread. That's enough for a
  single-user demo; a second simultaneous user would wait their turn.
- **Live protocol:** the browser sends mono float32 audio at its own sample rate after a JSON
  `start` message, and the server resamples with the streaming soxr resampler. When the model is
  busy, only the newest window is analyzed; skipped windows are counted in each event (`dropped`).
- **File upload is decoded on the server** with soundfile (WAV, FLAC, OGG, MP3).
  - Results come back as NDJSON, one event per line, as soon as each is ready.
  - The file is fed through the session one update interval at a time, so memory stays flat
    for long files.
  - Limits: 100 MB and 15 min. Under 1 s of audio is rejected with 422. Unreadable formats such
    as M4A get a 415 that says which formats work.
- **Startup:** loading takes about 7 s, then timing one inference takes about 3 s, which picks
  the update interval. `--update-seconds` skips the timing.
- **Measured end to end** (`scripts/stream_wav.py`): the 4 sample clips were streamed at 48 kHz
  in real time, in 100 ms chunks.
  - No windows were skipped at the 1.0 s interval, and the results equal file mode.
  - Delay from audio sent to event received, with full 5 s windows: about 0.9 s (about 0.8 s of
    inference plus about 0.1 s while the resampler waits for the next chunk). The browser's own
    capture delay isn't included.
- **The test client uses `httpx2`** (2026-11-14), as Starlette 1.10 recommends. `httpx` was
  removed, which also removed its deprecation warning.

## 2026-11-13: Map zones (step 7)

- **Source:** Natural Earth 10m admin-1 states and provinces, v5.1.2 (public domain), pinned to
  commit `f1890d9f` of nvkelso/natural-earth-vector. File SHA-256 `22d0e3ad…2fb5`; the build
  script refuses any other file. The download stays in `data/naturalearth/`; only the generated
  `zones.geojson` (210 KB) is committed. Rebuilding gives a byte-identical file.
- **The zone table is `models/commonaccent_es6/zones_spec.yaml`, readable and commented.**
  - Regions are selected by ISO 3166-2 code, by Spanish autonomous community, or by whole country.
    The most specific rule wins. A typo, or a code listed twice, stops the build.
  - Debatable choices. Decided 2026-11-14 by reading each class as a description of how people
    speak, at the user's request ("pick the most plausible"):
    - Campeche and Yucatán → **Mexico** (changed from Caribbean). Yucatán-peninsula Spanish keeps
      the final s and lacks Caribbean features. The label's "Gulf coast" dialect means Veracruz
      and Tabasco. Quintana Roo stays Caribbean as the "Mexican Caribbean".
    - Tarija → Rioplatense ("eastern Bolivia"). Its Spanish leans toward north-western Argentina,
      which is Rioplatense too.
    - Tamaulipas → Mexico: on the Gulf, but its Spanish is northern Mexican.
    - North-western Argentina → Rioplatense, because the model's label says "Argentina".
- **"Not covered" areas are drawn greyed out:** the rest of Spain and Central America except
  Panama. Everything else, such as Brazil, Haiti and the US, is plain land with no borders.
  The Canaries, Equatorial Guinea and the Philippines fall outside the map panels.
- **Two panels:** the Americas, and a Spain inset. They're defined in `config.yaml` → `map.panels`
  and each feature carries its panel, so another model can define different panels.
- **Neighbouring zones are simplified together** (`shapely.coverage_simplify`), so they keep
  exactly shared borders, with no gaps or slivers. Islands smaller than 0.02 square degrees
  (Americas panel) or 0.005 (Spain panel) are dropped.
- **Checked by tests:**
  - every class has its zone
  - the geometry is valid, stays inside its panel and has no overlaps
  - 44 cities land in the intended zone (e.g. Port-au-Prince is plain land, Santo Domingo is
    Caribbean, Mendoza is Chilean, Tarija is Rioplatense, Mérida (Venezuela) is Andean while
    Caracas is Caribbean)

## 2026-11-14: Web page (step 8)

- **Plain HTML, CSS and classic scripts.** No build step, no CDN, so the page works offline. The
  server serves `web/` at `/`, and the API routes take precedence.
- **Split for testing:**
  - `logic.js` holds the pure functions: projection, path building, zone probabilities, opacity,
    timeline lookup, NDJSON parsing and status text. pytest runs it in QuickJS, a small JS
    engine added as a dev-only dependency.
  - `map.js` is tested against a tiny fake DOM with the real zones.
  - Every script gets a syntax check, and a test confirms every element id that `app.js` uses
    exists in `index.html`.
  - The browser wiring in `app.js` (microphone, playback, tooltips) can only be checked by hand.
- **Smoothing stays on the server.** The browser only fades colours: the CSS transition lasts
  90 % of the update interval, so the map moves continuously without flickering.
- **Colour: one hue for every zone, whatever the class count.** Opacity is
  `0.04 + 0.96·p^0.75`. Each zone has an opaque base under its coloured overlay, so an unlikely
  zone still looks like land rather than sea.
  - While `listening`, the map is dimmed to 45 % and the top-3 list is faded. While `no_speech`,
    the map shows the last result at 60 %.
- **Microphone:** echo cancellation, noise suppression and auto gain are switched off, so the
  model hears the raw voice. An AudioWorklet sends 100 ms mono chunks at the browser's own sample
  rate. Chunks are queued until the server is ready (at most 5 s).
- **File mode:** the browser plays the file locally while the server analyzes it, and the page
  shows the newest event at or before the playback position. Seeking works the same way. If
  analysis falls behind playback, the page says so.

## 2026-11-14: README and from-scratch test (step 9)

- **The README was tested by following it word for word** in a fresh clone, with its own empty
  model cache:
  - The clone has 50 files and no `data/`.
  - `uv venv` and `uv pip install --torch-backend cpu` worked.
  - `scripts/fetch_model.py` downloaded both pinned models in 5 min 44 s. This was the script's
    first run against an empty cache; step 2 had used a scratch script.
  - 187 fast tests and 4 real-model tests passed.
  - `scripts/serve.py` started, the page, scripts and zones were served, and an upload and a
    real-time live stream worked: no skipped windows, median delay 0.73 s.
  - Not tested: the plain-pip alternative in the README.
- **The startup timing is noisy.** It uses 3 runs. That startup measured 910 ms per window
  (earlier startups: 777–821 ms), so it chose 1.25 s instead of 1.0 s. The README now says
  "1.0 s, or 1.25 s if the computer was busy" and points to `--update-seconds 1.0`. Using more
  runs would cost about 1 s of startup per extra run; left as is for now.
- **Large scratch work goes on the home disk.** `/tmp` here is a 3.4 GB in-memory tmpfs. The first
  try of this test filled it (venv plus 2.5 GB of models) and was deleted with the user's
  permission.

## 2026-11-22: 10 s window

- **The window is now 10 s instead of 5 s.** The user found the map changed too fast.
  CLAUDE.md was updated with permission ("over the last ~10 s of audio").
- **Measured on the Ryzen 7 7730U, 8 threads, median of 15 runs:**

  | Window | Time per update |
  |---|---|
  | 5 s | 808 ms |
  | 7.5 s | 1185 ms |
  | 10 s | 1585 ms |

  Inference time is linear in the window length. With a 10 s window the server updates every
  2.0–2.25 s (the target is still 0.5 s), and a live stream takes about 1.2–1.7 s from audio to
  map.
- **Effect,** checked on the 4 sample clips joined into 26 s of audio:
  - The 10 s window gives half as many updates, steadier percentages, and 3 flips of the top
    zone instead of 4.
  - The price is slower switching: a new speaker needs roughly 5–10 s to take over the map. The
    4.2 s Spain clip at the end ends "uncertain, Mexico 38 %" because the window still holds 6 s
    of the Mexico clip.
  - "No speech" now appears only after about 9 s of silence, since the window must hold under
    0.75 s of sound.
  - Smoothing is unchanged (τ = 1.5 s), and so is the 2 s "listening" threshold.
- **The startup timing discards 2 warm-up runs instead of 1.** Right after loading, the first
  runs take about 1.8 s and then settle at about 1.65 s. Startups measured 1758–1926 ms
  (2.25–2.5 s updates) and the benchmark 1585 ms. This machine sits right at the 1667 ms
  boundary for 2.0 s updates, so the interval can differ between startups;
  `--update-seconds 2.0` fixes it. The 1.2× safety margin is kept, so file analysis stays ahead
  of playback.
- **The benchmark joins all sample clips** (about 26 s), because the longest single clip
  (8.7 s) is shorter than a 10 s window. Default windows: 1, 2.5, 5, 7.5, 10 s.

## 2026-11-22: File mode follows the playhead

- **Problem:** the file was analyzed in one pass from the start. After a skip forward the page
  said "N s behind", because the model first had to work through everything in between.
- **Upload once, then analyze from any position.**
  - `POST /api/files` decodes the file and returns an id, the duration and the update interval.
    `GET /api/files/{id}/events?start=S` streams results from the last window ending at or
    before S.
  - `POST /api/analyze` (the whole file from 0) stays for scripts and curl.
  - Windows are cut and numbered the same way in every pass, as in live mode, so the page caches
    results by window number.
  - The server keeps the last 4 files, decoded at 16 kHz (a 15 min file takes about 58 MB). An
    older id gets "Unknown or expired file. Choose it again."
- **After a skip, smoothing starts fresh.** The first result is the model's output for the full
  10 s window there. The speech count runs from the start of the file, so a skip doesn't bring
  back "Listening…".
- **The server stops a pass when the page leaves it.** It checks for a disconnect before each
  window; a test with real uvicorn confirms this. The inference already running can't be
  interrupted: it finishes and its result is dropped. So the first result after a skip takes up
  to 2 inferences: 1.8–3.6 s here.
- **Results must reach the page before playback gets to them.** Each result is shown for one
  update (+0.3 s) after its window ends.
  - My first version started the new pass at the playhead. With 2.0 s updates and 1.8 s
    inferences that pass gains only 0.2 s per update, so the map would have shown "Analyzing…"
    for many seconds after each skip. The real-server check found this; the unit tests had not
    let the playhead move while waiting.
  - **Now, while playing:** a new pass starts at the first window that ends after the moment its
    first result can arrive: the rest of the running inference plus one more, timed from the
    events' `infer_ms`. Later results come faster than playback, so they stay ahead.
  - **While paused:** the pass starts at the playhead, so the result for that moment appears
    while paused.
  - **Look-ahead:** while an already analyzed stretch plays, the page checks whether the running
    pass will deliver the window after it in time. If not, it starts a new pass there at once.
  - **No restart loops:** a pass still waiting for its first result is replaced only after a
    seek or play. A pass that left out a window it couldn't finish in time is kept, unless a new
    pass would fill the gap sooner.
- **Verified:**
  - **Unit tests, plus a playback simulation.** The simulation runs `logic.js` in QuickJS
    against a simulated one-worker server, with 2.0 s/1.8 s, 2.25 s/1.8 s and 0.5 s/0.35 s
    (update/inference).
    - Each skip starts exactly one pass, and there are no gaps after the first result.
    - The wait is at most 2 inferences + 1 update.
    - With the look-ahead switched off, the same simulation has a 0.7–3.7 s gap, so the test
      catches its loss.
  - **The real server** (77.5 s file, 2.0 s updates, 1.75 s inferences), driven by the same
    `logic.js` with a simulated playhead:
    - After a skip ahead or back into an unanalyzed part, the map was back after 4.0 s, with no
      gaps after that.
    - A skip into a part analyzed earlier was seamless: the next pass started at once.
    - Pausing exactly at the end of an analyzed stretch and pressing play left a 3.7 s gap,
      because nothing can be ready sooner.
  - In a real browser (user, 2026-11-22): a skip ahead while playing, a skip back to an analyzed
    part, and a skip while paused all worked as described, with no flicker and no long waits.
- **Trade-off:** a pass runs to the end of the file, even over windows already cached. The page
  ignores the duplicates. Stopping earlier would need an end position in the API; not worth it
  yet.

## 2026-11-22: Stage 2, step 1: what Common Voice Spanish contains

- **All work on our own model is on the branch `stage2-own-model`** (the user's wish), not on main.
- **Data:** Common Voice 27.0 Spanish (`cv-corpus-27.0-2026-09-11-es.tar.gz`, 51.9 GB). The user
  downloaded it from Mozilla Data Collective, the only source since October 2025, and put it in
  `data/commonvoice/` (not committed). SHA-256
  `b7c1f303ebe3c4a637afdd3d31c5aaa3da131a893ce77169db182068f342e4c1`, computed by us. Mozilla
  showed no checksum to compare it with.
- **Reading the archive safely:** `scripts/cv_extract.py` reads it in one pass of about 4.5 min.
  - It computes the checksum and lists all 1,683,749 members.
  - It extracts only the 11 metadata files: TSV tables and release notes.
  - Only regular files with safe paths are written. Links, absolute paths and ".." are refused,
    which a test checks.
  - Audio is extracted later, and only the clips we need.
  - Scripts that read the downloaded files run with `python -I`.
- **Counting:** `scripts/cv_inventory.py` counts only validated clips (440,794 rows). The
  release's own README counts all 1.68 million clips, including 1.15 million that nobody has
  validated yet.
  - A speaker is one `client_id`. A speaker counts for an accent only if every labelled clip
    names that one accent: 4,328 speakers (296,643 clips).
  - 69 speakers named several accents (678 clips), and 13,291 named none (143,473 clips).
  - The inventory prints no speaker ids, and its JSON stays in `data/`.
- **The 10 regional accents** (validated clips, one-accent speakers):

  | Accent | Speakers | Hours | h, ≤ 50 clips/speaker | Top speaker's share |
  |---|---|---|---|---|
  | México | 1,083 | 194.3 | 27.5 | 8 % |
  | Andino-Pacífico | 693 | 26.3 | 15.3 | 12 % |
  | España: Norte peninsular | 463 | 57.9 | 11.1 | 47 % |
  | Caribe | 419 | 15.8 | 8.8 | 14 % |
  | Rioplatense | 415 | 20.5 | 9.2 | 22 % |
  | España: Centro-Sur peninsular | 348 | 15.9 | 8.9 | 10 % |
  | América central | 287 | 10.8 | 6.4 | 9 % |
  | Chileno | 216 | 8.6 | 4.7 | 15 % |
  | España: Sur peninsular | 186 | 58.1 | 5.2 | 83 % |
  | España: Islas Canarias | 167 | 3.9 | 2.5 | 21 % |

  - The 11th option, "Español de Filipinas", has 3 speakers. 42 free-text labels have 1–3
    speakers each (for example "España: Catalunya").
- **What this means:**
  - Every regional accent has enough speakers for a speaker-disjoint test set.
  - Raw hours are dominated by a few speakers. In southern Spain one speaker made 83 % of the
    clips, so we must cap clips per speaker.
  - With a cap of 50, the usable audio is about 100 h, from 2.5 h (Canaries) to 27.5 h (Mexico).
  - Clips are short: 4.8–5.6 s on average.
  - Labelled speakers are mostly male (roughly 70 %, where gender is given), so results should
    also be reported by gender.

## 2026-11-22: Stage 2, step 2: speaker-disjoint split (10 accents)

- **The user's choices:** the 10 regional accents, at most 50 clips per speaker, and validated
  clips only.
  - Left out: "Español de Filipinas" (3 speakers), the free-text accents, and speakers who
    named several accents.
- **The settings are in `training/cv27_es10/dataset.yaml`** (committed). The spec pins the
  archive's SHA-256, and `scripts/cv_split.py` refuses to run if the extracted data came from a
  different archive.
- **Rules,** for each accent separately:
  - Test and dev speakers come only from speakers with at least 5 clips (about 25 s), so each
    can fill at least two 10 s test windows.
  - Test takes 15 % of those speakers (at least 20), dev 10 % (at least 10). Fractions give
    the big classes more reliable numbers, and the minimums keep the small ones testable.
  - Every other speaker trains, and each speaker counts for at most 50 clips.
  - "Random" means ordered by SHA-256 of a fixed seed and the speaker id (or clip name). The
    split is the same on every run whatever the row order, which a test checks.
- **Sentences are kept out of training too.** A training clip whose sentence is also read in dev
  or test is dropped before the cap.
  - Without this, a model could score on test by recognizing sentence content instead of the
    accent.
  - Cost: 2.8 h of training audio (73.4 h instead of 76.2 h). 175 training speakers read
    only held-out sentences and drop out entirely.
  - The script checks that no speaker, and no held-out sentence, is in training.
- **Result** (speakers / hours):

  | Class | Train | Dev | Test (f/m) |
  |---|---|---|---|
  | es_north | 378 / 8.0 | 30 / 1.2 | 46 / 1.5 (9/34) |
  | es_centre_south | 278 / 6.7 | 25 / 0.8 | 37 / 1.1 (7/29) |
  | es_south | 145 / 3.8 | 14 / 0.4 | 22 / 0.7 (7/15) |
  | es_canary | 137 / 1.7 | 10 / 0.2 | 20 / 0.5 (7/12) |
  | mexico | 850 / 20.5 | 73 / 2.2 | 110 / 4.2 (34/72) |
  | central_america | 226 / 4.9 | 19 / 0.5 | 28 / 0.8 (12/15) |
  | caribbean | 336 / 6.3 | 26 / 0.8 | 39 / 1.4 (12/24) |
  | andean_pacific | 546 / 11.2 | 44 / 1.5 | 66 / 2.2 (16/48) |
  | rioplatense | 320 / 6.6 | 29 / 0.9 | 43 / 1.4 (14/26) |
  | chilean | 168 / 3.6 | 15 / 0.4 | 22 / 0.6 (5/15) |
  | total | 3,384 / 73.4 | 285 / 9.1 | 433 / 14.3 |

  - Test gender: f/m counts only speakers who gave a gender.
  - Training data is uneven: 20.5 h for Mexico against 1.7 h for the Canaries. Training will
    need class-balanced sampling or loss weights.
- **The lists stay in `data/commonvoice/splits/cv27_es10/`** (not committed), because they hold
  speaker ids. The Common Voice terms forbid sharing the data. The spec plus the script remake
  them exactly.
