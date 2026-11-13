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
- **Still `httpx` for the test client.** Starlette 1.10 recommends `httpx2`, but the swap was
  blocked by a permission check. The only effect is a deprecation warning in the tests.

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
