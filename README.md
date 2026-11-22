# Accent Map

Guesses the regional accent of Spanish speech, from your microphone or an audio file,
and shows the probabilities on a map of dialect zones that updates while you listen.

Everything runs locally: a small Python server runs the model on your CPU, and the page
runs in your browser. Nothing is sent anywhere else.

## Quick start

You need:
- Linux, macOS or Windows with WSL2. It was developed on WSL2/Ubuntu.
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). uv
  installs Python 3.12 itself if you don't have it.
- About 4 GB of free disk and 4 GB of free RAM. No GPU needed.
- An internet connection for the first setup only. No account or login.

```bash
git clone <this repository> accent-map
cd accent-map

# 1. Python environment with pinned versions (CPU-only PyTorch)
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt --torch-backend cpu

# 2. Download the model at its pinned revision into data/hf_cache/ (about 2.5 GB, once)
.venv/bin/python scripts/fetch_model.py models/commonaccent_es6

# 3. Start the server: it loads the model, times one update and prints "Ready" (about 15 s)
.venv/bin/python scripts/serve.py
```

Then open **http://localhost:8000** in Chrome, Edge or Firefox. If you use WSL2, open it in
your Windows browser. Press Ctrl+C in the terminal to stop the server.

<details>
<summary>Without uv (plain pip)</summary>

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
The extra index is where the pinned `torch==2.11.0+cpu` and `torchaudio==2.11.0+cpu` come from.
</details>

## Using it

- **Start microphone**, allow access, and speak Spanish. The level bar shows that sound is
  arriving.
- **Analyze a file…** accepts WAV, FLAC, OGG and MP3, but not M4A/AAC: convert those first.
  Press play, and the map follows the audio as it plays.
  - The model analyzes the part you are listening to. Parts it has already analyzed show at once.
  - After you skip to a new part, the map says "Analyzing this part…" for about 4 s (2–6 s) on
    the laptop below, then follows the audio again. Paused, the result for the new position
    appears after about 2–4 s.

The map colours each zone by how likely it is, smoothed over about 1.5 s so it moves instead
of flickering. The model looks at the last 10 s of audio, so each result rests on several
sentences and changes calmly. The status above the map says how far to trust it:

| Status | Meaning |
|---|---|
| Listening… | Less than 2 s of speech heard so far. Early guesses are shown faintly. |
| No speech | Less than 0.75 s of sound in the last 10 s. The last result is kept, dimmed. |
| Uncertain | No clear winner: the top zone is under 50 %, or less than 15 points ahead of the second. |
| Confident | The top zone is at least 50 % and clearly ahead. |

**Speed:** the server updates as often as your CPU allows. It aims for every 0.5 s. At startup
it times one update and picks a slower interval if needed.
- On an 8-core laptop CPU (Ryzen 7 7730U), one update of the 10 s window takes about 1.6 s. The
  server picks 2.0 s, or a little more if the computer was busy during that startup measurement.
- The map then reacts about 1.7 s after you speak.
- The startup log prints the interval it chose. `--update-seconds 2.0` fixes it.

## What the stage-1 model can and can't do

The model is [`Jzuluaga/accent-id-commonaccent_xlsr-es-spanish`](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-es-spanish)
(MIT license), a wav2vec2 XLSR-53 model fine-tuned on Common Voice. Its 6 classes are
dialect zones, not countries:

| Zone | Covers |
|---|---|
| Southern Spain | Andalusia, Extremadura, Murcia (**not** Madrid or northern Spain) |
| Mexico | Mexico except Veracruz, Tabasco and Quintana Roo |
| Andean-Pacific | Colombia, Peru, Ecuador, western Bolivia, Andean Venezuela |
| Caribbean | Cuba, Puerto Rico, Dominican Republic, Venezuela, Panama, Caribbean Colombia, Veracruz, Tabasco, Quintana Roo |
| Rioplatense | Argentina, Uruguay, Paraguay, eastern Bolivia |
| Chilean | Chile and Argentina's Cuyo region |

- The model reports 68.5 % accuracy on Common Voice read speech. Expect less on your own
  recordings. Treat 68.5 % as an upper bound.
- A speaker from outside these zones, for example Madrid or Central America, is still
  assigned to one of the six. Those areas are hatched on the map as "not covered".
- The zone shapes are approximate dialect areas made from administrative regions, not borders.
  `models/commonaccent_es6/zones_spec.yaml` lists which region goes where.
- **The published model config is broken, and this repo fixes it.** The `hyperparams.yaml` on the
  model page leaves out part of the trained classifier, so the model card's example gives random
  predictions. `models/commonaccent_es6/hyperparams.yaml` contains the fix. The weights still come
  unchanged from the pinned revision. See [DECISIONS.md](DECISIONS.md).

## Troubleshooting

- **The page doesn't load from Windows (WSL2).** Start the server with
  `--host 0.0.0.0`. Note that this also makes it reachable from your local network.
- **"Microphone access was blocked".** Allow the microphone in the browser's address bar.
  Browsers only allow microphones on `https://` or `http://localhost`, so use `localhost`, not
  the machine's IP address.
- **`No module named ...` after running `uv sync`.** Don't use `uv sync` here: this repo pins its
  packages in `requirements.txt`, so `uv sync` finds no dependencies and removes everything from
  `.venv`. Run the two install commands of step 1 again. (`uv run` is safe: `pyproject.toml` tells
  uv not to manage this environment.)
- **`ModelFilesMissing` at startup.** Run step 2, `scripts/fetch_model.py`.
- **Killed while loading, or out of memory.** Loading needs about 2.8 GB of RAM. On WSL2, raise
  the limit with `memory=8GB` under `[wsl2]` in `%UserProfile%\.wslconfig`, then run `wsl --shutdown`.
- **Updates are slow.** That's the CPU. `--update-seconds 2` forces a slower, steady pace.

## Development

```bash
uv pip install --python .venv -r requirements-dev.txt
.venv/bin/python -m pytest              # fast tests, fake model, about 5 s
.venv/bin/python -m pytest -m model     # tests with the real model, about 20 s
```

There is no browser in the test suite: `tests/test_web.py` runs the page's logic in QuickJS.
The microphone and the look of the page are checked by hand.

Useful scripts (each prints its usage with `--help` or no arguments):

| Script | What it does |
|---|---|
| `scripts/serve.py` | Start the server (`--model`, `--host`, `--port`, `--update-seconds`). |
| `scripts/fetch_model.py <model dir>` | Download a model's files at the pinned revisions. |
| `scripts/check_model.py <model dir> [files…]` | Classify whole files, or the model's sample clips. |
| `scripts/bench_latency.py` | Time one update for window lengths 1–10 s and 4/8/16 threads. |
| `scripts/stream_wav.py <file>` | Play a file into a running server like a microphone, and print the delays. |
| `scripts/build_zones.py <model dir>` | Rebuild the map zones from Natural Earth (download command in the script). |

`DECISIONS.md` records each choice and why.

### Adding a model

The app learns everything about a model from `models/<name>/config.yaml`: display name,
source and pinned revision, the class list (id, label, description, map zone), the audio
window, the map panels and which backend runs it. Nothing about the classes is hardcoded, so a
model with a different number of classes works without code changes.

- Each class's `model_label` must match the label in the model's own files. The app reads the
  class order from the model and refuses to start if the labels don't match.
- A backend implements `predict(16 kHz mono audio) -> {class_id: probability}` (see
  `src/server/model_api.py`). Stage 1 has `speechbrain_custom_classifier` and `fake`, which
  is used for tests.
- Start another model with `scripts/serve.py --model models/<name>`.

## Layout

```
models/<name>/   one folder per model: config.yaml, map zones, backend files
src/server/      Python server: model plug-ins, audio windowing, smoothing, web API
web/             the browser page (plain HTML/CSS/JS, no build step)
scripts/         setup, benchmarks and tools (table above)
tests/           pytest
data/            downloads and your own recordings (not committed)
```

## Credits and licenses

- Model: Jzuluaga/accent-id-commonaccent_xlsr-es-spanish (MIT), from the CommonAccent paper
  (Zuluaga-Gomez et al., 2023, arXiv:2305.18283).
- Base model: facebook/wav2vec2-large-xlsr-53 (Apache-2.0).
- Map data: made with [Natural Earth](https://www.naturalearthdata.com/) (public domain).
- Training our own model (branch `stage2-own-model`): speech from Mozilla Common Voice 27.0 Spanish, via
  Mozilla Data Collective under its Data Consumer Terms. Room echo and noise from the
  [Room Impulse Response and Noise Database](https://www.openslr.org/28/) (OpenSLR 28, Apache 2.0; real
  room recordings from RWCP, REVERB 2014 and AIR; noise sounds from the public-domain Freesound part of
  MUSAN).
