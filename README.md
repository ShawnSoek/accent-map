# Accent Map

Guesses the regional accent of Spanish speech, from your microphone or an audio file, and
shows the probabilities on a map of dialect zones that updates while you listen.

Everything runs locally: a small Python server runs the model on your CPU, and the page runs
in your browser. Nothing is sent anywhere else.

**Status:** finished in November 2026 and no longer developed. Everything below works as
described. [What could come next](#what-could-come-next) lists the open ends.

## Quick start

You need:
- Linux, macOS or Windows with WSL2. It was developed on WSL2/Ubuntu.
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). uv
  installs Python 3.12 itself if you don't have it.
- About 4 GB of free disk and 3 GB of free RAM. No GPU needed.
- An internet connection for the first setup only. No account or login.

```bash
git clone <this repository> accent-map
cd accent-map

# 1. Python environment with pinned versions (CPU-only PyTorch)
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt --torch-backend cpu

# 2. Download the base model (wav2vec2 XLSR-53, 1.3 GB, once) at its pinned revision into data/hf_cache/
.venv/bin/python scripts/fetch_model.py models/cv27_es10

# 3. Start the server: it loads the model, times one update and prints "Ready" (about 15 s)
.venv/bin/python scripts/serve.py
```

Then open **http://localhost:8000** in Chrome, Edge or Firefox. If you use WSL2, open it in
your Windows browser. Press Ctrl+C in the terminal to stop the server.

This starts the 10-zone model. To try the older 6-zone model instead (see [The two
models](#the-two-models)):

```bash
.venv/bin/python scripts/fetch_model.py models/commonaccent_es6     # 1.3 GB more
.venv/bin/python scripts/serve.py --model models/commonaccent_es6
```

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
  - After you skip to a new part, the map says "Analyzing this part…" for a few seconds (2–6 s
    on the laptop below), then follows the audio again. Paused, the result for the new
    position appears after 2–4 s.
  - The line under the player shows how much is analyzed and how often the map updates.

The map colours each zone by how likely it is, smoothed over about 1.5 s so it moves instead
of flickering. Hover over a zone for its name, what it covers and its probability. The model
looks at the last 10 s of audio, so each result rests on several sentences and changes
calmly. The status above the map says how far to trust it:

| Status | Meaning |
|---|---|
| Listening… | Less than 2 s of speech heard so far. Early guesses are shown faintly. |
| No speech | Less than 0.75 s of sound in the last 10 s. The last result is kept, dimmed. |
| Uncertain | No clear winner: the top zone is under 50 %, or less than 15 points ahead of the second. |
| Confident | The top zone is at least 50 % and clearly ahead. |

For the 10-zone model, on Common Voice dev speakers, "Confident" was right about 63 % of the
time and "Uncertain" about 37 %.

**Speed:** the server updates as often as your CPU allows. It aims for every 0.5 s. At startup
it times one update and picks a slower interval if needed.
- On an 8-core laptop CPU (Ryzen 7 7730U), one update on a 10 s window takes 1.4–1.5 s with
  the 10-zone model, so the map updates every 1.75–2 s. The 6-zone model takes about 1.7 s.
- The map changes within about 1 s after a window ends (0.6–0.9 s measured).
- A computer can be slower later than at startup, for example with the browser playing and
  other programs busy. The server keeps timing the model, and each new file or microphone
  session gets an interval that fits. If a file that is playing falls behind, the page asks for
  a longer interval for that file once, keeps the results it has, and continues.
- The terminal prints the interval chosen at startup and each change. `--update-seconds 2.0`
  fixes it.

## The two models

| | **10-zone model** (default) | 6-zone model |
|---|---|---|
| Folder | `models/cv27_es10` | `models/commonaccent_es6` |
| Zones | all 10 Common Voice accent zones | 6 (no northern/central Spain, Canaries, Central America) |
| Made by | this project: the frozen base model XLSR-53 (layer 17) plus a linear classifier trained on Common Voice 27.0 | [CommonAccent](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-es-spanish) (Zuluaga-Gomez et al., 2023): XLSR-53 fine-tuned on Common Voice 11 |
| Accuracy on Common Voice | 48 % of 10 s windows (95 % interval 44–52 %; chance 10 %), on 433 speakers kept out of training, each zone counted equally | 68.5 % as reported, on 6 zones, with speakers not kept apart between training and test (so likely optimistic) |
| One update (10 s, laptop CPU) | about 1.4 s | about 1.7 s |
| Memory | about 1.5 GB | about 2.7 GB |

The two accuracy numbers can't be compared: different zones, different counting, and only the
10-zone model was tested on speakers it had never heard. Both were measured on read speech
(Common Voice). Expect less on your own recordings: other microphones, free speech, and
speakers whose accent doesn't match a zone. Neither model was tested on such recordings.

### 10-zone model

| Zone | Covers | Right (10 s, test) |
|---|---|---|
| Northern Spain | Asturias, Castile and León, Cantabria, Basque Country, Navarre, Aragon, La Rioja, Guadalajara, Cuenca | 29 % |
| Central Spain | Madrid, the rest of Castilla-La Mancha | 51 % |
| Southern Spain | Andalusia, Extremadura, Murcia | 33 % |
| Canary Islands | | 60 % |
| Mexico | Mexico except Veracruz, Tabasco and Quintana Roo | 49 % |
| Central America | Guatemala, Honduras, El Salvador, Nicaragua, Costa Rica | 53 % |
| Caribbean | Cuba, Puerto Rico, Dominican Republic, Venezuela, Panama, Caribbean Colombia, Veracruz, Tabasco, Quintana Roo | 41 % |
| Andean-Pacific | Colombia, Peru, Ecuador, western Bolivia, Andean Venezuela | 38 % |
| Rioplatense | Argentina, Uruguay, Paraguay, eastern Bolivia | 66 % |
| Chilean | Chile and Argentina's Cuyo region | 61 % |

- **Most mistakes are between neighbours.** Spain vs. the Americas is right 92 % of the time.
  Northern Spain is often taken for central Spain. The Caribbean, Andean and Central American
  zones mix, and so do Mexico and Central America.
- **Use the whole map, not only the top zone.** On dev speakers, the right zone was among the
  top 2 in 69 % of 10 s windows and among the top 3 in 83 %. The probabilities are calibrated:
  across many windows, an 80 % answer is right about 80 % of the time.
- **Longer listening helps up to about 40 s.** On dev speakers, 20 s of a speaker gave 52 % right
  and 40 s gave 54 %; more didn't help. The remaining mistakes are systematic for a speaker, and
  some are probably wrong labels: Common Voice speakers choose their accent themselves.
- Galicia, Catalonia, Valencia, the Balearic Islands, Ceuta and Melilla have no accent option in
  Common Voice, so no zone. They are hatched on the map as "not covered". Speakers from there,
  or from anywhere else, are still assigned to one of the ten zones.

### 6-zone model

| Zone | Covers |
|---|---|
| Southern Spain | Andalusia, Extremadura, Murcia (**not** Madrid or northern Spain) |
| Mexico | Mexico except Veracruz, Tabasco and Quintana Roo |
| Andean-Pacific | Colombia, Peru, Ecuador, western Bolivia, Andean Venezuela |
| Caribbean | Cuba, Puerto Rico, Dominican Republic, Venezuela, Panama, Caribbean Colombia, Veracruz, Tabasco, Quintana Roo |
| Rioplatense | Argentina, Uruguay, Paraguay, eastern Bolivia |
| Chilean | Chile and Argentina's Cuyo region |

- A speaker from outside these zones, for example Madrid or Central America, is still assigned
  to one of the six. Those areas are hatched on the map as "not covered".
- **The published model config is broken, and this repo fixes it.** The `hyperparams.yaml` on the
  model page leaves out part of the trained classifier, so the model card's example gives random
  predictions. `models/commonaccent_es6/hyperparams.yaml` contains the fix. The weights still come
  unchanged from the pinned revision.
- The original authors left several of Common Voice's accent options out in their preparation
  script, without giving a reason. That's why it has 6 zones.

For both models, the zone shapes are approximate dialect areas made from administrative
regions, not borders. `models/<name>/zones_spec.yaml` lists which region goes where.

## How the 10-zone model was made

In short (every choice and its reason is in [DECISIONS.md](DECISIONS.md), "Stage 2"):

1. **Data:** Common Voice 27.0 Spanish, validated clips of speakers who chose one of the 10
   regional accents. That's 3,384 training speakers (73 h), 285 dev and 433 test.
   - Speakers are kept apart between training, dev and test, with at most 50 clips per speaker.
   - Sentences read in dev or test were removed from training, so the model can't score by
     recognizing sentences.
2. **Training copies:** each training clip also got a noisy copy (room echo, background noise,
   laptop-microphone filters, volume), from the OpenSLR 28 collection. Dev and test stay clean.
3. **Features:** the frozen base model XLSR-53 turned every clip into the mean and spread of
   each of its 25 layers.
   - That's 120,000 pieces of audio, including 10 s pairs of test clips, which match the app's
     window. It took 2 h on two Kaggle GPUs; this CPU would have needed about 45 h.
   - The GPU's results were checked against the laptop's before and after.
4. **Classifier:** a probe of each layer showed that the accent sits in the middle layers
   (7–18). Layer 17 with a plain linear classifier was best on dev; bigger classifiers only
   memorized the training speakers.
   - One temperature, fitted on dev, makes the probabilities honest.
   - Test was evaluated once, at the end.
5. **In the app:** the app computes exactly what was evaluated (checked on dev clips within
   0.002). It runs the base model only up to layer 17, which gives the same values and is faster.

### Reproducing it

The classifier (`models/cv27_es10/head.safetensors`, 116 KB) is in the repo, so this is only
needed to retrain or change the model. You need about 80 GB of disk, an account at Mozilla
Data Collective (where Common Voice is published since October 2025), and a
[Kaggle](https://www.kaggle.com/) account for the GPU step (free, 30 GPU hours a week; this run
used about 5).

```bash
# Data (download by hand, accepting each source's terms):
#   data/commonvoice/cv-corpus-27.0-2026-09-11-es.tar.gz   Common Voice 27.0 Spanish, 51.9 GB
#   data/noise/rirs_noises.zip                             https://www.openslr.org/28/, 1.3 GB
# The expected SHA-256 of both is in training/cv27_es10/*.yaml; the scripts refuse other files.
.venv/bin/python -I scripts/cv_extract.py data/commonvoice/cv-corpus-27.0-2026-09-11-es.tar.gz
.venv/bin/python -I scripts/cv_inventory.py data/commonvoice/cv-corpus-27.0-2026-09-11-es   # optional overview
.venv/bin/python -I scripts/cv_split.py training/cv27_es10/dataset.yaml
.venv/bin/python -I scripts/cv_extract.py data/commonvoice/cv-corpus-27.0-2026-09-11-es.tar.gz --clips data/commonvoice/splits/cv27_es10
.venv/bin/python -I scripts/cv_check_clips.py training/cv27_es10/dataset.yaml
.venv/bin/python -I scripts/noise_extract.py data/noise/rirs_noises.zip

# Features on a Kaggle GPU (the bundle holds only clips and item ids, no speaker information)
.venv/bin/python -I scripts/kaggle_bundle.py build training/cv27_es10
.venv/bin/python -I scripts/kaggle_bundle.py meta training/cv27_es10 --user <your kaggle username>
uv tool install kaggle==2.2.5 --with six
kaggle auth login
kaggle datasets create -p data/kaggle/cv27_es10/upload        # a private dataset; wait until it's ready
kaggle kernels push -p data/kaggle/cv27_es10/job --accelerator NvidiaTeslaT4 -t 21600
kaggle kernels status <your kaggle username>/accent-map-features-cv27-es10    # queued, running, complete
kaggle kernels output <your kaggle username>/accent-map-features-cv27-es10 -p data/features/cv27_es10 --page-size 200
mv data/features/cv27_es10/features/* data/features/cv27_es10/ && rmdir data/features/cv27_es10/features
.venv/bin/python -I scripts/check_features.py training/cv27_es10

# Classifier, on the laptop (about 15 min)
.venv/bin/python -I scripts/train_probe.py training/cv27_es10
.venv/bin/python -I scripts/train_head.py training/cv27_es10
.venv/bin/python -I scripts/evaluate_head.py training/cv27_es10          # test: once
.venv/bin/python -I scripts/export_head.py training/cv27_es10 models/cv27_es10
# then copy the printed head_sha256 into models/cv27_es10/config.yaml
```

Notes:
- **The Kaggle run waited almost 4 h in the queue and then ran for 5 h.** 3 h of that went into
  Kaggle reading 64,000 small files for the first time. Uploading the data as one large file
  that Kaggle doesn't unpack would likely avoid it.
- **Kaggle unpacks uploaded archives,** including archives inside archives. The job script
  handles that and checks every file against its checksum.
- **The split is exactly reproducible** (fixed seeds), and so are the noisy copies. The
  classifier can come out slightly different.
- Common Voice's terms forbid sharing the data and trying to identify speakers. Everything
  derived from it stays in `data/`, which is never committed. The repo holds only the
  classifier.

## What could come next

- **Test on your own recordings.** This is the most useful next step, because both models
  were only measured on read Common Voice speech. `data/own_tests/<zone id>/` (for example
  `data/own_tests/mexico/`) is the agreed place: for evaluation only, never for training or
  tuning. A script to score both models on that folder isn't written yet.
- **Better accuracy,** with rough estimates of the gain on the 10 s test windows (48 % now):

  | Idea | Expected gain | Cost |
  |---|---|---|
  | Fine-tune XLSR-53 itself (cut after layer 18), as CommonAccent did, on a Kaggle GPU | +6 to +12 points | most of a week's Kaggle GPU quota, and a training job that saves and resumes |
  | A classifier that attends to moments in time instead of averaging them | +3 to +6 | a few GPU hours |
  | A verdict over the whole recording next to the live map | +4 to +6, measured on dev | small; app only |
  | More speakers: free-text Common Voice accents, a second corpus | +1 to +2 | data work; a second corpus needs evaluation by source |

## Troubleshooting

- **The page doesn't load from Windows (WSL2).** Start the server with `--host 0.0.0.0`. Note
  that this also makes it reachable from your local network.
- **"Microphone access was blocked".** Allow the microphone in the browser's address bar.
  Browsers only allow microphones on `https://` or `http://localhost`, so use `localhost`, not
  the machine's IP address.
- **`No module named ...` after running `uv sync`.** Don't use `uv sync` here: this repo pins its
  packages in `requirements.txt`, so `uv sync` finds no dependencies and removes everything from
  `.venv`. Run the two install commands of step 1 again. (`uv run` is safe: `pyproject.toml` tells
  uv not to manage this environment.)
- **`ModelFilesMissing` at startup.** Run step 2, `scripts/fetch_model.py` with the same model folder.
- **Killed while loading, or out of memory.** The 10-zone model needs about 1.5 GB of RAM, the
  6-zone model about 2.8 GB. On WSL2, raise the limit with `memory=8GB` under `[wsl2]` in
  `%UserProfile%\.wslconfig`, then run `wsl --shutdown`.
- **Updates are slow.** That's the CPU. `--update-seconds 3` forces a slower, steady pace.
- **"Analyzing this part…" keeps coming back while a file plays, without skipping.** The
  computer is slower than the update interval allows. The page fixes this itself within a few
  seconds by asking for a longer interval; the line under the player shows the new one. If it
  doesn't, restart the server with a larger `--update-seconds`.

## Development

```bash
uv pip install --python .venv -r requirements-dev.txt
.venv/bin/python -m pytest              # fast tests with a fake model, about 20 s
.venv/bin/python -m pytest -m model     # tests with the real models, about 30 s (need the downloads)
```

There is no browser in the test suite: `tests/test_web.py` runs the page's logic in QuickJS,
including a simulated playback against a simulated server. The microphone and the look of the
page were checked by hand.

`DECISIONS.md` records each choice and why, in order. `CLAUDE.md` holds the project's working
rules.

Scripts (each prints its usage with `--help` or no arguments):

| Script | What it does |
|---|---|
| `scripts/serve.py` | Start the server (`--model`, `--host`, `--port`, `--update-seconds`). |
| `scripts/fetch_model.py <model dir>` | Download a model's files at the pinned revisions. |
| `scripts/check_model.py <model dir> [files…]` | Classify whole files (default: the 6-zone model repo's 4 sample clips). |
| `scripts/bench_latency.py [model dir]` | Time one update for several window lengths and thread counts. |
| `scripts/stream_wav.py <file>` | Play a file into a running server like a microphone, and print the delays. |
| `scripts/build_zones.py <model dir>` | Rebuild the map zones from Natural Earth (download command in the script). |
| `scripts/cv_*.py`, `noise_extract.py`, `augment_preview.py` | Data preparation for training (see above). |
| `scripts/kaggle_bundle.py`, `kaggle_run.py`, `check_features.py` | The feature run on Kaggle and its checks. |
| `scripts/train_probe.py`, `train_head.py`, `evaluate_head.py`, `export_head.py` | Classifier training, evaluation and export. |

### Adding a model

The app learns everything about a model from `models/<name>/config.yaml`: display name,
source and pinned revision, the class list (id, label, description, map zone), the audio
window, the map panels and which backend runs it. Nothing about the classes is hardcoded, so a
model with a different number of classes works without code changes.

- Each class's `model_label` must match the label in the model's own files. The app reads the
  class order from the model and refuses to start if the labels don't match.
- A backend implements `predict(16 kHz mono audio) -> {class_id: probability}` (see
  `src/server/model_api.py`). There are three:
  - `xlsr_pooled_head` (10-zone model): the base model plus a classifier file pinned by its SHA-256
  - `speechbrain_custom_classifier` (6-zone model)
  - `fake`, used for tests
- Start another model with `scripts/serve.py --model models/<name>`.

## Layout

```
models/<name>/     one folder per model: config.yaml, map zones, classifier or backend files
src/server/        Python server: model plug-ins, audio windowing, smoothing, web API
src/training/      the 10-zone model's data split, noisy copies, features, classifier, measures
training/<name>/   settings of a training run (data, noise, features), pinned by checksums
web/               the browser page (plain HTML/CSS/JS, no build step)
scripts/           setup, tools and the training pipeline (table above)
tests/             pytest
data/              downloads, training data and results, your own recordings (never committed)
```

## Credits and licenses

- 6-zone model: Jzuluaga/accent-id-commonaccent_xlsr-es-spanish (MIT), from the CommonAccent paper
  (Zuluaga-Gomez et al., 2023, arXiv:2305.18283).
- Base model of both: facebook/wav2vec2-large-xlsr-53 (Apache-2.0).
- 10-zone model: trained on Mozilla Common Voice 27.0 Spanish (CC0), downloaded from Mozilla
  Data Collective under its terms. Room echo and noise from the
  [Room Impulse Response and Noise Database](https://www.openslr.org/28/) (OpenSLR 28, Apache 2.0;
  real room recordings from RWCP, REVERB 2014 and AIR; noise sounds from the public-domain
  Freesound part of MUSAN).
- Map data: made with [Natural Earth](https://www.naturalearthdata.com/) (public domain).
- The project's own code has no license file yet.
