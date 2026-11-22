# Accent Map

Goal: web app that guesses the regional accent of Spanish speech (file upload
or live microphone) and shows probabilities on a map that updates in real time.
Stage 1 uses the pretrained model Jzuluaga/accent-id-commonaccent_xlsr-es-spanish
(6 classes = dialect zones, not countries). Later stages add own models with
more classes, so the app must never hardcode the class list.

## Model plug-in design (do not change without asking)
- Each model lives in models/<name>/ with a config.yaml: display name, source,
  pinned revision, class list (id, label, short description, map zone).
- The app reads only config.yaml and calls: audio (16 kHz mono) -> {class: probability},
  one result every ~0.5 s over the last ~10 s of audio.
- Read the real class order from the model files; do not assume it.

## Known facts about the stage-1 model
- Reported 68.5% accuracy on Common Voice test (read speech). Not verified on
  my own recordings; treat as an upper bound.
- "Spain" class = south only (Andalusia, Extremadura, Murcia); Madrid/north are
  not covered. Caribbean and Chilean zones include neighboring regions.
- Map shows zones with a legend, not national borders.

## Rules
- Work in small steps; after each step run tests and tell me what you verified.
- Never put credentials in code or commit them. Ask me to log in or accept licenses.
- Never train or tune on data/own_tests/. Split any future data by speaker AND source.
- Pin versions in requirements.txt (SpeechBrain is version-sensitive) and pin the model revision.
- Keep DECISIONS.md with short notes on each choice and why.
- README must let a stranger run the demo.

## Layout
models/ | src/server | web/ | tests/ | data/ (not committed) | data/own_tests/