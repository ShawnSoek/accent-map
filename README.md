# Accent Map

Guesses the regional accent of Spanish speech, from your microphone or an uploaded file,
and shows the probabilities on a map of dialect zones that updates while you listen.

> **Status:** Stage 1 is in progress. Setup instructions will be added when the demo runs end to end.

## What the stage-1 model can and can't do

The model is [`Jzuluaga/accent-id-commonaccent_xlsr-es-spanish`](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-es-spanish)
(MIT license). It has 6 classes, which are dialect zones, not countries:

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
- A speaker from outside these zones, for example Madrid or Central America, will still be
  assigned to one of the six zones.

## Layout

```
models/<name>/   one folder per model: config.yaml (+ map zones)
src/server/      Python server: audio windowing, smoothing, model adapters
web/             the browser page
tests/           pytest
data/            downloads and your own recordings (not committed)
```
