"""Play an audio file into a running server's live WebSocket, in real time, like a microphone.

Usage: python scripts/stream_wav.py FILE [--url ws://127.0.0.1:8000/ws/live] [--chunk-ms 100]

The file is sent at its own sample rate (so the server's resampler is used), one chunk
every chunk-ms, each chunk only after it would have been "recorded". For every event it
prints the end-to-end delay: from the moment the window's last audio was sent until the
event arrived. That is the delay a speaker would see, minus the browser's own capture delay.
"""

import argparse
import asyncio
import json
import math
import statistics
import time

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect


async def run(path: str, url: str, chunk_ms: int) -> None:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1).astype("<f4")
    chunk = int(rate * chunk_ms / 1000)
    chunk_s = chunk / rate
    async with connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "sample_rate": rate}))
        ready = json.loads(await ws.recv())
        print(f"{path}: {len(audio) / rate:.2f} s at {rate} Hz | server: {ready}")
        start = time.perf_counter()

        async def send():
            for k, i in enumerate(range(0, len(audio), chunk)):
                delay = start + (k + 1) * chunk_s - time.perf_counter()   # chunk k is "recorded" by then
                if delay > 0:
                    await asyncio.sleep(delay)
                await ws.send(audio[i:i + chunk].tobytes())
            await ws.send(json.dumps({"type": "stop"}))

        sender = asyncio.create_task(send())
        delays = []
        while True:
            msg = json.loads(await ws.recv())
            now = time.perf_counter()
            if msg["type"] == "event":
                sent_at = start + min(math.ceil(round(msg["t"] / chunk_s, 6)) * chunk_s, len(audio) / rate)
                delays.append((now - sent_at) * 1000)
                top = ", ".join(f"{d['id']} {d['p']:.2f}" for d in msg["top"])
                infer = f"{msg['infer_ms']:.0f}" if msg["infer_ms"] else "-"
                print(f"  t={msg['t']:5.2f} {msg['status']:10s} infer={infer:>4s}ms "
                      f"queue={msg['queue_ms']:4.0f}ms end-to-end={delays[-1]:5.0f}ms "
                      f"dropped={msg['dropped']} | {top}")
            elif msg["type"] == "done":
                print(f"  done: {msg['events']} events, {msg['dropped']} windows skipped | end-to-end delay "
                      f"median {statistics.median(delays):.0f} ms, max {max(delays):.0f} ms")
                break
            else:
                print("  ", msg)
                break
        await sender


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/live")
    ap.add_argument("--chunk-ms", type=int, default=100)
    args = ap.parse_args()
    for f in args.files:
        asyncio.run(run(f, args.url, args.chunk_ms))


if __name__ == "__main__":
    main()
