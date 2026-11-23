"""Start the Accent Map server.

Usage:
  python scripts/serve.py [--model models/cv27_es10] [--host 127.0.0.1] [--port 8000]
                          [--update-seconds S]

Loads the model and times one inference (about 10 s), then picks how often to update
(--update-seconds skips the timing and forces a value). Open http://localhost:8000.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402

from server.app import DEFAULT_MODEL_DIR, create_app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="model directory with config.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--update-seconds", type=float, default=None, help="skip timing, update this often")
    args = ap.parse_args()
    app = create_app(model_dir=args.model, update_seconds=args.update_seconds)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
