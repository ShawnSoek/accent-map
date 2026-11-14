"""Download the files a model needs, at the revisions pinned in its config.yaml.

Usage: python scripts/fetch_model.py models/commonaccent_es6

Files go to data/hf_cache/ (override with ACCENT_MAP_HF_CACHE). Nothing is downloaded
when the server starts; it only reads this cache.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from server.config import load_config  # noqa: E402
from server.model_api import backend_class  # noqa: E402
from server.paths import hf_cache_dir  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    config = load_config(sys.argv[1])
    cache = hf_cache_dir()
    print(f"Fetching files for {config.name} into {cache} ...")
    backend_class(config.backend).fetch(config, cache)
    print("Done.")


if __name__ == "__main__":
    main()
