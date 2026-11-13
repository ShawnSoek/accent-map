"""Build a model's map zones from Natural Earth admin-1 regions.

Usage:
  python scripts/build_zones.py models/commonaccent_es6 [--natural-earth PATH]

Reads the model's config.yaml (map panels) and zones_spec.yaml (which regions form which
zone) and writes the GeoJSON named by config.yaml map.zones_file. Features:
  {"kind": "zone", "zone": <zone id>, "panel": <panel id>}
  {"kind": "uncovered", "id": ..., "label": ..., "panel": ...}   Spanish-speaking, no class
  {"kind": "land", "panel": ...}                                 everything else, no borders
Neighbouring groups are simplified together (shapely.coverage_simplify), so shared borders
stay identical: no gaps, no overlaps.

Input (not committed, public domain), get it once with:
  mkdir -p data/naturalearth && curl -L -o data/naturalearth/ne_10m_admin_1_states_provinces.geojson \\
    https://raw.githubusercontent.com/nvkelso/natural-earth-vector/f1890d9f152c896d250a77557a5751a93d494776/geojson/ne_10m_admin_1_states_provinces.geojson
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import shapely
import yaml
from shapely.geometry import box, mapping, shape

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from server.config import load_config  # noqa: E402

NE_DEFAULT = REPO / "data" / "naturalearth" / "ne_10m_admin_1_states_provinces.geojson"
NE_SHA256 = "22d0e3ad85eb3e27f17cabf8ba2d50e554fbc27a87796ff891d958185da62fb5"  # v5.1.2 @ f1890d9f
LAND = ("land", None)


def load_regions(path: Path) -> list[dict]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != NE_SHA256:
        sys.exit(f"{path}: unexpected SHA-256 {digest}; expected the pinned Natural Earth v5.1.2 file")
    regions = []
    for f in json.loads(path.read_text(encoding="utf-8"))["features"]:
        p = f["properties"]
        regions.append({
            "code": p["iso_3166_2"], "country": p["adm0_a3"], "community": p["region"],
            "name": p["name"] or p["name_en"], "geom": shapely.make_valid(shape(f["geometry"])),
        })
    return regions


def build_rules(spec: dict) -> dict[str, dict]:
    """{"codes"|"communities"|"countries": {value: group}}, rejecting any value used twice."""
    rules = {"codes": {}, "communities": {}, "countries": {}}
    groups = [(("zone", z), r) for z, r in spec["zones"].items()]
    groups += [(("uncovered", u), r) for u, r in spec.get("uncovered", {}).items()]
    for group, rule in groups:
        for kind in rules:
            for value in rule.get(kind, []):
                if value in rules[kind]:
                    sys.exit(f"{kind} {value!r} is assigned twice: {rules[kind][value]} and {group}")
                rules[kind][value] = group
    return rules


def assign(regions: list[dict], rules: dict) -> None:
    used = {kind: set() for kind in rules}
    for r in regions:
        for kind, key in (("codes", "code"), ("communities", "community"), ("countries", "country")):
            if r[key] in rules[kind]:
                r["group"] = rules[kind][r[key]]
                used[kind].add(r[key])
                break
        else:
            r["group"] = LAND
    for kind in rules:
        unused = sorted(set(rules[kind]) - used[kind])
        if unused:
            sys.exit(f"these {kind} in zones_spec.yaml match no Natural Earth region: {unused}")


def drop_small_parts(geom, min_area: float):
    parts = [p for p in getattr(geom, "geoms", [geom]) if p.geom_type == "Polygon" and p.area >= min_area]
    return shapely.MultiPolygon(parts) if parts else None


def rounded(obj, ndigits: int):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, (list, tuple)):
        return [rounded(x, ndigits) for x in obj]
    if isinstance(obj, dict):
        return {k: rounded(v, ndigits) for k, v in obj.items()}
    return obj


def build_panel(regions, panel, settings: dict, uncovered_labels: dict) -> list[dict]:
    frame = box(*panel.bbox)
    parts = defaultdict(list)
    for r in regions:
        if r["geom"].intersects(frame):
            parts[r["group"]].append(r["geom"].intersection(frame))
    # Zones first, then uncovered, then land: any overlap in the source data goes to the earlier one.
    order = sorted(parts, key=lambda g: (["zone", "uncovered", "land"].index(g[0]), g[1] or ""))
    keys, geoms, taken = [], [], None
    for key in order:
        geom = shapely.union_all(parts[key])
        if taken is not None:
            geom = geom.difference(taken)
        geom = drop_small_parts(geom, settings["min_area"])
        if geom is None:
            continue
        taken = geom if taken is None else shapely.union(taken, geom)
        keys.append(key)
        geoms.append(geom)
    simplified = shapely.coverage_simplify(geoms, settings["tolerance"])
    grid = settings["tolerance"] / 8
    simplified = shapely.set_precision(simplified, grid)
    features = []
    for (kind, ident), geom in zip(keys, simplified):
        if geom.is_empty:
            continue
        props = {"kind": kind, "panel": panel.id}
        if kind == "zone":
            props["zone"] = ident
        elif kind == "uncovered":
            props.update(id=ident, label=uncovered_labels[ident])
        features.append({"type": "Feature", "properties": props,
                         "geometry": rounded(mapping(geom), 4)})
    return features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--natural-earth", default=str(NE_DEFAULT))
    args = ap.parse_args()

    config = load_config(args.model_dir)
    spec = yaml.safe_load((config.model_dir / "zones_spec.yaml").read_text(encoding="utf-8"))
    class_zones = {c.zone for c in config.classes}
    if set(spec["zones"]) != class_zones:
        sys.exit(f"zones_spec.yaml zones {sorted(spec['zones'])} != config class zones {sorted(class_zones)}")
    if set(spec["build"]) != {p.id for p in config.map.panels}:
        sys.exit("zones_spec.yaml build settings must match config.yaml map panels")

    regions = load_regions(Path(args.natural_earth))
    assign(regions, build_rules(spec))
    labels = {u: r["label"] for u, r in spec.get("uncovered", {}).items()}
    features = []
    for panel in config.map.panels:
        features += build_panel(regions, panel, spec["build"][panel.id], labels)

    out = config.model_dir / config.map.zones_file
    out.write_text(json.dumps({
        "type": "FeatureCollection",
        "attribution": "Made with Natural Earth (v5.1.2). Zones are approximate dialect areas, not borders.",
        "features": features,
    }, separators=(",", ":")), encoding="utf-8")
    counts = defaultdict(int)
    for f in features:
        counts[(f["properties"]["panel"], f["properties"]["kind"])] += 1
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB): " + ", ".join(f"{p}/{k}: {n}" for (p, k), n in sorted(counts.items())))


if __name__ == "__main__":
    main()
