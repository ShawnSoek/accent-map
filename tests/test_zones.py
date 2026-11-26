"""The committed map zones of every model: complete, valid, non-overlapping, and right."""

import json
from itertools import combinations

import pytest
import yaml
from shapely.geometry import Point, shape

from server.config import load_config
from server.paths import MODELS_DIR

MODEL_NAMES = ("commonaccent_es6", "cv27_es10")


def load(model_name):
    config = load_config(MODELS_DIR / model_name)
    path = config.model_dir / config.map.zones_file
    features = [(f["properties"], shape(f["geometry"])) for f in json.loads(path.read_text())["features"]]
    return config, path, {p.id: p for p in config.map.panels}, features


LOADED = {}


@pytest.fixture(params=MODEL_NAMES)
def model(request):
    if request.param not in LOADED:
        LOADED[request.param] = load(request.param)
    return LOADED[request.param]


def name(props):
    if props["kind"] == "zone":
        return props["zone"]
    return f"uncovered:{props['id']}" if props["kind"] == "uncovered" else "land"


def test_every_class_zone_is_drawn_and_nothing_else(model):
    config, _, _, features = model
    assert {p["zone"] for p, _ in features if p["kind"] == "zone"} == {c.zone for c in config.classes}


def test_spec_matches_config(model):
    config, _, panels, _ = model
    spec = yaml.safe_load((config.model_dir / "zones_spec.yaml").read_text())
    assert set(spec["zones"]) == {c.zone for c in config.classes}
    assert set(spec["build"]) == set(panels)


def test_feature_properties(model):
    _, _, panels, features = model
    for props, _ in features:
        assert props["kind"] in {"zone", "uncovered", "land"}
        assert props["panel"] in panels
        if props["kind"] == "uncovered":
            assert "not covered" in props["label"]


def test_geometries_are_valid_and_inside_their_panel(model):
    _, _, panels, features = model
    for props, geom in features:
        assert geom.is_valid and not geom.is_empty and geom.geom_type in {"Polygon", "MultiPolygon"}
        w, s, e, n = panels[props["panel"]].bbox
        gw, gs, ge, gn = geom.bounds
        assert gw >= w - 1e-6 and gs >= s - 1e-6 and ge <= e + 1e-6 and gn <= n + 1e-6


def test_no_overlaps_within_a_panel(model):
    _, _, panels, features = model
    for panel in panels:
        geoms = [(name(p), g) for p, g in features if p["panel"] == panel]
        for (a, ga), (b, gb) in combinations(geoms, 2):
            overlap = ga.intersection(gb).area
            assert overlap <= 1e-6 * min(ga.area, gb.area), f"{panel}: {a} overlaps {b} by {overlap}"


def test_file_is_small(model):
    assert model[1].stat().st_size < 300 * 1024


CITIES = [
    ("Seville", -5.99, 37.39, "es_south"), ("Málaga", -4.42, 36.72, "es_south"),
    ("Badajoz", -6.97, 38.88, "es_south"), ("Murcia", -1.13, 37.99, "es_south"),
    ("Barcelona", 2.17, 41.39, "uncovered:spain_other"),
    ("Lisbon", -9.14, 38.72, "land"),
    ("Mexico City", -99.13, 19.43, "mexico"), ("Monterrey", -100.32, 25.69, "mexico"),
    ("Guadalajara", -103.35, 20.66, "mexico"),
    ("Xalapa", -96.92, 19.54, "caribbean"), ("Villahermosa", -92.93, 17.99, "caribbean"),
    ("Mérida (MX)", -89.62, 20.97, "mexico"), ("Campeche", -90.53, 19.85, "mexico"),
    ("Ciudad Victoria", -99.14, 23.74, "mexico"), ("Cancún", -86.85, 21.16, "caribbean"),
    ("Havana", -82.37, 23.11, "caribbean"), ("Santo Domingo", -69.93, 18.49, "caribbean"),
    ("Port-au-Prince", -72.34, 18.54, "land"), ("San Juan (PR)", -66.11, 18.40, "caribbean"),
    ("Panama City", -79.52, 8.98, "caribbean"), ("Caracas", -66.90, 10.48, "caribbean"),
    ("Barranquilla", -74.80, 10.96, "caribbean"), ("Cartagena", -75.51, 10.39, "caribbean"),
    ("Mérida (VE)", -71.14, 8.59, "andean_pacific"), ("San Cristóbal (VE)", -72.22, 7.77, "andean_pacific"),
    ("Bogotá", -74.07, 4.71, "andean_pacific"), ("Medellín", -75.57, 6.25, "andean_pacific"),
    ("Quito", -78.47, -0.18, "andean_pacific"), ("Lima", -77.04, -12.05, "andean_pacific"),
    ("La Paz", -68.15, -16.50, "andean_pacific"), ("Cochabamba", -66.16, -17.39, "andean_pacific"),
    ("Santa Cruz (BO)", -63.18, -17.78, "rioplatense"), ("Tarija", -64.73, -21.53, "rioplatense"),
    ("Asunción", -57.58, -25.26, "rioplatense"), ("Buenos Aires", -58.38, -34.60, "rioplatense"),
    ("Córdoba (AR)", -64.18, -31.42, "rioplatense"), ("Montevideo", -56.16, -34.90, "rioplatense"),
    ("Mendoza", -68.83, -32.89, "chilean"), ("San Juan (AR)", -68.54, -31.54, "chilean"),
    ("Santiago", -70.65, -33.45, "chilean"), ("Punta Arenas", -70.91, -53.16, "chilean"),
    ("São Paulo", -46.63, -23.55, "land"), ("Houston", -95.37, 29.76, "land"),
]

# Where the two models differ (stage 1 had no class for these areas).
ONLY = {
    "commonaccent_es6": [
        ("Madrid", -3.70, 40.42, "uncovered:spain_other"), ("Bilbao", -2.93, 43.26, "uncovered:spain_other"),
        ("Guatemala City", -90.51, 14.63, "uncovered:central_america"),
        ("San José (CR)", -84.09, 9.93, "uncovered:central_america"),
    ],
    "cv27_es10": [
        ("Madrid", -3.70, 40.42, "es_centre_south"), ("Toledo", -4.02, 39.86, "es_centre_south"),
        ("Albacete", -1.86, 38.99, "es_centre_south"),
        ("Guadalajara (ES)", -3.17, 40.63, "es_north"), ("Cuenca", -2.13, 40.07, "es_north"),
        ("Bilbao", -2.93, 43.26, "es_north"), ("Valladolid", -4.72, 41.65, "es_north"),
        ("Zaragoza", -0.88, 41.65, "es_north"), ("Oviedo", -5.85, 43.36, "es_north"),
        ("Santiago de Compostela", -8.54, 42.88, "uncovered:spain_other"),
        ("Valencia", -0.38, 39.47, "uncovered:spain_other"), ("Palma", 2.65, 39.57, "uncovered:spain_other"),
        ("Las Palmas", -15.43, 28.12, "es_canary"), ("Santa Cruz de Tenerife", -16.25, 28.47, "es_canary"),
        ("Guatemala City", -90.51, 14.63, "central_america"), ("San José (CR)", -84.09, 9.93, "central_america"),
        ("Managua", -86.25, 12.11, "central_america"), ("Tegucigalpa", -87.21, 14.07, "central_america"),
        ("San Salvador", -89.19, 13.69, "central_america"),
    ],
}
CASES = [(m, *c) for m in MODEL_NAMES for c in CITIES + ONLY[m]]


@pytest.mark.parametrize("model_name, city, lon, lat, expected", CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_city_is_in_the_expected_zone(model_name, city, lon, lat, expected):
    if model_name not in LOADED:
        LOADED[model_name] = load(model_name)
    features = LOADED[model_name][3]
    point = Point(lon, lat)
    distance, got = min((g.distance(point), name(p)) for p, g in features)
    assert distance < 0.1, f"{city} is {distance:.2f} degrees from any land (simplified away?)"
    assert got == expected
