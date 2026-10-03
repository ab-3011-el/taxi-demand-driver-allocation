"""Shared constants and pure-Python helpers (no pyspark import, so the dashboard can use it)."""
import math
import os

# Storage root. Default is HDFS; override for local tests: RIDES_BASE=file:///tmp/rides
BASE = os.environ.get("RIDES_BASE", "hdfs:///rides").rstrip("/")
RAW = f"{BASE}/raw/train.csv"
CLEAN = f"{BASE}/clean"
FEATURES = f"{BASE}/features"
MODEL = f"{BASE}/models/gbt"
RESULTS = f"{BASE}/results"

# NYC bounding box and ~1 km grid
LON_MIN, LON_MAX = -74.05, -73.70
LAT_MIN, LAT_MAX = 40.55, 40.95
CELL = 0.01
TOP_ZONES = 200

# Time-based split: train on date < TRAIN_END, test on the rest (June 2016)
TRAIN_END = "2016-06-01"
FEATURE_COLS = ["hour", "dow", "is_weekend", "zone_x", "zone_y", "hist_mean", "lag_168"]

# Allocation defaults (both are adjustable in the dashboard)
DEFAULT_FLEET = 3000
DEFAULT_TRIPS_PER_DRIVER = 2.0


def zone_id(lon, lat):
    """Same formula as the Spark jobs: floor((coord - min) / CELL), id = x*100 + y."""
    x = math.floor((lon - LON_MIN) / CELL)
    y = math.floor((lat - LAT_MIN) / CELL)
    return x * 100 + y
