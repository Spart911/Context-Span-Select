from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MANIFESTS = DATA / "manifests"
CACHE = DATA / "cache"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
EXPERIMENTS = ROOT / "experiments"
LOGS = ROOT / "logs"
ARTIFACTS = ROOT / "artifacts"
