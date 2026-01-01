import os
from pathlib import Path
from typing import Optional

"""
Unified path configuration module.
- Default data directory: <repo>/src/sword/data
- You can override via environment variables:
  PROJECT_ROOT, DATA_DIR, MATCHES_DIR, WARDS_DIR, REPLAYS_DIR, PICKS_DIR, SCRIPTS_DIR
"""

# Module root (src/sword)
_MODULE_ROOT = Path(__file__).resolve().parents[1]

# Project root defaults to src/sword
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(_MODULE_ROOT))).resolve()

# Data directory unified to src/sword/data
DATA_DIR = Path(os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()

MATCHES_DIR = Path(os.environ.get("MATCHES_DIR", str(DATA_DIR / "matches"))).resolve()
WARDS_DIR   = Path(os.environ.get("WARDS_DIR",   str(DATA_DIR / "obs_logs"))).resolve()
REPLAYS_DIR = Path(os.environ.get("REPLAYS_DIR", str(DATA_DIR / "replays"))).resolve()
PICKS_DIR   = Path(os.environ.get("PICKS_DIR",   str(DATA_DIR / "picks"))).resolve()

# Scripts directory (default keeps backend/scripts; change if desired)
SCRIPTS_DIR = Path(os.environ.get("SCRIPTS_DIR", str(PROJECT_ROOT / "backend" / "scripts"))).resolve()

def ensure_dirs():
    for d in (DATA_DIR, MATCHES_DIR, WARDS_DIR, REPLAYS_DIR, PICKS_DIR):
        d.mkdir(parents=True, exist_ok=True)

def summary() -> dict:
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_DIR": str(DATA_DIR),
        "MATCHES_DIR": str(MATCHES_DIR),
        "WARDS_DIR": str(WARDS_DIR),
        "REPLAYS_DIR": str(REPLAYS_DIR),
        "PICKS_DIR": str(PICKS_DIR),
        "SCRIPTS_DIR": str(SCRIPTS_DIR),
    }