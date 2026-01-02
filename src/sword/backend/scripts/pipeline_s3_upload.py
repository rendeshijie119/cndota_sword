from __future__ import annotations

# Bootstrap sys.path so absolute imports work even on Streamlit Cloud
import sys, pathlib, os
THIS = pathlib.Path(__file__).resolve()
SRC_DIR = THIS.parents[3]          # .../src
REPO_ROOT = THIS.parents[4]        # repo root
for p in (str(SRC_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Prefer absolute import; fall back to local if needed
try:
    from src.sword.backend.scripts.storage_s3 import upload  # returns s3://bucket/key
except Exception:
    # Fallback: import from sibling file directly
    from storage_s3 import upload

from pathlib import Path
from typing import Iterable, List, Dict, Tuple

def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return str(v) if v is not None else default

S3_BUCKET = _env("S3_BUCKET")
S3_PREFIX = _env("S3_PREFIX", "").rstrip("/") + "/" if _env("S3_PREFIX") else ""

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ ." else "_" for c in (s or "").strip())

def infer_kind_and_key(local: Path, team_name: str) -> Tuple[str, str]:
    p = local.as_posix().lower()
    team_safe = safe_name(team_name)
    filename = local.name

    if "/matches/" in p and filename.endswith(".json"):
        kind = "matches"
        key = f"{S3_PREFIX}matches/{filename}"
    elif "/obs_logs/" in p and "/teams/" in p and filename.endswith(".csv"):
        kind = "obs_logs"
        key = f"{S3_PREFIX}obs_logs/teams/{team_safe}/{filename}"
    elif "/replays/" in p:
        kind = "replays"
        key = f"{S3_PREFIX}replays/{filename}"
    elif "/picks/" in p:
        kind = "picks"
        key = f"{S3_PREFIX}picks/{filename}"
    else:
        kind = "artifacts"
        key = f"{S3_PREFIX}artifacts/{filename}"
    return kind, key

def upload_paths(paths: Iterable[Path], team_name: str) -> List[Dict]:
    results: List[Dict] = []
    for p in paths:
        try:
            if not p.exists() or not p.is_file():
                results.append({"local": str(p), "ok": False, "msg": "not found"})
                continue
            kind, key = infer_kind_and_key(p, team_name)
            s3_url = upload(p, S3_BUCKET, key)
            results.append({"local": str(p), "kind": kind, "key": key, "s3_url": s3_url, "ok": True, "msg": "uploaded"})
        except Exception as e:
            results.append({"local": str(p), "ok": False, "msg": f"error: {e}"})
    return results