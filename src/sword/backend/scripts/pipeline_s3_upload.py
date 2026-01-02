from __future__ import annotations

"""
Utilities to upload locally produced pipeline artifacts to S3.

Environment variables (typically provided via Streamlit Secrets):
- S3_BUCKET: target bucket, e.g. "cn-dota-sword-s3"
- S3_PREFIX: key prefix, e.g. "data/" (optional; normalized to have trailing '/')

Typical S3 key layout produced by this module (customize infer_kind_and_key if needed):
- {prefix}matches/{filename}.json
- {prefix}obs_logs/teams/{team_safe}/{filename}.csv
- {prefix}replays/{filename}
- {prefix}picks/{filename}
- {prefix}artifacts/{filename}   (fallback)

Public API:
- upload_paths(paths: Iterable[Path], team_name: str, bucket: str | None = None, prefix: str | None = None,
               make_presigned: bool = False, expires: int = 3600) -> list[dict]
- infer_kind_and_key(local: Path, team_name: str, prefix: str) -> tuple[str, str]
- safe_name(s: str) -> str
"""

import os
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

# Relative import within the same package (requires __init__.py in this folder)
try:
    from .storage_s3 import upload, presigned_url  # type: ignore
except Exception:
    from .storage_s3 import upload  # type: ignore
    presigned_url = None  # type: ignore


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return str(v) if v is not None else default


def _normalize_prefix(p: Optional[str]) -> str:
    if not p:
        return ""
    p = p.strip()
    if not p:
        return ""
    while p.startswith("/"):
        p = p[1:]
    if not p.endswith("/"):
        p = p + "/"
    return p


S3_BUCKET = _env("S3_BUCKET")
S3_PREFIX = _normalize_prefix(_env("S3_PREFIX"))


def safe_name(s: str) -> str:
    """
    Keep alnum, dash, underscore, dot and space; replace others with '_'
    """
    return "".join(c if c.isalnum() or c in "-_ ." else "_" for c in (s or "").strip())


def infer_kind_and_key(local: Path, team_name: str, prefix: str) -> Tuple[str, str]:
    """
    Infer artifact kind and S3 key for a given local file path.

    Rules (case-insensitive path checks):
      - .../matches/**/<file>.json      -> matches/<file>.json
      - .../obs_logs/teams/<team>/**/<file>.csv -> obs_logs/teams/<team_safe>/<file>.csv
      - .../replays/**/<file>           -> replays/<file>
      - .../picks/**/<file>             -> picks/<file>
      - otherwise                       -> artifacts/<file>
    """
    p = local.as_posix()
    pl = p.lower()
    team_safe = safe_name(team_name)
    filename = local.name

    if "/matches/" in pl and filename.lower().endswith(".json"):
        kind = "matches"
        key = f"{prefix}matches/{filename}"
    elif "/obs_logs/" in pl and "/teams/" in pl and filename.lower().endswith(".csv"):
        kind = "obs_logs"
        key = f"{prefix}obs_logs/teams/{team_safe}/{filename}"
    elif "/replays/" in pl:
        kind = "replays"
        key = f"{prefix}replays/{filename}"
    elif "/picks/" in pl:
        kind = "picks"
        key = f"{prefix}picks/{filename}"
    else:
        kind = "artifacts"
        key = f"{prefix}artifacts/{filename}"
    return kind, key


def upload_one(
    local: Path,
    team_name: str,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    make_presigned: bool = False,
    expires: int = 3600,
) -> Dict:
    """
    Upload a single local file to S3. Returns a result dict:
      {
        "local": "/path/to/file",
        "kind": "obs_logs",
        "key": "data/obs_logs/teams/Team_Falcons/match_123_wards.csv",
        "s3_url": "s3://bucket/key",
        "presigned_url": "https://..." | "",
        "ok": True/False,
        "msg": "uploaded" | "error: ...",
      }
    """
    result: Dict = {"local": str(local)}
    try:
        if not local.exists() or not local.is_file():
            result.update({"ok": False, "msg": "not found"})
            return result

        bkt = (bucket or S3_BUCKET or "").strip()
        if not bkt:
            result.update({"ok": False, "msg": "missing S3 bucket (env S3_BUCKET not set)"})
            return result

        pfx = _normalize_prefix(prefix or S3_PREFIX)

        kind, key = infer_kind_and_key(local, team_name, pfx)
        s3_url = upload(local, bkt, key)
        ps_url = ""
        if make_presigned and callable(presigned_url):
            try:
                ps_url = presigned_url(bkt, key, expires)  # type: ignore
            except Exception as e:
                ps_url = ""
                result.setdefault("warnings", []).append(f"presign failed: {e}")

        result.update(
            {
                "kind": kind,
                "key": key,
                "s3_url": s3_url,
                "presigned_url": ps_url,
                "ok": True,
                "msg": "uploaded",
            }
        )
        return result
    except Exception as e:
        result.update({"ok": False, "msg": f"error: {e}"})
        return result


def upload_paths(
    paths: Iterable[Path],
    team_name: str,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    make_presigned: bool = False,
    expires: int = 3600,
) -> List[Dict]:
    """
    Batch upload a list of local files to S3.
    See upload_one for each result dict schema.
    """
    results: List[Dict] = []
    for p in paths:
        res = upload_one(
            local=Path(p),
            team_name=team_name,
            bucket=bucket,
            prefix=prefix,
            make_presigned=make_presigned,
            expires=expires,
        )
        results.append(res)
    return results


# Optional: light CLI for ad-hoc testing
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print(
            "Usage: python -m src.sword.backend.scripts.pipeline_s3_upload <team_name> <file1> [file2 ...] [--bucket BUCKET] [--prefix PREFIX] [--presign]"
        )
        sys.exit(1)

    args = sys.argv[1:]
    team = args.pop(0)

    bkt = None
    pfx = None
    presign = False
    files: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--bucket" and i + 1 < len(args):
            bkt = args[i + 1]
            i += 2
        elif a == "--prefix" and i + 1 < len(args):
            pfx = args[i + 1]
            i += 2
        elif a == "--presign":
            presign = True
            i += 1
        else:
            files.append(a)
            i += 1

    paths = [Path(f) for f in files]
    res = upload_paths(paths, team_name=team, bucket=bkt, prefix=pfx, make_presigned=presign)
    print(json.dumps(res, indent=2, ensure_ascii=False))