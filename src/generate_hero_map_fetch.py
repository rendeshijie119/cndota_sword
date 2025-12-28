#!/usr/bin/env python3
"""
generate_hero_map_fetch.py

Fetch hero list and write CSV/JSON with columns:
  id, localized_name, hero_name

Behavior:
- Try to fetch Chinese localized names from Steam Web API (GetHeroes with language=zh_cn)
  if a Steam API key is provided (env STEAM_API_KEY or --steam-key).
- Otherwise fetch from OpenDota (localized_name in English) and set hero_name = localized_name
  unless overridden by --heroes-file (local mapping with Chinese names).
- Local --heroes-file (JSON) can contain either:
    { "1": "敌法师", "2": "斧王", ... }
  or a list of objects [ {"id":1,"name":"敌法师"}, ... ]
  Local entries override fetched entries.
- Outputs:
    {out_dir}/hero_map.csv    (columns: id,localized_name,hero_name)
    {out_dir}/hero_map.json   (id -> {"localized_name":..., "hero_name":...})

Usage:
  python generate_hero_map_fetch.py --out-dir data --verbose
  # with Steam API key via env:
  STEAM_API_KEY=XXXXX python generate_hero_map_fetch.py --out-dir data --verbose
  # with local chinese file to override
  python generate_hero_map_fetch.py --out-dir data --heroes-file my_cn.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger("generate_hero_map_fetch")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def fetch_opendota_heroes(timeout: int = 20) -> Dict[str, Dict[str, Any]]:
    """
    Fetch from OpenDota: returns mapping id(str) -> {"localized_name": <str>}
    """
    try:
        import requests
    except Exception:
        logger.error("requests not installed. Install with: pip install requests")
        return {}
    url = "https://api.opendota.com/api/heroes"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        mapping: Dict[str, Dict[str, Any]] = {}
        for item in data:
            hid = item.get("id")
            loc = item.get("localized_name") or item.get("name")
            if hid is not None and loc:
                mapping[str(hid)] = {"localized_name": str(loc)}
        return mapping
    except Exception as e:
        logger.error("Failed to fetch OpenDota heroes: %s", e)
        return {}


def fetch_steam_heroes(steam_key: str, timeout: int = 20) -> Dict[str, Dict[str, Any]]:
    """
    Use Steam Web API GetHeroes to request language=zh_cn (if available).
    Requires steam_key (may work without but key is recommended).
    Returns mapping id(str) -> {"localized_name": <str>, "hero_name": <chinese>}
    """
    try:
        import requests
    except Exception:
        logger.error("requests not installed. Install with: pip install requests")
        return {}

    # IEconDOTA2_570/GetHeroes/v1/?key=...&language=zh_cn
    url = "https://api.steampowered.com/IEconDOTA2_570/GetHeroes/v1/"
    params = {"key": steam_key, "language": "zh_cn"}
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        j = resp.json()
        arr = j.get("result", {}).get("heroes", [])
        mapping: Dict[str, Dict[str, Any]] = {}
        for item in arr:
            hid = item.get("id")
            # Steam returns 'name' as internal name (e.g. npc_dota_hero_antimage), 'localized_name' exists when language set
            loc = item.get("localized_name") or item.get("name")
            if hid is not None and loc:
                mapping[str(hid)] = {"localized_name": str(loc), "hero_name": str(loc)}
        return mapping
    except Exception as e:
        logger.warning("Failed to fetch Steam heroes (zh_cn): %s", e)
        return {}


def load_local_hero_file(path: Path) -> Dict[str, str]:
    """
    Load optional local hero chinese mapping.
    Accepts dict (id->name) or list [{"id":1,"name":".."}]
    Returns mapping id(str) -> chinese name
    """
    if not path or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        logger.error("Failed to read local heroes file %s: %s", path, e)
        return {}

    mapping: Dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            mapping[str(k)] = str(v)
    elif isinstance(data, list):
        for it in data:
            if isinstance(it, dict):
                hid = it.get("id") or it.get("hero_id")
                name = it.get("name") or it.get("localized_name") or it.get("hero_name")
                if hid is not None and name:
                    mapping[str(hid)] = str(name)
    return mapping


def merge_mappings(base: Dict[str, Dict[str, Any]], overrides_cn: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    base: id -> {"localized_name":...} or {"localized_name":..., "hero_name":...}
    overrides_cn: id -> chinese name
    Returns id -> {"localized_name":..., "hero_name":...}
    """
    out: Dict[str, Dict[str, Any]] = {}
    for hid, info in base.items():
        loc = info.get("localized_name") or ""
        cn = info.get("hero_name") or ""
        # if override exists, use it
        if hid in overrides_cn and overrides_cn[hid]:
            cn = overrides_cn[hid]
        # if cn empty, default to localized_name (English) to avoid blanks
        if not cn:
            cn = loc
        out[hid] = {"localized_name": loc, "hero_name": cn}
    # also add any overrides not present in base
    for hid, cn in overrides_cn.items():
        if hid not in out:
            out[hid] = {"localized_name": "", "hero_name": cn}
    return out


def write_outputs(out_map: Dict[str, Dict[str, Any]], out_dir: Path, csv_name: str = "hero_map.csv", json_name: str = "hero_map.json") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / csv_name
    json_path = out_dir / json_name

    # write JSON with structure id -> {localized_name, hero_name}
    try:
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(out_map, fh, indent=2, ensure_ascii=False)
        logger.info("Wrote JSON: %s (%d entries)", json_path, len(out_map))
    except Exception as e:
        logger.error("Failed to write JSON %s: %s", json_path, e)

    # write CSV sorted by numeric id where possible
    try:
        rows = sorted(out_map.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "localized_name", "hero_name"])
            for hid, info in rows:
                writer.writerow([hid, info.get("localized_name", ""), info.get("hero_name", "")])
        logger.info("Wrote CSV: %s (%d rows)", csv_path, len(rows))
    except Exception as e:
        logger.error("Failed to write CSV %s: %s", csv_path, e)


def parse_args():
    p = argparse.ArgumentParser(description="Fetch hero id->localized_name and hero_name (Chinese) and write CSV/JSON.")
    p.add_argument("--heroes-file", type=Path, help="Optional local JSON heroes file to merge (id->cn).")
    p.add_argument("--out-dir", type=Path, default=Path("data"), help="Output directory (default: data/).")
    p.add_argument("--steam-key", type=str, help="Optional Steam Web API key to fetch zh_cn localized names (can also set STEAM_API_KEY env).")
    p.add_argument("--no-fetch", action="store_true", help="Do not fetch from remote (use only local file if provided).")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    out_dir: Path = args.out_dir
    local_cn = load_local_hero_file(args.heroes_file) if args.heroes_file else {}

    base_map: Dict[str, Dict[str, Any]] = {}

    steam_key = args.steam_key or os.getenv("STEAM_API_KEY")
    # Try Steam zh_cn if key provided
    if not args.no_fetch and steam_key:
        logger.info("Attempting to fetch Steam heroes with language=zh_cn (Steam API key provided)...")
        steam_map = fetch_steam_heroes(steam_key)
        if steam_map:
            base_map.update(steam_map)
        else:
            logger.warning("Steam zh_cn fetch failed or empty; falling back to OpenDota (English localized_name).")

    # If base_map empty and fetch allowed, fetch OpenDota (English localized_name)
    if not args.no_fetch and not base_map:
        logger.info("Fetching heroes from OpenDota (localized_name in English)...")
        od = fetch_opendota_heroes()
        if od:
            base_map.update(od)
        else:
            logger.warning("OpenDota fetch failed or empty.")

    # If no remote data but local file exists, we still want produce mapping
    if not base_map and local_cn:
        # create base entries from local_cn using empty localized_name
        for hid, cn in local_cn.items():
            base_map[hid] = {"localized_name": "", "hero_name": cn}

    if not base_map:
        logger.error("No hero data available (no fetch and no local file). Exiting.")
        return 2

    # Merge with local Chinese overrides (local overrides fetched names)
    merged = merge_mappings(base_map, local_cn)

    write_outputs(merged, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())