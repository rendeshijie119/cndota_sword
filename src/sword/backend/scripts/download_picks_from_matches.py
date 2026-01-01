#!/usr/bin/env python3
"""
download_picks_from_matches.py

Extract per-match picks/bans (BP) from local match JSONs (matches-dir) and
write canonical per-match picks JSONs into a dedicated picks directory
(default: data/picks). Also optionally copy per-team picks files into
data/picks/teams/<TeamName>/... to match your obs_logs structure.

Additionally this script can produce an aggregated picks table (parquet or csv)
at data/picks/picks.parquet (or picks.csv) for fast analysis.

Hero name completion strategy (in order):
  1. Use hero_name fields if present in match JSON's picks_bans or players.
  2. Use an optional local hero mapping file (--heroes-file) mapping hero_id -> hero_name.
  3. Optionally fetch hero list from OpenDota (--fetch-heroes). (Only if requested.)

Usage examples:
  python download_picks_from_matches.py --matches-dir ./data/matches --out-dir ./data/picks --group-by-team --export-aggregate --verbose
  python download_picks_from_matches.py --matches-dir ./data/matches --out-dir ./data/picks --heroes-file heroes.json --export-aggregate

Notes:
- Requires requests only if --fetch-heroes is used.
- For parquet output, pandas + pyarrow (or fastparquet) is recommended. Falls back to CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("download_picks_from_matches")
logging_handler = logging.StreamHandler()
logging_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(logging_handler)
logger.setLevel(logging.INFO)

DEFAULT_MATCHES_DIR = Path("data/matches")
DEFAULT_OUT_DIR = Path("data/picks")
DEFAULT_WORKERS = 4

# -----------------------
# Helper: team/ids/names
# -----------------------
def safe_name(s: Optional[str], max_len: int = 120) -> str:
    if not s:
        return "unknown"
    s2 = str(s).strip()
    s2 = "".join(ch if (ch.isalnum() or ch in " ._-") else "_" for ch in s2)
    s2 = s2.strip(" _-.")
    if not s2:
        return "unknown"
    if len(s2) > max_len:
        s2 = s2[: max_len - 3] + "..."
    return s2

def is_radiant(player_slot: int) -> bool:
    try:
        return int(player_slot) < 128
    except Exception:
        return False

def get_team_meta(match_data: dict) -> Dict[str, Dict[str, str]]:
    meta = {"radiant": {"team_id": "", "team_name": ""}, "dire": {"team_id": "", "team_name": ""}}
    meta["radiant"]["team_id"] = str(match_data.get("radiant_team_id") or "")
    meta["dire"]["team_id"] = str(match_data.get("dire_team_id") or "")
    meta["radiant"]["team_name"] = match_data.get("radiant_name") or ""
    meta["dire"]["team_name"] = match_data.get("dire_name") or ""
    for side in ("radiant", "dire"):
        v = match_data.get(f"{side}_team")
        if isinstance(v, dict):
            meta[side]["team_id"] = meta[side]["team_id"] or str(v.get("team_id") or v.get("id") or "")
            meta[side]["team_name"] = meta[side]["team_name"] or (v.get("name") or v.get("team_name") or "")
    # final fallback
    for side in ("radiant", "dire"):
        if not meta[side]["team_name"]:
            tid = meta[side]["team_id"]
            if tid:
                meta[side]["team_name"] = f"team_{tid}"
            else:
                meta[side]["team_name"] = "unknown"
    return meta

# -----------------------
# Picks extraction
# -----------------------
def infer_match_id(match_data: dict, filename: Optional[str] = None) -> Optional[int]:
    mid = match_data.get("match_id") or match_data.get("matchid") or match_data.get("matchId")
    try:
        if mid is not None:
            return int(mid)
    except Exception:
        pass
    if filename:
        import re
        m = re.search(r"(\d{6,})", Path(filename).name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def extract_hero_picks(match_data: dict) -> Dict[str, Any]:
    """
    Extract and normalize picks/bans and per-team picks from match JSON.
    Returns dict with keys: match_id, picks_bans (raw if any), radiant_picks, dire_picks, radiant_bans, dire_bans
    Each pick entry: {hero_id, hero_name (maybe None), player_slot?, account_id?}
    """
    out: Dict[str, Any] = {}
    mid = match_data.get("match_id") or match_data.get("matchid") or match_data.get("matchId")
    try:
        out["match_id"] = int(mid) if mid is not None else None
    except Exception:
        out["match_id"] = None

    # copy raw picks_bans if present
    raw_pb = match_data.get("picks_bans") if isinstance(match_data.get("picks_bans"), list) else []
    out["picks_bans"] = raw_pb

    radiant_picks: List[Dict[str, Any]] = []
    dire_picks: List[Dict[str, Any]] = []
    radiant_bans: List[Dict[str, Any]] = []
    dire_bans: List[Dict[str, Any]] = []

    # If picks_bans has entries, try to parse them
    if raw_pb:
        for pb in raw_pb:
            try:
                is_pick = bool(pb.get("is_pick") or pb.get("pick") or pb.get("isPick") or False)
                team_val = pb.get("team")  # can be 0/1 or 'radiant'/'dire'
                team = None
                if isinstance(team_val, int):
                    team = "radiant" if int(team_val) == 0 else "dire"
                elif isinstance(team_val, str):
                    tv = team_val.lower()
                    if tv in ("radiant", "r", "0"):
                        team = "radiant"
                    elif tv in ("dire", "d", "1"):
                        team = "dire"
                # hero id/name fields vary
                hero_id = pb.get("hero_id") or pb.get("hero") or pb.get("heroId") or None
                hero_name = pb.get("hero_name") or pb.get("hero_string") or pb.get("heroName") or None
                entry = {"hero_id": hero_id, "hero_name": hero_name}
                if pb.get("player_slot") is not None:
                    entry["player_slot"] = pb.get("player_slot")
                if pb.get("account_id") is not None:
                    entry["account_id"] = pb.get("account_id")
                if is_pick:
                    if team == "radiant":
                        radiant_picks.append(entry)
                    elif team == "dire":
                        dire_picks.append(entry)
                else:
                    if team == "radiant":
                        radiant_bans.append(entry)
                    elif team == "dire":
                        dire_bans.append(entry)
            except Exception:
                continue

    # Fallback: if no picks from picks_bans, derive from players[] hero_id
    if not radiant_picks and not dire_picks:
        players = match_data.get("players", []) or []
        for p in players:
            try:
                slot = p.get("player_slot")
                team = "radiant" if is_radiant(slot) else "dire"
                hero_id = p.get("hero_id") or p.get("hero") or p.get("heroId") or None
                hero_name = p.get("hero_name") or p.get("hero_string") or None
                entry = {"hero_id": hero_id, "hero_name": hero_name, "player_slot": slot, "account_id": p.get("account_id")}
                if team == "radiant":
                    radiant_picks.append(entry)
                else:
                    dire_picks.append(entry)
            except Exception:
                continue

    out["radiant_picks"] = radiant_picks
    out["dire_picks"] = dire_picks
    out["radiant_bans"] = radiant_bans
    out["dire_bans"] = dire_bans
    return out

# -----------------------
# Hero name completion helpers
# -----------------------
def build_hero_map_from_matches(matches_dir: Path) -> Dict[str, str]:
    """
    Attempt to build a mapping hero_id->hero_name by scanning match files for hero_name occurrences.
    Returns mapping with str(hero_id) -> hero_name.
    """
    mapping: Dict[str, str] = {}
    for p in matches_dir.rglob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as fh:
                mj = json.load(fh)
        except Exception:
            continue
        # scan picks_bans
        for pb in (mj.get("picks_bans") or []):
            try:
                hid = pb.get("hero_id") or pb.get("hero")
                hname = pb.get("hero_name") or pb.get("hero_string") or pb.get("heroName")
                if hid and hname:
                    mapping[str(hid)] = str(hname)
            except Exception:
                continue
        # scan players hero_name
        for pl in (mj.get("players") or []):
            try:
                hid = pl.get("hero_id") or pl.get("hero")
                hname = pl.get("hero_name") or pl.get("hero_string")
                if hid and hname:
                    mapping[str(hid)] = str(hname)
            except Exception:
                continue
    return mapping

def load_hero_map_from_file(path: Path) -> Dict[str, str]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        # accept either list of {"id":..,"name":..} or dict id->name
        mapping: Dict[str, str] = {}
        if isinstance(data, dict):
            # keys may be ints or strings
            for k, v in data.items():
                mapping[str(k)] = str(v)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and ("id" in item and ("name" in item or "localized_name" in item)):
                    hid = item.get("id")
                    hname = item.get("name") or item.get("localized_name")
                    if hid and hname:
                        mapping[str(hid)] = str(hname)
        return mapping
    except Exception:
        return {}

def fetch_heroes_from_opendota() -> Dict[str, str]:
    try:
        import requests
    except Exception:
        logger.warning("requests not installed; cannot fetch hero list from OpenDota")
        return {}
    url = "https://api.opendota.com/api/heroes"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        mapping: Dict[str, str] = {}
        for item in data:
            hid = item.get("id")
            name = item.get("localized_name") or item.get("name")
            if hid is not None and name:
                mapping[str(hid)] = str(name)
        return mapping
    except Exception as e:
        logger.warning("Failed to fetch heroes from OpenDota: %s", e)
        return {}

def complete_hero_names_for_picks(picks_obj: dict, hero_map: Dict[str, str]) -> dict:
    """
    Fill hero_name fields in picks_obj using hero_map where missing.
    Mutates a shallow copy and returns it.
    """
    def fill_list(lst: List[dict]):
        for entry in lst:
            hid = entry.get("hero_id")
            if hid is None:
                # try other keys
                hid = entry.get("hero")
            if hid is not None and (entry.get("hero_name") in (None, "", [])):
                name = hero_map.get(str(hid))
                if name:
                    entry["hero_name"] = name

    # Work on copies to avoid surprising callers
    out = json.loads(json.dumps(picks_obj))
    for key in ("radiant_picks", "dire_picks", "radiant_bans", "dire_bans"):
        lst = out.get(key) or []
        fill_list(lst)
    # also try to update picks_bans raw entries
    for pb in out.get("picks_bans", []) or []:
        hid = pb.get("hero_id") or pb.get("hero")
        if hid and not (pb.get("hero_name") or pb.get("hero_string")):
            name = hero_map.get(str(hid))
            if name:
                pb["hero_name"] = name
    return out

# -----------------------
# IO: saving picks and copies
# -----------------------
def save_picks_canonical_and_team_copies(picks_obj: dict, base_out: Path, group_by_team: bool, match_data: dict, skip_existing: bool, verbose: bool = False) -> Tuple[Path, List[Path]]:
    """
    Save picks_obj to base_out/all/match_<id>_picks.json and optionally copy to per-team folders.
    Returns (canonical_path, list_of_team_paths)
    """
    all_dir = base_out / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    mid = picks_obj.get("match_id")
    if mid is not None:
        fname = f"match_{mid}_picks.json"
    else:
        fname = f"{int(time.time())}_match_unknown_picks.json"
    canonical = all_dir / fname

    if skip_existing and canonical.exists():
        if verbose:
            logger.debug("Skipping write of existing picks canonical: %s", canonical)
    else:
        with canonical.open("w", encoding="utf-8") as fh:
            json.dump(picks_obj, fh, ensure_ascii=False, indent=2)
        logger.info("Wrote canonical picks JSON: %s", canonical)

    team_paths: List[Path] = []
    if group_by_team:
        meta = get_team_meta(match_data)
        team_names = [meta["radiant"]["team_name"], meta["dire"]["team_name"]]
        seen = set()
        for name in team_names:
            if not name:
                continue
            safe = safe_name(name)
            if safe in seen:
                continue
            seen.add(safe)
            team_dir = base_out / "teams" / safe
            if not team_dir.exists():
                team_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Created picks team dir: %s", team_dir)
            team_json = team_dir / canonical.name
            try:
                shutil.copy2(canonical, team_json)
                team_paths.append(team_json)
                logger.info("Wrote team picks JSON: %s", team_json)
            except Exception as e:
                logger.warning("Failed to copy picks to team dir %s: %s", team_dir, e)
    return canonical, team_paths

# -----------------------
# Pipeline & aggregation
# -----------------------
def find_match_json_files(matches_dir: Path) -> List[Path]:
    files: List[Path] = []
    if not matches_dir.exists():
        return files
    for p in matches_dir.rglob("*.json"):
        files.append(p)
    return sorted(files)

def process_match_file(match_file: Path, out_dir: Path, group_by_team: bool, skip_existing: bool,
                       hero_map: Dict[str, str], export_picks: bool, verbose: bool = False) -> Tuple[Optional[int], bool, List[Path], Optional[dict]]:
    """
    Process one match file:
     - extract picks
     - complete hero names
     - save per-match picks JSON (canonical) and optionally per-team copies
    Returns (match_id, generated(bool), paths_written, picks_obj)
    """
    try:
        with match_file.open("r", encoding="utf-8") as fh:
            mj = json.load(fh)
    except Exception as e:
        logger.warning("Failed to load JSON %s: %s", match_file, e)
        return None, False, [], None

    match_id = infer_match_id(mj, filename=str(match_file))
    picks_obj = extract_hero_picks(mj)
    # fill hero names using hero_map
    picks_completed = complete_hero_names_for_picks(picks_obj, hero_map)

    out_paths: List[Path] = []
    if export_picks:
        try:
            canonical, team_paths = save_picks_canonical_and_team_copies(picks_completed, out_dir, group_by_team, mj, skip_existing, verbose)
            out_paths.append(canonical)
            out_paths.extend(team_paths)
            return match_id, True, out_paths, picks_completed
        except Exception as e:
            logger.error("Failed to save picks for %s: %s", match_file, e)
            return match_id, False, [], picks_completed
    else:
        # not exporting picks, but return picks for aggregation if requested by caller
        return match_id, False, [], picks_completed

def aggregate_and_write(picks_list: List[dict], out_dir: Path, parquet: bool = True, verbose: bool = False) -> Path:
    """
    Given a list of per-match picks objects, build a match-level table and write to parquet (preferred) or csv.
    Each row: match_id, radiant_team, dire_team, radiant_picks_ids, radiant_picks_names, dire_picks_ids, dire_picks_names, radiant_bans_ids, dire_bans_ids
    """
    rows: List[dict] = []
    for p in picks_list:
        try:
            mid = p.get("match_id")
            # attempt to find team names via picks or leave empty
            # picks object doesn't include team_name, so user can rely on match_to_teams generated separately
            rad_p = p.get("radiant_picks") or []
            dire_p = p.get("dire_picks") or []
            rad_b = p.get("radiant_bans") or []
            dire_b = p.get("dire_bans") or []
            def ids_names(lst):
                ids = [int(x.get("hero_id")) for x in lst if x.get("hero_id") is not None]
                names = [x.get("hero_name") or "" for x in lst]
                return ids, names
            rad_ids, rad_names = ids_names(rad_p)
            dire_ids, dire_names = ids_names(dire_p)
            rad_bids, rad_bnames = ids_names(rad_b)
            dire_bids, dire_bnames = ids_names(dire_b)
            rows.append({
                "match_id": mid,
                "radiant_picks_ids": rad_ids,
                "radiant_picks_names": rad_names,
                "dire_picks_ids": dire_ids,
                "dire_picks_names": dire_names,
                "radiant_bans_ids": rad_bids,
                "dire_bans_ids": dire_bids,
                "raw": json.dumps(p, ensure_ascii=False),
            })
        except Exception:
            continue

    out_dir.mkdir(parents=True, exist_ok=True)
    # write parquet if possible
    dest_parquet = out_dir / "picks.parquet"
    dest_csv = out_dir / "picks.csv"
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows)
        if parquet:
            try:
                df.to_parquet(dest_parquet, index=False)
                logger.info("Wrote aggregated picks parquet: %s", dest_parquet)
                return dest_parquet
            except Exception as e:
                logger.warning("Failed to write parquet (%s), falling back to CSV: %s", dest_parquet, e)
        df.to_csv(dest_csv, index=False)
        logger.info("Wrote aggregated picks csv: %s", dest_csv)
        return dest_csv
    except Exception as e:
        # fallback: write JSON-lines
        dest = out_dir / "picks.jsonl"
        with dest.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote aggregated picks jsonl: %s", dest)
        return dest

# -----------------------
# CLI
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract per-match picks (BP) from local match JSONs and save to data/picks (per-match + per-team + aggregated).")
    p.add_argument("--matches-dir", type=Path, default=DEFAULT_MATCHES_DIR, help="Folder containing match JSONs")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output picks root (default data/picks)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--group-by-team", action="store_true", help="Also copy picks JSON to per-team subfolders under out-dir/teams/<TeamName>")
    p.add_argument("--skip-existing", action="store_true", help="Skip writing canonical picks if it already exists (still may create team copies)")
    p.add_argument("--heroes-file", type=Path, help="Optional local JSON hero map file (id->name or list of {id,name}).")
    p.add_argument("--fetch-heroes", action="store_true", help="Fetch hero list from OpenDota to supplement hero name mapping (requires internet).")
    p.add_argument("--export-aggregate", action="store_true", help="Also generate aggregated picks.parquet (or picks.csv fallback) at out-dir/picks.parquet")
    p.add_argument("--parquet", action="store_true", help="Prefer parquet output for aggregated file (default try parquet then csv)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    matches_dir: Path = args.matches_dir
    out_dir: Path = args.out_dir
    group_by_team = bool(args.group_by_team)
    skip_existing = bool(args.skip_existing)
    workers = max(1, int(args.workers))

    logger.info("Scanning matches dir: %s", matches_dir)
    files = find_match_json_files(matches_dir)
    if not files:
        logger.error("No match JSON files found under %s", matches_dir)
        return 2

    # Build hero map: from local file, from existing matches, and optionally fetch
    hero_map: Dict[str, str] = {}
    if args.heroes_file:
        hero_map.update(load_hero_map_from_file(args.heroes_file))
        logger.info("Loaded hero map from %s (%d entries)", args.heroes_file, len(hero_map))
    # augment from matches
    hm_from_matches = build_hero_map_from_matches(matches_dir)
    if hm_from_matches:
        hero_map.update(hm_from_matches)
        logger.info("Augmented hero map from matches (%d entries)", len(hm_from_matches))
    # optionally fetch
    if args.fetch_heroes:
        fetched = fetch_heroes_from_opendota()
        if fetched:
            hero_map.update(fetched)
            logger.info("Fetched hero map from OpenDota (%d entries)", len(fetched))

    # process matches in parallel
    picks_collected: List[dict] = []
    saved_paths: List[Path] = []
    skipped = 0
    generated = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(process_match_file, f, out_dir, group_by_team, skip_existing, hero_map, True, args.verbose): f for f in files}
        for fut in as_completed(fut_map):
            f = fut_map[fut]
            try:
                mid, did_generate, paths, picks_obj = fut.result()
            except Exception as e:
                logger.warning("Worker failed for %s: %s", f, e)
                continue
            if did_generate:
                generated += 1
                saved_paths.extend(paths)
                if picks_obj:
                    picks_collected.append(picks_obj)
                logger.info("[%s] generated -> %d path(s): %s", mid or f.name, len(paths), ", ".join(str(p) for p in paths))
            else:
                # If returned picks_obj it means it was not written but we can aggregate
                if picks_obj:
                    picks_collected.append(picks_obj)
                if paths:
                    skipped += 1
                    logger.info("[%s] skipped existing -> %d path(s): %s", mid or f.name, len(paths), ", ".join(str(p) for p in paths))
                else:
                    logger.info("[%s] no output (failed or picks not exported)", mid or f.name)

    # write aggregated file if requested
    if args.export_aggregate:
        try:
            agg_path = aggregate_and_write(picks_collected, out_dir, parquet=bool(args.parquet), verbose=args.verbose)
            logger.info("Wrote aggregated picks to %s", agg_path)
        except Exception as e:
            logger.warning("Failed to write aggregated picks: %s", e)

    # write match->teams map (simple)
    match_to_teams: Dict[str, List[str]] = {}
    for p in (out_dir / "all").glob("match_*_picks.json"):
        import re
        m = re.search(r"match_(\d+)_picks\.json", p.name)
        if not m:
            continue
        mid = m.group(1)
        possible = list(matches_dir.rglob(f"*{mid}*.json"))
        if possible:
            try:
                with possible[0].open("r", encoding="utf-8") as fh:
                    mj = json.load(fh)
                rad_name = mj.get("radiant_name") or ""
                dire_name = mj.get("dire_name") or ""
                if not rad_name and isinstance(mj.get("radiant_team"), dict):
                    rad_name = mj.get("radiant_team").get("name") or rad_name
                if not dire_name and isinstance(mj.get("dire_team"), dict):
                    dire_name = mj.get("dire_team").get("name") or dire_name
                match_to_teams[mid] = [rad_name, dire_name]
            except Exception:
                match_to_teams[mid] = []
        else:
            match_to_teams[mid] = []

    try:
        map_path = out_dir / "match_to_teams.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        with map_path.open("w", encoding="utf-8") as fh:
            json.dump(match_to_teams, fh, indent=2, ensure_ascii=False)
        logger.info("Wrote match->teams map to %s", map_path)
    except Exception as e:
        logger.warning("Failed to write match->teams map: %s", e)

    logger.info("Done. generated=%d, skipped_existing=%d, aggregated_matches=%d", generated, skipped, len(picks_collected))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())