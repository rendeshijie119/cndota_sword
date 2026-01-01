#!/usr/bin/env python3
"""
download_obs_logs_from_matches.py (v4)

Read match JSON files from a matches folder (default ./data/matches), extract
obs_log and sen_log placement events and their corresponding "left" (removed)
events, compute per-ward lifetime (left.time - placed.time) when possible, and
save one canonical CSV per match into out_dir/all/match_<id>_wards.csv.

Features (v4):
- Pair placements with removals using JSON removals indexed by ehandle -> key -> rounded cell.
  For each placement we pick the earliest removal_time >= placement_time.
- If no JSON removal is found, we will (optionally) fall back to any existing CSV removed_time entries.
- Optionally fill missing lifetimes with TTL defaults (Observer=360s, Sentry=420s) if --fill-missing-ttl is supplied.
  Default: do NOT fill (you must opt in).
- Reports extraction statistics (placements, matched_json, matched_csv_removal, computed_nonneg, computed_neg, filled, original_with_lifetime).
- Keeps parallel processing (workers) and ability to copy canonical CSV into per-team folders (--group-by-team).
- Safe for CSVs produced by other tools (utf-8-sig) and preserves extra fields.

Usage example:
  python3 download_obs_logs_from_matches.py \
    --matches-dir data/matches --out-dir data/obs_logs \
    --group-by-team --workers 6 --verbose --fill-missing-ttl

CLI flags of interest (defaults):
- --fill-missing-ttl (default OFF)
- --obs-ttl 360.0
- --sen-ttl 420.0
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import shutil
import time
import bisect
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_obs_logs_from_matches")

DEFAULT_MATCHES_DIR = Path("data/matches")
DEFAULT_OUT_DIR = Path("data/obs_logs")
DEFAULT_TIME_START = -60
DEFAULT_TIME_END = 7200
DEFAULT_OBS_TTL = 360.0
DEFAULT_SEN_TTL = 420.0


# ----------------- helpers -----------------
def safe_name(s: Optional[str], max_len: int = 120) -> str:
    if s is None:
        return "unknown"
    s2 = str(s).strip()
    s2 = "".join(ch if (ch.isalnum() or ch in " ._-") else "_" for ch in s2)
    s2 = s2.strip(" _-.")
    if not s2:
        return "unknown"
    if len(s2) > max_len:
        s2 = s2[: max_len - 3] + "..."
    return s2


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
    for side in ("radiant", "dire"):
        if not meta[side]["team_name"]:
            tid = meta[side]["team_id"]
            if tid:
                meta[side]["team_name"] = f"team_{tid}"
            else:
                meta[side]["team_name"] = "unknown"
    return meta


def get_personal_name_from_player(p: dict) -> str:
    try:
        if not isinstance(p, dict):
            return str(p) if p else ""
        if p.get("personaname"):
            return str(p.get("personaname"))
        prof = p.get("profile") or {}
        if isinstance(prof, dict):
            if prof.get("personaname"):
                return str(prof.get("personaname"))
            if prof.get("name"):
                return str(prof.get("name"))
        if p.get("name"):
            return str(p.get("name"))
    except Exception:
        pass
    return ""


def is_radiant(player_slot: int) -> bool:
    try:
        return int(player_slot) < 128
    except Exception:
        return False


def team_for_slot(player_slot: int) -> str:
    return "radiant" if is_radiant(player_slot) else "dire"


def _parse_xy_from_event(ev: dict) -> Optional[Tuple[float, float]]:
    if ev is None:
        return None
    for kx in ("x", "pos_x", "posx"):
        if kx in ev and ev[kx] not in (None, "", "nan"):
            try:
                x = float(ev[kx]); break
            except Exception:
                x = None
    else:
        x = None
    for ky in ("y", "pos_y", "posy"):
        if ky in ev and ev[ky] not in (None, "", "nan"):
            try:
                y = float(ev[ky]); break
            except Exception:
                y = None
    else:
        y = None
    if x is not None and y is not None:
        return (x, y)
    key = ev.get("key") or ev.get("pos_key") or ""
    if isinstance(key, str) and key:
        m = __import__("re").search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", key)
        if m:
            try:
                return (float(m.group(1)), float(m.group(2)))
            except Exception:
                pass
        m2 = __import__("re").search(r"\[(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\]", key)
        if m2:
            try:
                return (float(m2.group(1)), float(m2.group(2)))
            except Exception:
                pass
    return None


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
                out.append(r)
    except Exception:
        pass
    return out


def _extract_numeric_from_row_keys(row: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in row and row[k] not in (None, "", "nan"):
            try:
                return float(row[k])
            except Exception:
                try:
                    s = str(row[k])
                    m = __import__("re").search(r"-?\d+\.?\d*", s)
                    if m:
                        return float(m.group(0))
                except Exception:
                    pass
    return None


# ----------------- core extraction with robust matching -----------------
def collect_ward_events_with_lefts(match_id: Optional[int], match_data: dict, time_start: int, time_end: int,
                                   fill_missing_ttl: bool = False, obs_ttl: float = DEFAULT_OBS_TTL,
                                   sen_ttl: float = DEFAULT_SEN_TTL, verbose: bool = False,
                                   fallback_csv: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Return list of placement rows (one per placement) enriched with 'removed_time' and 'lifetime' (seconds).
    Matching priority:
      1) JSON removals indexed by ehandle -> key -> rounded cell, choose earliest removal >= placement_time
      2) If none found, fallback to removed_time values present in an existing CSV (if fallback_csv provided)
      3) If still none and fill_missing_ttl=True, fill with obs_ttl/sen_ttl depending on ward type
    Negative computed lifetimes are treated as invalid and will be ignored (can be filled by TTL if enabled).
    """
    rows_out: List[Dict[str, Any]] = []
    players = match_data.get("players", []) or []
    team_meta = get_team_meta(match_data)

    # Build lists of removal times per key from JSON
    removals_by_ehandle: Dict[str, List[float]] = defaultdict(list)
    removals_by_key: Dict[str, List[float]] = defaultdict(list)
    removals_by_cell: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    # collect removal events from JSON left logs
    for p in players:
        for ev in (p.get("obs_left_log", []) or []) + (p.get("sen_left_log", []) or []):
            try:
                t = float(ev.get("time"))
            except Exception:
                continue
            e = ev.get("ehandle") or ev.get("entityleft") or ""
            key = (ev.get("key") or ev.get("pos_key") or "").strip()
            xy = _parse_xy_from_event(ev)
            cell = (int(round(xy[0])), int(round(xy[1]))) if xy else None
            if e not in (None, "", False):
                removals_by_ehandle[str(e)].append(t)
            if key:
                removals_by_key[key].append(t)
            if cell is not None:
                removals_by_cell[cell].append(t)

    # optionally incorporate removals from existing CSV fallback
    if fallback_csv is not None and fallback_csv.exists():
        csv_rows = read_csv_rows(fallback_csv)
        for r in csv_rows:
            rt = None
            # possible removal columns
            for k in ("removed_time", "removed", "time_removed", "time_destroyed"):
                if k in r and r.get(k) not in (None, ""):
                    try:
                        rt = float(r.get(k))
                        break
                    except Exception:
                        try:
                            m = __import__("re").search(r"-?\d+\.?\d*", str(r.get(k)))
                            if m:
                                rt = float(m.group(0)); break
                        except Exception:
                            rt = None
            if rt is None:
                continue
            e = r.get("ehandle") or ""
            key = (r.get("key") or "").strip()
            xy = None
            try:
                if r.get("x") not in (None, "") and r.get("y") not in (None, ""):
                    xy = (float(r.get("x")), float(r.get("y")))
            except Exception:
                xy = None
            cell = (int(round(xy[0])), int(round(xy[1]))) if xy else None
            if e not in (None, "", False):
                removals_by_ehandle[str(e)].append(rt)
            if key:
                removals_by_key[key].append(rt)
            if cell is not None:
                removals_by_cell[cell].append(rt)

    # sort lists for bisect lookup
    for lst in removals_by_ehandle.values(): lst.sort()
    for lst in removals_by_key.values(): lst.sort()
    for lst in removals_by_cell.values(): lst.sort()

    def earliest_ge(sorted_list: List[float], tval: float) -> Optional[float]:
        if not sorted_list:
            return None
        idx = bisect.bisect_left(sorted_list, tval)
        if idx < len(sorted_list):
            return sorted_list[idx]
        return None

    # counters for reporting
    total = 0
    matched_json = 0
    matched_csv_removal = 0
    computed_nonneg = 0
    computed_neg = 0
    filled = 0
    orig_with = 0

    # process placements per player
    for p in players:
        p_slot = p.get("player_slot")
        if p_slot is None:
            continue
        side = team_for_slot(p_slot)
        account_id = p.get("account_id", "")
        personal_name = get_personal_name_from_player(p)

        def process_placements(ev_list: list, ward_type: str):
            nonlocal total, matched_json, matched_csv_removal, computed_nonneg, computed_neg, filled, orig_with
            for ev in ev_list or []:
                try:
                    ev_time_raw = ev.get("time")
                    if ev_time_raw is None:
                        continue
                    ev_time = float(ev_time_raw)
                except Exception:
                    continue
                if ev_time < time_start or ev_time > time_end:
                    continue
                total += 1

                row: Dict[str, Any] = {
                    "match_id": match_id,
                    "radiant_or_dire": side,
                    "team_id": team_meta.get(side, {}).get("team_id", ""),
                    "team_name": team_meta.get(side, {}).get("team_name", ""),
                    "ward_type": ward_type,
                    "account_id": account_id,
                    "personal_name": personal_name,
                    "player_slot": p_slot,
                    "time": ev_time,
                }
                # copy original event fields
                for k, v in ev.items():
                    row[k] = v

                # record if placement row already had a lifetime-like field
                if _extract_numeric_from_row_keys(ev, ["lifetime", "life", "duration", "time_alive", "entity_life"]) is not None:
                    orig_with += 1

                # try to find earliest removal >= placement in JSON-indexed lists
                removed_time = None
                matched_by = ""
                e = ev.get("ehandle") or ev.get("entityleft") or ""
                key = (ev.get("key") or ev.get("pos_key") or "").strip()
                xy = _parse_xy_from_event(ev)
                cell = (int(round(xy[0])), int(round(xy[1]))) if xy else None

                if e and str(e) in removals_by_ehandle:
                    removed_time = earliest_ge(removals_by_ehandle[str(e)], ev_time)
                    matched_by = "json_ehandle"
                if removed_time is None and key and key in removals_by_key:
                    removed_time = earliest_ge(removals_by_key[key], ev_time)
                    matched_by = "json_key"
                if removed_time is None and cell and cell in removals_by_cell:
                    removed_time = earliest_ge(removals_by_cell[cell], ev_time)
                    matched_by = "json_cell"

                # if still none, we'll try CSV-sourced removal lists (they're already merged into lists above)
                # detect whether the chosen matched_by came from json or was from CSV fallback by checking presence in original JSON lists
                # (we can't easily distinguish after merging fallback; matched_by will remain json_* if found via JSON lists)

                lifetime = None
                if removed_time is not None:
                    lifetime = float(removed_time) - float(ev_time)
                    if lifetime >= 0:
                        computed_nonneg += 1
                    else:
                        computed_neg += 1
                        # treat negative as invalid
                        if verbose:
                            logger.warning("[warn] negative lifetime for match %s ehandle=%s key=%s placed=%s removed=%s",
                                           match_id, e, key, ev_time, removed_time)
                        lifetime = None

                # duration fields on placement override
                duration = _extract_numeric_from_row_keys(ev, ["duration", "life", "lifetime", "time_alive", "entity_life"])
                if duration is not None:
                    lifetime = float(duration)

                # fill missing using TTL if requested
                if lifetime is None and fill_missing_ttl:
                    if ward_type == "obs":
                        lifetime = float(obs_ttl)
                    elif ward_type == "sen":
                        lifetime = float(sen_ttl)
                    if lifetime is not None:
                        filled += 1

                if matched_by.startswith("json"):
                    matched_json += 1
                else:
                    # If we found removed_time via fallback CSV we would still have matched_by empty here;
                    # we try to detect fallback match by checking presence in CSV data - but we appended CSV removals into the same lists,
                    # so treat non-json matched as csv fallback when appropriate (best-effort)
                    if removed_time is not None:
                        # determine if removed_time exists in JSON-only sets (simple heuristic)
                        is_in_json = False
                        if e and str(e) in removals_by_ehandle:
                            # this list may contain both json and csv entries; we assume matched_by json if there was a JSON left event with same time
                            # not perfect but acceptable; leave matched_by empty otherwise
                            pass
                        # if matched_by still empty, treat as csv fallback
                        if not matched_by:
                            matched_by = "csv_fallback"
                            matched_csv_removal += 1

                row["removed_time"] = removed_time if removed_time is not None else ""
                row["lifetime"] = lifetime if lifetime is not None else ""
                row["_matched_by"] = matched_by
                rows_out.append(row)

        process_placements(p.get("obs_log", []), "obs")
        process_placements(p.get("sen_log", []), "sen")

    if verbose:
        logger.info("[extract] match %s: placements=%d, matched_json=%d, computed_nonneg=%d, computed_neg=%d, filled=%d, original_with_lifetime=%d",
                    match_id, total, matched_json, computed_nonneg, computed_neg, filled, orig_with)
    return rows_out


# ----------------- CSV saving -----------------
def save_csv_canonical_and_team_copies(match_id: Optional[int], rows: List[Dict[str, Any]], base_out: Path,
                                       group_by_team: bool, match_data: dict, verbose: bool = False) -> Tuple[Path, List[Path]]:
    all_dir = base_out / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    if match_id is not None:
        csv_name = f"match_{match_id}_wards.csv"
    else:
        csv_name = f"{int(time.time())}_match_unknown_wards.csv"
    canonical = all_dir / csv_name

    # canonical columns: ensure removed_time and lifetime exist
    default_fields = [
        "match_id", "radiant_or_dire", "team_id", "team_name", "ward_type",
        "account_id", "personal_name", "player_slot", "time", "removed_time", "lifetime",
        "type", "x", "y", "z", "entityleft", "ehandle", "key",
    ]
    cols = list(default_fields)
    extra = set()
    for r in rows:
        extra.update(r.keys())
    for k in sorted(extra):
        if k not in cols:
            cols.append(k)

    try:
        # prefer pandas if available for nicer CSV
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(canonical, index=False)
    except Exception:
        with canonical.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            for r in rows:
                writer.writerow([r.get(c, "") for c in cols])

    if verbose:
        logger.info("Wrote canonical CSV: %s", canonical)

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
            try:
                if not team_dir.exists():
                    team_dir.mkdir(parents=True, exist_ok=True)
                    if verbose:
                        logger.info("Created team dir: %s", team_dir)
                team_csv = team_dir / canonical.name
                shutil.copy2(canonical, team_csv)
                team_paths.append(team_csv)
                if verbose:
                    logger.info("Wrote team CSV: %s", team_csv)
            except Exception as e:
                logger.warning("Failed to write/copy team CSV %s: %s", team_dir, e)

    return canonical, team_paths


# ----------------- file discovery / processing -----------------
def find_match_json_files(matches_dir: Path) -> List[Path]:
    files: List[Path] = []
    if not matches_dir.exists():
        return files
    for p in matches_dir.rglob("*.json"):
        files.append(p)
    return sorted(files)


def infer_match_id(match_data: dict, filename: Optional[str] = None) -> Optional[int]:
    mid = match_data.get("match_id") or match_data.get("matchid") or match_data.get("matchId")
    try:
        if mid is not None:
            return int(mid)
    except Exception:
        pass
    if filename:
        fn = Path(filename).name
        m = __import__("re").search(r"(\d{6,})", fn)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def process_match_file(match_file: Path, out_dir: Path, time_start: int, time_end: int, skip_existing: bool,
                       group_by_team: bool, fill_missing_ttl: bool, obs_ttl: float, sen_ttl: float,
                       workers: int, verbose: bool = False) -> Tuple[int, bool, List[Path]]:
    try:
        with match_file.open("r", encoding="utf-8") as fh:
            match_data = json.load(fh)
    except Exception as e:
        logger.warning("Failed to load JSON %s: %s", match_file, e)
        return (0, False, [])

    match_id = infer_match_id(match_data, filename=str(match_file))
    canonical_path = out_dir / "all" / (f"match_{match_id}_wards.csv" if match_id is not None else f"{match_file.stem}_wards.csv")

    if skip_existing and canonical_path.exists():
        # create team copies if requested
        created = []
        if group_by_team:
            try:
                _, created = create_team_copies_from_existing_csv(out_dir, match_id, match_data, group_by_team, verbose)
            except Exception:
                created = []
        paths = [canonical_path] + created
        if verbose:
            logger.debug("[%s] skipped existing -> %s", match_id or match_file.name, ", ".join(str(p) for p in paths))
        return (match_id or 0, False, paths)

    # compute rows using JSON, with fallback to existing CSV when matching if present
    fallback_csv = canonical_path if canonical_path.exists() else None
    rows = collect_ward_events_with_lefts(match_id, match_data, time_start, time_end,
                                          fill_missing_ttl=fill_missing_ttl, obs_ttl=obs_ttl, sen_ttl=sen_ttl,
                                          verbose=verbose, fallback_csv=fallback_csv)
    try:
        canonical, team_paths = save_csv_canonical_and_team_copies(match_id, rows, out_dir, group_by_team, match_data, verbose)
        paths = [canonical] + team_paths
        if verbose:
            logger.info("[%s] saved -> %s", match_id or match_file.name, ", ".join(str(p) for p in paths))
        return (match_id or 0, True, paths)
    except Exception as e:
        logger.error("Failed to save CSV for %s: %s", match_file, e)
        return (match_id or 0, False, [])


def create_team_copies_from_existing_csv(base_out: Path, match_id: Optional[int], match_data: dict,
                                         group_by_team: bool, verbose: bool = False) -> Tuple[Path, List[Path]]:
    if match_id is None:
        raise FileNotFoundError("canonical CSV name unknown")
    canonical = base_out / "all" / f"match_{match_id}_wards.csv"
    if not canonical.exists():
        raise FileNotFoundError(canonical)
    created: List[Path] = []
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
            try:
                if not team_dir.exists():
                    team_dir.mkdir(parents=True, exist_ok=True)
                    if verbose:
                        logger.info("Created team dir: %s", team_dir)
                team_csv = team_dir / canonical.name
                if not team_csv.exists():
                    shutil.copy2(canonical, team_csv)
                    created.append(team_csv)
                    if verbose:
                        logger.info("Copied canonical to team CSV: %s", team_csv)
            except Exception as e:
                logger.warning("Failed to create team copy %s: %s", team_dir, e)
    return canonical, created


# ----------------- CLI and main -----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract obs/sen ward events from local match JSONs and save CSVs per-match and per-team.")
    p.add_argument("--matches-dir", type=Path, default=DEFAULT_MATCHES_DIR, help="Root folder containing match JSONs (default ./data/matches)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output folder for per-match ward CSVs (default ./data/obs_logs)")
    p.add_argument("--start", type=int, default=DEFAULT_TIME_START, help="Start time (sec) inclusive (default -60)")
    p.add_argument("--end", type=int, default=DEFAULT_TIME_END, help="End time (sec) inclusive (default 7200)")
    p.add_argument("--skip-existing", action="store_true", help="Skip CSV generation if the canonical CSV already exists; will still create missing team copies.")
    p.add_argument("--group-by-team", action="store_true", help="Also copy canonical CSV into per-team folders under out-dir/teams/<team_name>.")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers for processing matches.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    # v4 lifetime options
    p.add_argument("--fill-missing-ttl", action="store_true",
                   help="If enabled, fill missing lifetime values with default TTL per ward type (obs/sen).")
    p.add_argument("--obs-ttl", type=float, default=DEFAULT_OBS_TTL,
                   help="Default TTL (seconds) for observer wards when filling missing lifetime (default 360).")
    p.add_argument("--sen-ttl", type=float, default=DEFAULT_SEN_TTL,
                   help="Default TTL (seconds) for sentry wards when filling missing lifetime (default 420).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    matches_dir: Path = args.matches_dir
    out_dir: Path = args.out_dir
    time_start: int = args.start
    time_end: int = args.end
    skip_existing: bool = args.skip_existing
    group_by_team: bool = args.group_by_team
    workers: int = max(1, int(args.workers))
    fill_missing_ttl: bool = args.fill_missing_ttl
    obs_ttl: float = float(args.obs_ttl)
    sen_ttl: float = float(args.sen_ttl)

    logger.info("Scanning matches dir: %s", matches_dir)
    files = find_match_json_files(matches_dir)
    if not files:
        logger.error("No match JSON files found under %s", matches_dir)
        return 2

    generated = 0
    skipped = 0
    created_team_copies_total = 0
    saved_paths_all: List[Path] = []

    # process in parallel
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(process_match_file, f, out_dir, time_start, time_end, skip_existing, group_by_team,
                                fill_missing_ttl, obs_ttl, sen_ttl, workers, args.verbose): f for f in files}
        for fut in as_completed(future_map):
            match_file = future_map[fut]
            try:
                mid, did_generate, paths = fut.result()
            except Exception as e:
                logger.warning("Worker failed for %s: %s", match_file, e)
                continue

            if did_generate:
                generated += 1
                saved_paths_all.extend(paths)
                created_team_copies_total += max(0, len(paths) - 1)
                logger.info("[%s] generated -> %d path(s): %s", mid or match_file.name, len(paths), ", ".join(str(p) for p in paths))
            else:
                skipped += 1
                if paths:
                    logger.info("[%s] skipped existing -> %d path(s): %s", mid or match_file.name, len(paths), ", ".join(str(p) for p in paths))
                else:
                    logger.info("[%s] no output (failed or empty)", mid or match_file.name)

    logger.info("Done. generated=%d, skipped_existing=%d, created_team_copies=%d, csv_written=%d",
                generated, skipped, created_team_copies_total, len(saved_paths_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())