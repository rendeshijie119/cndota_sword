#!/usr/bin/env python3
"""
download_gold_timeline.py

Extract and standardize gold timeline CSV files for matches.

Produces a standardized CSV per match with columns:
  match_id, time, radiant_gold, dire_gold, radiant_team_name, dire_team_name, gold_diff, radiant_gold_adv

Behavior:
- Reads raw match JSON files (match_<id>.json) from --in dir OR fetches from OpenDota if --fetch.
- Extracts absolute gold arrays if available, or radiant advantage arrays if that's all that's present.
- Detects whether the extracted 'time' series is in minutes and (when likely) converts it to seconds.
- Writes standardized CSV match_<id>_gold_std.csv to --out dir.

Usage example:
  python3 download_gold_timeline.py --in ./obs_logs/json --out ./data/gold_timeline --match-file ids.txt --fetch --verbose

"""
from pathlib import Path
import argparse
import json
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None

import requests
import math

OPENDOTA_MATCH_URL = "https://api.opendota.com/api/matches/{match_id}"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def fetch_match_json(match_id: int, timeout: int = 30) -> Dict[str, Any]:
    url = OPENDOTA_MATCH_URL.format(match_id=match_id)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_team_meta(match_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Return dict: {'radiant_name': str, 'dire_name': str}
    Tries several possible fields present in different JSON formats.
    """
    rad_name = match_data.get("radiant_name") or match_data.get("radiant_team") or ""
    dire_name = match_data.get("dire_name") or match_data.get("dire_team") or ""
    # some JSONs carry nested 'radiant_team' object
    for side in ("radiant", "dire"):
        v = match_data.get(f"{side}_team")
        if isinstance(v, dict):
            if side == "radiant":
                rad_name = rad_name or v.get("name") or ""
            else:
                dire_name = dire_name or v.get("name") or ""
    return {"radiant_name": str(rad_name), "dire_name": str(dire_name)}


def guess_time_array_from_length(length: int) -> List[float]:
    return [float(i) for i in range(length)]


def extract_gold_series(match_data: Dict[str, Any]) -> Tuple[Optional[List[float]], Dict[str, List[float]]]:
    """
    Return (times, series_dict)
    series_dict keys may include:
      - 'radiant_gold', 'dire_gold'  (absolute)
      - 'radiant_gold_adv'           (advantage)
      - other keys containing 'gold' or 'adv'
    times may be None (we'll guess one-per-second)
    """
    series: Dict[str, List[float]] = {}
    times: Optional[List[float]] = None

    # explicit time arrays
    for key in ("time", "times", "t"):
        if key in match_data:
            try:
                arr = match_data[key]
                if isinstance(arr, list) and len(arr) > 0:
                    times = [float(x) for x in arr]
                    break
            except Exception:
                pass

    # direct radiant/dire arrays
    if "radiant_gold" in match_data and "dire_gold" in match_data:
        try:
            rg = [float(x) for x in match_data["radiant_gold"]]
            dg = [float(x) for x in match_data["dire_gold"]]
            series["radiant_gold"] = rg
            series["dire_gold"] = dg
            if times is None:
                times = guess_time_array_from_length(max(len(rg), len(dg)))
            return times, series
        except Exception:
            pass

    # radiant advantage
    if "radiant_gold_adv" in match_data:
        try:
            adv = [float(x) for x in match_data["radiant_gold_adv"]]
            series["radiant_gold_adv"] = adv
            if times is None:
                times = guess_time_array_from_length(len(adv))
            return times, series
        except Exception:
            pass

    # search top-level for any gold/adv arrays
    for k, v in match_data.items():
        if isinstance(k, str) and ("gold" in k.lower() or "adv" in k.lower()):
            if isinstance(v, list) and len(v) > 0:
                try:
                    series[k] = [float(x) for x in v]
                    if times is None:
                        times = guess_time_array_from_length(len(v))
                except Exception:
                    pass

    # recursive search as last resort
    if not series:
        def rec_find(obj: Any, found: Dict[str, List[float]]):
            if isinstance(obj, dict):
                for kk, vv in obj.items():
                    if isinstance(kk, str) and ("gold" in kk.lower() or "adv" in kk.lower()):
                        if isinstance(vv, list) and vv:
                            try:
                                found[kk] = [float(x) for x in vv]
                            except Exception:
                                pass
                    rec_find(vv, found)
            elif isinstance(obj, list):
                for it in obj:
                    rec_find(it, found)
        rec_find(match_data, series)
        if series and times is None:
            first_k = next(iter(series))
            times = guess_time_array_from_length(len(series[first_k]))

    return times, series


def _times_look_like_minutes(times: List[float]) -> bool:
    """
    Heuristic: decide whether a times array is expressed in minutes (0..~N where N ~ game minutes)
    rather than seconds. Return True if likely minutes.

    Heuristic rules:
    - If max time <= 300 and len(times) >= 30 => likely minutes series (e.g., 0..69).
    - If max time <= 120 and len(times) >= 10 and len(times) roughly equals max+1 (within small tol).
    - Otherwise assume seconds.
    """
    if not times:
        return False
    # filter NaN
    vals = [float(t) for t in times if t is not None and not (isinstance(t, float) and math.isnan(t))]
    if not vals:
        return False
    mx = max(vals)
    n = len(vals)
    # rule 1
    if mx <= 300 and n >= 30:
        return True
    # rule 2
    if mx <= 120 and n >= 10 and abs((mx + 1) - n) <= max(2, int(0.05 * (mx + 1))):
        return True
    return False


def _maybe_convert_rows_time_minutes_to_seconds(out_rows: List[Dict[str, Any]], verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Inspect out_rows' time column and, if likely in minutes, convert to seconds (multiply by 60).
    Returns a possibly-modified copy of out_rows.
    """
    if not out_rows:
        return out_rows
    times = []
    for r in out_rows:
        try:
            times.append(float(r.get("time", 0.0)))
        except Exception:
            times.append(0.0)
    # if any time already large (>1000), assume seconds
    if any(t > 1000 for t in times):
        if verbose:
            print("[time-detect] times appear to already be in seconds (found value >1000).")
        return out_rows
    # apply heuristic
    if _times_look_like_minutes(times):
        if verbose:
            print("[time-detect] times look like minutes -> converting to seconds (x60).")
        new_rows = []
        for r in out_rows:
            new_r = dict(r)
            try:
                new_r["time"] = float(r.get("time", 0.0)) * 60.0
            except Exception:
                new_r["time"] = r.get("time", 0.0)
            new_rows.append(new_r)
        return new_rows
    else:
        if verbose:
            print("[time-detect] times appear to be in seconds (no conversion).")
        return out_rows


def write_standard_gold_csv(match_id: int, match_data: Dict[str, Any], out_dir: Path, verbose: bool = False) -> bool:
    """
    Produce a standardized CSV with columns:
    match_id, time, radiant_gold, dire_gold, radiant_team_name, dire_team_name, gold_diff, radiant_gold_adv
    Returns True if written.
    """
    times, series = extract_gold_series(match_data)
    if not series:
        if verbose:
            print(f"[warn] match {match_id}: no gold-like series found.")
        return False

    teams = get_team_meta(match_data)
    rad_name = teams.get("radiant_name", "")
    dire_name = teams.get("dire_name", "")

    out_rows = []
    # if absolute series available:
    if "radiant_gold" in series and "dire_gold" in series:
        rg = series["radiant_gold"]
        dg = series["dire_gold"]
        L = max(len(rg), len(dg))
        if times is None:
            times = guess_time_array_from_length(L)
        for i in range(L):
            t = float(times[i]) if i < len(times) else float(i)
            rgv = float(rg[i]) if i < len(rg) else float(rg[-1])
            dgv = float(dg[i]) if i < len(dg) else float(dg[-1])
            diff = rgv - dgv
            out_rows.append({
                "match_id": match_id,
                "time": t,
                "radiant_gold": rgv,
                "dire_gold": dgv,
                "radiant_team_name": rad_name,
                "dire_team_name": dire_name,
                "gold_diff": diff,
                "radiant_gold_adv": diff
            })
    else:
        # pick an adv or other series (prefer radiant_gold_adv)
        adv_key = None
        if "radiant_gold_adv" in series:
            adv_key = "radiant_gold_adv"
        else:
            # pick first series available
            adv_key = next(iter(series.keys()))
        arr = series[adv_key]
        if times is None:
            times = guess_time_array_from_length(len(arr))
        for i, v in enumerate(arr):
            t = float(times[i]) if i < len(times) else float(i)
            adv_val = float(v)
            out_rows.append({
                "match_id": match_id,
                "time": t,
                "radiant_gold": "",
                "dire_gold": "",
                "radiant_team_name": rad_name,
                "dire_team_name": dire_name,
                "gold_diff": adv_val,
                "radiant_gold_adv": adv_val
            })

    # Detect & convert times if they look like minutes -> seconds
    out_rows = _maybe_convert_rows_time_minutes_to_seconds(out_rows, verbose=verbose)

    # write to out_dir as match_<id>_gold_std.csv
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"match_{match_id}_gold_std.csv"
    if pd:
        df = pd.DataFrame(out_rows, columns=["match_id", "time", "radiant_gold", "dire_gold",
                                             "radiant_team_name", "dire_team_name",
                                             "gold_diff", "radiant_gold_adv"])
        df.to_csv(out_path, index=False)
    else:
        # fallback csv writer
        import csv
        keys = ["match_id", "time", "radiant_gold", "dire_gold",
                "radiant_team_name", "dire_team_name", "gold_diff", "radiant_gold_adv"]
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for r in out_rows:
                writer.writerow(r)
    if verbose:
        print(f"[write] match {match_id}: wrote standardized gold CSV -> {out_path} ({len(out_rows)} rows)")
    return True


def parse_match_ids_from_file(path: Path) -> List[int]:
    ids = []
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            for token in ln.replace(",", " ").split():
                try:
                    ids.append(int(token))
                except ValueError:
                    print(f"Warning: skipping invalid match id token '{token}' in {path}", file=sys.stderr)
    return ids


def parse_match_ids_arg(arg: Optional[str], in_dir: Path, match_file: Optional[Path]) -> List[int]:
    ids: List[int] = []
    # priority: match_file -> arg (comma list) -> scan in_dir
    if match_file:
        if not match_file.exists():
            print(f"Match file {match_file} not found.", file=sys.stderr)
            return []
        return parse_match_ids_from_file(match_file)

    if arg:
        toks = [t.strip() for t in arg.split(",") if t.strip()]
        for t in toks:
            try:
                ids.append(int(t))
            except Exception:
                print(f"[warn] skipping invalid match id token: {t}")
        return ids

    # gather from files in in_dir
    for p in sorted(in_dir.iterdir()):
        if p.is_file() and p.name.startswith("match_") and p.name.endswith(".json"):
            try:
                mid = int(p.name[len("match_"):-len(".json")])
                ids.append(mid)
            except Exception:
                continue
    return ids


def process_match(match_id: int, in_dir: Path, out_dir: Path, fetch_missing: bool, verbose: bool):
    json_path = in_dir / f"match_{match_id}.json"
    match_data = load_json(json_path)
    if match_data is None:
        if fetch_missing:
            try:
                if verbose:
                    print(f"[fetch] match {match_id} from OpenDota ...")
                match_data = fetch_match_json(match_id)
                # save a copy into in_dir for caching
                try:
                    with json_path.open("w", encoding="utf-8") as fh:
                        json.dump(match_data, fh, ensure_ascii=False, indent=2)
                    if verbose:
                        print(f"[cache] saved fetched json to {json_path}")
                except Exception:
                    pass
            except Exception as e:
                print(f"[error] cannot fetch match {match_id}: {e}", file=sys.stderr)
                return
        else:
            if verbose:
                print(f"[skip] match {match_id}: json not found and --fetch not set.")
            return
    else:
        if verbose:
            print(f"[load] match {match_id} json loaded from {json_path}")

    try:
        ok = write_standard_gold_csv(match_id, match_data, out_dir, verbose=verbose)
        if not ok and verbose:
            print(f"[info] match {match_id}: no standardized gold CSV produced.")
    except Exception as e:
        print(f"[error] match {match_id} extraction failed: {e}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Extract standardized gold timeline CSVs from raw match JSONs (or fetch from OpenDota).")
    p.add_argument("--in", dest="in_dir", type=Path, default=Path("./output"), help="Input dir containing match_<id>.json files")
    p.add_argument("--out", dest="out_dir", type=Path, default=Path("./data/gold_timeline"), help="Output dir for standardized gold CSVs")
    p.add_argument("--match-ids", type=str, default=None, help="Comma-separated match ids (overrides scanning input dir)")
    p.add_argument("--match-file", type=Path, default=None, help="Path to a text file listing match ids (one per line). '#' starts a comment.")
    p.add_argument("--fetch", action="store_true", help="If JSON not found locally, fetch match JSON from OpenDota and cache it")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    match_ids = parse_match_ids_arg(args.match_ids, in_dir, args.match_file)
    if not match_ids:
        print("No match ids found. Provide --match-ids or --match-file, or put match_<id>.json files in --in dir.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"[run] processing {len(match_ids)} matches with fetch={args.fetch}")

    for mid in match_ids:
        process_match(mid, in_dir, out_dir, fetch_missing=args.fetch, verbose=args.verbose)
        if args.fetch:
            time.sleep(0.25)


if __name__ == "__main__":
    main()