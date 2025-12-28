#!/usr/bin/env python3
"""
check_hero_coverage.py

Usage:
  python3 check_hero_coverage.py --matches-dir data/matches --results-dir results/falcons_malr1ne/Team Falcons \
      --account 898455820 --hero-map data/heros_mapping/hero_map.csv

Outputs a summary of heroes the account played (from JSONs) and heroes that have plotting outputs.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

def read_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

def load_hero_map(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not path:
        return out
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rdr = csv.DictReader(fh)
            rows = list(rdr)
            if rows:
                # try find id/name columns
                sample = rows[0]
                id_keys = [k for k in sample.keys() if k and k.strip().lower() in ("id","hero_id","heroid","hero")]
                name_keys = [k for k in sample.keys() if k and k.strip().lower() in ("hero_name","localized_name","name")]
                cn_keys = [k for k in sample.keys() if k and k.strip().lower() in ("hero_name_cn","name_cn","cn")]
                if id_keys:
                    idk = id_keys[0]
                    namek = name_keys[0] if name_keys else None
                    cnk = cn_keys[0] if cn_keys else None
                    for r in rows:
                        hid = r.get(idk)
                        if hid is None or str(hid).strip()=="":
                            continue
                        hid_s = str(hid).strip()
                        hero_name = (r.get(namek) or "").strip() if namek else hid_s
                        hero_name_cn = (r.get(cnk) or "").strip() if cnk else hero_name
                        out[hid_s] = {"hero_name": hero_name or hid_s, "hero_name_cn": hero_name_cn or hero_name}
                    return out
            # fallback simple csv rows
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rdr = csv.reader(fh)
            for row in rdr:
                if not row: continue
                hid = str(row[0]).strip()
                name = row[1].strip() if len(row)>1 else hid
                cn = row[2].strip() if len(row)>2 else name
                out[hid] = {"hero_name": name, "hero_name_cn": cn}
    except Exception:
        pass
    return out

def scan_matches_for_account(matches_dir: Path, account: str) -> Tuple[Dict[str,int], Dict[str,List[str]]]:
    # returns hero_id -> count, hero_id -> list of sample match ids
    counts: Dict[str,int] = {}
    samples: Dict[str,List[str]] = {}
    for p in matches_dir.rglob("*.json"):
        mj = read_json_optional(p)
        if not mj:
            continue
        mid = str(mj.get("match_id") or mj.get("matchid") or p.stem)
        players = mj.get("players") or []
        for pl in players:
            acc = pl.get("account_id")
            if acc is None:
                continue
            if str(acc) == str(account):
                # found player
                hid = pl.get("hero_id") or pl.get("hero")
                if hid is None:
                    continue
                hid_s = str(hid)
                counts[hid_s] = counts.get(hid_s, 0) + 1
                samples.setdefault(hid_s, []).append(mid)
                break
    return counts, samples

def scan_results_for_heroes(results_dir: Path) -> Set[str]:
    """
    Look for files named like hero_<heroid>_... in the results tree.
    Return set of hero ids found.
    """
    out: Set[str] = set()
    if not results_dir.exists():
        return out
    for p in results_dir.rglob("hero_*"):
        name = p.name
        m = re.match(r"hero_(\d+)_", name)
        if m:
            out.add(m.group(1))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matches-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--hero-map", type=Path, default=None)
    args = p.parse_args()

    hero_map = load_hero_map(args.hero_map) if args.hero_map else {}

    counts, samples = scan_matches_for_account(args.matches_dir, str(args.account))
    plotted = scan_results_for_heroes(args.results_dir)

    all_played = set(counts.keys())
    in_plots = plotted
    missing = sorted(list(all_played - in_plots), key=lambda x: (-counts.get(x,0), x))
    extra = sorted(list(in_plots - all_played))

    print("Account:", args.account)
    print(f"Matches scanned: {len(list(args.matches_dir.rglob('*.json')))} (note: this is number of files in folder)")
    print("")
    print("Heroes played (id -> count):")
    for hid, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        name = hero_map.get(hid, {}).get("hero_name_cn") or hero_map.get(hid, {}).get("hero_name") or hid
        sample_matches = samples.get(hid, [])[:5]
        print(f"  {hid} ({name}) : {cnt} matches  sample_match_ids={sample_matches}")
    print("")
    print("Heroes with plot outputs (from results dir):")
    for hid in sorted(in_plots):
        name = hero_map.get(hid, {}).get("hero_name_cn") or hero_map.get(hid, {}).get("hero_name") or hid
        print(f"  {hid} ({name})")
    print("")
    print("Missing (played but NOT plotted):")
    if not missing:
        print("  None — all played heroes have plot outputs.")
    else:
        for hid in missing:
            name = hero_map.get(hid, {}).get("hero_name_cn") or hero_map.get(hid, {}).get("hero_name") or hid
            print(f"  {hid} ({name})  count={counts.get(hid,0)} sample={samples.get(hid)[:5]}")
    print("")
    print("Extra (plotted but not found in matches):")
    if not extra:
        print("  None")
    else:
        for hid in extra:
            name = hero_map.get(hid, {}).get("hero_name_cn") or hero_map.get(hid, {}).get("hero_name") or hid
            print(f"  {hid} ({name})")

if __name__ == "__main__":
    main()