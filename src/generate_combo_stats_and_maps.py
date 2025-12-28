#!/usr/bin/env python3
"""
generate_combo_stats_and_maps.py

Scan per-team picks and per-team wards, enumerate hero 3-combinations (candidate:
mid + pos4 + pos5) that appear for each team across matches, aggregate stats and
optionally produce ward heatmaps for the top combos.

Design notes / heuristics:
- For each team folder under {picks_dir}/teams/<TeamSafe>/, each match file
  match_<id>_picks.json is inspected. Team side (radiant/dire) is inferred by
  reading the match JSON in matches_dir (if available) and comparing team names.
- We try to detect the true mid hero for the team in a match by scanning match
  JSON players[] for fields commonly used to indicate mid:
    - player.lane_role == 2 (common in OpenDota).
    - player.get("is_mid") truthy or player.get("role") == "mid"
  If a detected mid exists and its hero_id is in the team's picks, we mark it.
- For each team-match we enumerate all unordered 3-combinations of the team's
  picked heroes (C(5,3) = 10 combos when 5 picks). For each triple we record:
    - appearances (how many matches this exact triple occurred for the team),
    - wins (how many of those matches the team won),
    - mid_detection stats: how often we were able to detect a mid hero inside the triple,
      and how often the detected mid equals a specific hero in the triple.
  We also store sample match ids and the (match,side) pairs for generating ward maps.

Outputs:
- CSV summary: <out_dir>/combo_summary.csv
  columns: team,combo_ids,combo_names,appearances,wins,win_rate,mid_detected_count,mid_detected_matches_sample
- JSON detail: <out_dir>/combo_details.json  (dumps full dictionary of combos -> metadata)
- If --generate-maps, for top-N combos per team (by appearances) generate:
    <out_dir>/maps/<team_safe>/combo_<joined_ids>_heatmap.png
  Uses per-team ward CSV files under wards_dir/teams/<TeamSafe>/match_<id>_wards.csv if present,
  otherwise falls back to wards_dir/all/match_<id>_wards.csv.

Usage example:
  python generate_combo_stats_and_maps.py --picks-dir data/picks --wards-dir data/obs_logs \
    --matches-dir data/matches --hero-map data/hero_map.csv --out results/combo_stats --top-maps 10 --generate-maps --verbose

Dependencies:
  python standard libs + pillow, numpy, matplotlib (for maps)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
except Exception:
    np = None  # plotting disabled if libs missing

# ---------- helpers ----------
def safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in " ._-") else "_" for ch in str(s)).strip("_")[:150]

def load_hero_map_csv(path: Path) -> Tuple[Dict[str,str], Dict[str,str]]:
    id_to_name, name_to_id = {}, {}
    if not path.exists():
        return id_to_name, name_to_id
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            hid = str(r.get("id") or "").strip()
            loc = (r.get("localized_name") or "").strip()
            cn = (r.get("hero_name") or loc).strip()
            if hid:
                id_to_name[hid] = cn
                name_to_id[cn.lower()] = hid
                if loc and loc.lower() != cn.lower():
                    name_to_id[loc.lower()] = hid
    return id_to_name, name_to_id

def read_json(path: Path) -> Optional[Dict[str,Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

def read_csv_rows(path: Path) -> List[Dict[str,Any]]:
    out = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                out.append(r)
    except Exception:
        pass
    return out

# Try to detect mid hero from match JSON players[] heuristics
def detect_mid_from_match_json(match_json: Dict[str,Any], team_safe: str) -> Optional[str]:
    """
    Returns hero_id (string) that is likely mid for the given team (if detectable),
    otherwise None.
    Strategy:
      - scan players[] for lane_role==2 OR is_mid True OR role == 'mid'
      - ensure the hero belongs to the team by comparing player_slot side with team name
    """
    players = match_json.get("players") or []
    # determine radiant/dire team names to map team_safe -> side
    try:
        rad_name = match_json.get("radiant_name") or ""
        dire_name = match_json.get("dire_name") or ""
        # fallback nested
        if not rad_name and isinstance(match_json.get("radiant_team"), dict):
            rad_name = match_json.get("radiant_team").get("name") or rad_name
        if not dire_name and isinstance(match_json.get("dire_team"), dict):
            dire_name = match_json.get("dire_team").get("name") or dire_name
    except Exception:
        rad_name, dire_name = "", ""
    safe_rad = safe_name(str(rad_name))
    safe_dire = safe_name(str(dire_name))
    team_side = None
    if team_safe == safe_rad:
        team_side = "radiant"
    elif team_safe == safe_dire:
        team_side = "dire"
    # helper to check player's side
    def player_side_from_slot(slot) -> str:
        try:
            s = int(slot)
            return "radiant" if s < 128 else "dire"
        except Exception:
            return "unknown"
    # collect candidates
    candidates = []
    for pl in players:
        try:
            # prefer lane_role == 2
            if pl.get("lane_role") == 2 or pl.get("is_mid") or (str(pl.get("role") or "").lower() == "mid"):
                hid = pl.get("hero_id") or pl.get("hero")
                slot = pl.get("player_slot")
                side = player_side_from_slot(slot)
                if hid is None:
                    continue
                if team_side and side != team_side:
                    continue
                candidates.append(str(hid))
        except Exception:
            continue
    # if exactly one candidate, return it
    if len(candidates) == 1:
        return candidates[0]
    # else no reliable detection
    return None

# Collect ward placements for a list of (match_id,side) pairs for a team
def collect_wards_for_matches(match_side_pairs: List[Tuple[str,str]], wards_dir: Path, team_safe: str,
                              team_name: str, time_start_min: float, time_end_min: float,
                              ward_types: Optional[Set[str]] = None, verbose: bool=False) -> List[Dict[str,Any]]:
    pts = []
    ts = None if time_start_min is None else float(time_start_min)*60.0
    te = None if time_end_min is None else float(time_end_min)*60.0
    for mid, side in match_side_pairs:
        team_csv = wards_dir / "teams" / team_safe / f"match_{mid}_wards.csv"
        fallback = wards_dir / "all" / f"match_{mid}_wards.csv"
        rows = []
        if team_csv.exists():
            rows = read_csv_rows(team_csv)
            if verbose: print("Using team CSV:", team_csv)
        elif fallback.exists():
            rows = read_csv_rows(fallback)
            if verbose: print("Using fallback CSV:", fallback)
        else:
            if verbose: print("No ward CSV for match", mid)
            continue
        # filter rows by side/team/time/type
        seen = set()
        for r in rows:
            # side filter
            rside = str(r.get("radiant_or_dire") or r.get("team") or "").lower()
            if rside and side and rside != side:
                continue
            # team name filter
            tn = r.get("team_name") or r.get("team") or ""
            if tn and team_name and tn.strip().lower() != team_name.strip().lower():
                continue
            # time
            tval = None
            try:
                if r.get("time") not in (None,""):
                    tval = float(r.get("time"))
            except Exception:
                tval = None
            if tval is not None:
                if ts is not None and tval < ts: continue
                if te is not None and tval > te: continue
            # type
            typ = (str(r.get("ward_type") or r.get("type") or "")).lower()
            wtype = "obs" if "obs" in typ else ("sen" if "sen" in typ else "unknown")
            if ward_types and wtype not in ward_types:
                continue
            # coords
            x = None; y = None
            try:
                if r.get("x") not in (None,""):
                    x = float(r.get("x")); y = float(r.get("y"))
                else:
                    key = r.get("key") or ""
                    import re
                    m = re.search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", str(key))
                    if m:
                        x = float(m.group(1)); y = float(m.group(2))
            except Exception:
                continue
            if x is None or y is None:
                continue
            # dedupe by ehandle/key/match+rounded cell
            matchid = str(r.get("match_id") or r.get("match") or mid)
            ehandle = r.get("ehandle") or r.get("entityleft") or ""
            keyf = r.get("key") or ""
            if ehandle:
                uniq = ("ehandle", matchid, str(ehandle))
            elif keyf:
                uniq = ("key", matchid, str(keyf).strip())
            else:
                gx = int(round(x)); gy = int(round(y))
                uniq = ("cell", matchid, wtype, gx, gy)
            if uniq in seen:
                continue
            seen.add(uniq)
            pts.append({"match": matchid, "x": x, "y": y, "time_s": tval, "ward_type": wtype, "side": side})
    return pts

# Simple heatmap plot overlay (uses pillow + numpy + matplotlib)
def plot_heatmap_overlay(img_path: Path, points: List[Dict[str,Any]], out_png: Path, grid_size: int=256, flip_y: bool=True, bins: int=250, cmap: str="magma"):
    if np is None:
        print("NumPy/matplotlib missing; can't plot heatmap.")
        return
    if not points:
        print("No points to plot for", out_png)
        return
    img = Image.open(img_path).convert("RGBA")
    img_w, img_h = img.size
    xs=[]; ys=[]
    for p in points:
        # convert raw coords (OpenDota coords often 0..grid_size-1) to image px
        gx = float(p["x"]); gy = float(p["y"])
        max_idx = grid_size - 1
        gx = max(0.0, min(gx, float(max_idx)))
        gy = max(0.0, min(gy, float(max_idx)))
        px = (gx / max_idx) * (img_w - 1)
        py = (gy / max_idx) * (img_h - 1)
        if flip_y:
            py = img_h - py
        xs.append(px); ys.append(py)
    xs = np.array(xs); ys = np.array(ys)
    H, xedges, yedges = np.histogram2d(xs, ys, bins=bins)
    H = np.rot90(H); H = np.flipud(H)
    Hmask = np.ma.masked_where(H==0, H)
    fig, ax = plt.subplots(figsize=(img_w/100, img_h/100), dpi=100)
    ax.imshow(img, extent=[0,img_w,0,img_h])
    ax.imshow(Hmask, cmap=cmap, extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], alpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Wrote heatmap:", out_png)

# ---------- data structures ----------
@dataclass
class ComboStat:
    team: str
    combo_ids: Tuple[str,str,str]           # sorted tuple of three hero ids (strings)
    combo_names: Tuple[str,str,str]         # names via hero_map if available
    appearances: int = 0
    wins: int = 0
    match_ids: List[str] = None
    mid_detected_count: int = 0
    mid_detected_details: Dict[str,int] = None  # hero_id -> count when detected as mid inside this combo

    def __post_init__(self):
        if self.match_ids is None:
            self.match_ids = []
        if self.mid_detected_details is None:
            self.mid_detected_details = {}

# ---------- main pipeline ----------
def scan_and_aggregate(picks_dir: Path, matches_dir: Path, wards_dir: Path, hero_map_csv: Path,
                       time_start_min: float, time_end_min: float, ward_types: Optional[Set[str]],
                       generate_maps: bool, map_image: Optional[Path], top_maps: int, grid_size: int, flip_y: bool, verbose: bool):
    id_to_name, name_to_id = load_hero_map_csv(hero_map_csv) if hero_map_csv and hero_map_csv.exists() else ({}, {})
    results: Dict[Tuple[str,str,Tuple[str,str,str]], ComboStat] = {}  # key: (team_safe, team, combo_ids_sorted)
    # iterate teams
    teams_root = picks_dir / "teams"
    if not teams_root.exists():
        print("Picks teams folder not found:", teams_root)
        return
    for team_folder in sorted([p for p in teams_root.iterdir() if p.is_dir()]):
        team_safe = team_folder.name
        team_display = team_folder.name
        if verbose:
            print("Scanning team:", team_display)
        # for each picks file in that team folder
        for pj in sorted(team_folder.glob("match_*_picks.json")):
            try:
                picks = json.loads(pj.read_text(encoding="utf-8"))
            except Exception:
                continue
            mid_val = picks.get("match_id")
            if mid_val is None:
                continue
            midstr = str(mid_val)
            # find which side this team was on (radiant/dire) using matches_dir if available
            match_json_candidates = list(matches_dir.glob(f"*{midstr}*.json"))
            match_json = read_json(match_json_candidates[0]) if match_json_candidates else None
            side = None
            won = False
            if match_json:
                # determine side by comparing names
                rad_name = match_json.get("radiant_name") or ""
                dire_name = match_json.get("dire_name") or ""
                if not rad_name and isinstance(match_json.get("radiant_team"), dict):
                    rad_name = match_json.get("radiant_team").get("name") or rad_name
                if not dire_name and isinstance(match_json.get("dire_team"), dict):
                    dire_name = match_json.get("dire_team").get("name") or dire_name
                safe_rad = safe_name(str(rad_name))
                safe_dire = safe_name(str(dire_name))
                if team_safe == safe_rad:
                    side = "radiant"
                elif team_safe == safe_dire:
                    side = "dire"
                # winner
                if isinstance(match_json.get("radiant_win"), bool):
                    if side == "radiant":
                        won = bool(match_json.get("radiant_win"))
                    elif side == "dire":
                        won = not bool(match_json.get("radiant_win"))
                else:
                    # fallback: some JSON use "radiant_win" as string
                    try:
                        rv = match_json.get("radiant_win")
                        if rv is not None:
                            rvb = bool(rv)
                            if side == "radiant":
                                won = rvb
                            elif side == "dire":
                                won = not rvb
                    except Exception:
                        pass
            # assemble team picks ids
            rad_p = picks.get("radiant_picks") or []
            dire_p = picks.get("dire_picks") or []
            # pick the team's picks depending on side if known, else try to determine by presence of mid/detecting team name
            team_picks = []
            if side == "radiant":
                team_picks = rad_p
            elif side == "dire":
                team_picks = dire_p
            else:
                # unknown side: if this file is in team folder it's already for that team, try to see which side contains players from that team:
                team_picks = rad_p + dire_p  # fallback: consider all picks (we will filter combinations by presence)
            # normalize hero ids as strings and ensure uniqueness
            pick_ids = [str(x.get("hero_id") or x.get("hero") or "") for x in team_picks if x.get("hero_id") or x.get("hero")]
            pick_ids = [p for p in pick_ids if p]  # drop empties
            if len(pick_ids) < 3:
                continue
            # attempt to detect real mid hero in this match for this team
            detected_mid = None
            if match_json:
                detected_mid = detect_mid_from_match_json(match_json, team_safe)
            # build all unordered triples
            triples = itertools.combinations(sorted(set(pick_ids)), 3)
            for triple in triples:
                combo_ids_sorted = tuple(sorted(triple, key=lambda s: int(s) if s.isdigit() else s))
                key = (team_safe, team_display, combo_ids_sorted)
                if key not in results:
                    # build names tuple
                    names = tuple(id_to_name.get(hid, hid) for hid in combo_ids_sorted)
                    results[key] = ComboStat(team=team_display, combo_ids=combo_ids_sorted, combo_names=names)
                stat = results[key]
                stat.appearances += 1
                if won:
                    stat.wins += 1
                stat.match_ids.append(midstr)
                # mid detection: if detected_mid exists and is in triple, increment
                if detected_mid and str(detected_mid) in combo_ids_sorted:
                    stat.mid_detected_count += 1
                    stat.mid_detected_details[str(detected_mid)] = stat.mid_detected_details.get(str(detected_mid), 0) + 1

    # write summary CSV & details JSON
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "combo_summary.csv"
    details_json = out_dir / "combo_details.json"
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["team_safe","team","combo_ids","combo_names","appearances","wins","win_rate","mid_detected_count","sample_matches"])
        for key, stat in sorted(results.items(), key=lambda kv: (-kv[1].appearances, kv[0])):
            win_rate = (stat.wins / stat.appearances) if stat.appearances else 0.0
            sample = ",".join(stat.match_ids[:10])
            writer.writerow([key[0], stat.team, "|".join(stat.combo_ids), "|".join(stat.combo_names), stat.appearances, stat.wins, f"{win_rate:.3f}", stat.mid_detected_count, sample])
    with details_json.open("w", encoding="utf-8") as fh:
        # convert dataclasses to dicts
        outd = {}
        for key, stat in results.items():
            kstr = f"{key[0]}::{','.join(stat.combo_ids)}"
            outd[kstr] = asdict(stat)
        json.dump(outd, fh, indent=2, ensure_ascii=False)
    print("Wrote summary CSV:", summary_csv)
    print("Wrote details JSON:", details_json)

    # optionally generate maps for top combos per team
    if generate_maps:
        if np is None:
            print("NumPy/matplotlib not available; skipping map generation.")
            return
        maps_out_root = out_dir / "maps"
        # group results by team
        team_groups: Dict[str, List[ComboStat]] = defaultdict(list)
        for stat in results.values():
            team_groups[safe_name(stat.team)].append(stat)
        for team_safe, stats in team_groups.items():
            # pick top-N combos by appearances
            stats_sorted = sorted(stats, key=lambda s: -s.appearances)[:top_maps]
            team_maps_dir = maps_out_root / team_safe
            team_maps_dir.mkdir(parents=True, exist_ok=True)
            for stat in stats_sorted:
                # need to collect (match,side) pairs for matches where this combo occurred
                match_pairs = []
                # search picks files again to find side association for matches in this stat
                picks_team_dir = picks_dir / "teams" / team_safe
                for mid in stat.match_ids:
                    # read picks json file in team folder
                    pj = picks_team_dir / f"match_{mid}_picks.json"
                    mj_path = (matches_dir.glob(f"*{mid}*.json"))
                    match_json = read_json(next(iter(list(mj_path)), None)) if mj_path else None
                    side = None
                    if match_json:
                        rad_name = match_json.get("radiant_name") or ""
                        dire_name = match_json.get("dire_name") or ""
                        if not rad_name and isinstance(match_json.get("radiant_team"), dict):
                            rad_name = match_json.get("radiant_team").get("name") or rad_name
                        if not dire_name and isinstance(match_json.get("dire_team"), dict):
                            dire_name = match_json.get("dire_team").get("name") or dire_name
                        if safe_name(rad_name) == team_safe:
                            side = "radiant"
                        elif safe_name(dire_name) == team_safe:
                            side = "dire"
                    if not side:
                        # fallback: assume team CSV file presence indicates side? We will not assume
                        side = "unknown"
                    match_pairs.append((mid, side))
                # collect wards for these matches
                points = collect_wards_for_matches(match_pairs, wards_dir, team_safe, stat.team,
                                                   time_start_min, time_end_min, ward_types, verbose)
                if not points:
                    print(f"No wards for combo {stat.combo_ids} of team {team_safe} -> skipping map")
                    continue
                # plot
                combo_label = "_".join(stat.combo_ids)
                png_out = team_maps_dir / f"combo_{combo_label}_t{int(time_start_min)}_{int(time_end_min)}.png"
                plot_heatmap_overlay(map_image, points, png_out, grid_size=grid_size, flip_y=flip_y, bins=300, cmap="magma")
    return

# ---------- CLI ----------
def parse_args_cli():
    p = argparse.ArgumentParser(description="Aggregate combos (mid+pos4+pos5 candidate triples) and optionally plot ward heatmaps for top combos.")
    p.add_argument("--picks-dir", type=Path, default=Path("data/picks"), help="Picks root (teams/<Team>/match_*_picks.json)")
    p.add_argument("--wards-dir", type=Path, default=Path("data/obs_logs"), help="Wards root (teams/<Team>/match_*_wards.csv and all/)")
    p.add_argument("--matches-dir", type=Path, default=Path("data/matches"), help="Matches JSON dir (for side/win inference)")
    p.add_argument("--hero-map", type=Path, default=Path("data/hero_map.csv"), help="hero map CSV id,localized_name,hero_name")
    p.add_argument("--start", type=float, default=0.0, help="Start time (minutes) inclusive for ward collection")
    p.add_argument("--end", type=float, default=10.0, help="End time (minutes) inclusive for ward collection")
    p.add_argument("--out", type=Path, default=Path("results/combo_stats"), help="Output folder for CSV/JSON and maps")
    p.add_argument("--generate-maps", action="store_true", help="Generate heatmaps for top combos per team")
    p.add_argument("--top-maps", type=int, default=8, help="Top combos per team to generate maps for (by appearances)")
    p.add_argument("--map-image", type=Path, help="Minimap image to overlay heatmaps (required if --generate-maps)")
    p.add_argument("--grid-size", type=int, default=256, help="Coordinate grid size")
    p.add_argument("--flip-y", action="store_true", help="Flip Y when mapping coords to image")
    p.add_argument("--ward-types", type=str, default="obs", help="Comma-separated ward types to include (obs,sen)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args_cli()
    if args.generate_maps and not args.map_image:
        print("Error: --map-image is required when --generate-maps is set.")
        sys.exit(2)
    ward_types = set([w.strip().lower() for w in args.ward_types.split(",") if w.strip()]) if args.ward_types else None
    # call pipeline
    scan_and_aggregate(picks_dir=args.picks_dir,
                       matches_dir=args.matches_dir,
                       wards_dir=args.wards_dir,
                       hero_map_csv=args.hero_map,
                       time_start_min=args.start,
                       time_end_min=args.end,
                       ward_types=ward_types,
                       generate_maps=bool(args.generate_maps),
                       map_image=args.map_image,
                       top_maps=int(args.top_maps),
                       grid_size=int(args.grid_size),
                       flip_y=bool(args.flip_y),
                       verbose=bool(args.verbose))