#!/usr/bin/env python3
"""
plot_wards_on_map_team_aggregate.py

Aggregate and plot wards for a team (by team name or id) or by side (radiant/dire).
- Supports single-match (--csv) and folder (--in) modes.
- When given --team-name and --aggregate, produces per-side images:
    * wards_agg_<team>_radiant_window1_0-10.png ...
    * wards_agg_<team>_dire_window1_0-10.png ...
  i.e. separate images for matches where the team was on Radiant vs Dire.
- Keeps original mapping behavior (subtract 64 in collect_points_for_team).
- Default time windows: 0-10, 10-25, 25-40, 40+ minutes. Use --windows to override.
- New: by default also output per-player ward maps (organized by account_id).
  You get both the user-specified windows AND the default windows output.

Usage examples:
  python3 plot_wards_on_map_team_aggregate.py --in ./obs_logs --map images/map_1024.jpg \
      --team-name "Xtreme Gaming" --aggregate --out ./insights --verbose

  # include negative-start window -1..2 (quote it)
  python3 plot_wards_on_map_team_aggregate.py --in ./data/obs_logs/team_falcons --map images/map_738.jpg \
      --team radiant --aggregate --ward-types "obs" --out ./obs_facts/team_falcons --grid-size 128 \
      --team-name "Team Falcons" --windows "-1-2"

Dependencies:
  pip install matplotlib pillow numpy seaborn pandas
"""
from pathlib import Path
import argparse
import csv
import re
import math
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

try:
    import pandas as pd
except Exception:
    pd = None

# Color/marker mapping
TEAM_COLORS = {"radiant": "#00FF66", "dire": "#FF3333", "unknown": "#999999"}
WARD_MARKERS = {"obs": "o", "sen": "s", "unknown": "x"}

# Default windows expressed in minutes
DEFAULT_WINDOWS_MIN = [(0, 10), (10, 25), (25, 40), (40, None)]  # last None => 40+

def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(s).strip()).strip("_")[:200]

def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if pd:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)

def parse_xy_from_row(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    # prefer numeric x and y columns
    for kx in ("x", "pos_x", "posx", "position_x"):
        if kx in row and row[kx] not in (None, "", "nan"):
            try:
                x = float(row[kx])
            except Exception:
                continue
            for ky in ("y", "pos_y", "posy", "position_y"):
                if ky in row and row[ky] not in (None, "", "nan"):
                    try:
                        y = float(row[ky])
                        return (x, y)
                    except Exception:
                        continue
    key = row.get("key") or row.get("pos_key") or ""
    if isinstance(key, str) and key:
        m = re.search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", key)
        if m:
            try:
                return (float(m.group(1)), float(m.group(2)))
            except Exception:
                pass
        m2 = re.search(r"\[(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\]", key)
        if m2:
            try:
                return (float(m2.group(1)), float(m2.group(2)))
            except Exception:
                pass
    return None

def infer_grid_size_from_rows(rows: List[Dict[str, Any]], snap_to=(128,256,512,1024)) -> int:
    max_coord = 0.0
    for r in rows:
        try:
            xval = r.get("x") or r.get("pos_x") or r.get("posx") or r.get("position_x")
            yval = r.get("y") or r.get("pos_y") or r.get("posy") or r.get("position_y")
            if xval is None or yval is None:
                xy = parse_xy_from_row(r)
                if xy:
                    gx, gy = xy
                else:
                    continue
            else:
                gx = float(xval); gy = float(yval)
            max_coord = max(max_coord, abs(gx), abs(gy))
        except Exception:
            continue
    if max_coord <= 0:
        return 256
    candidate = int(math.ceil(max_coord)) + 1
    for s in snap_to:
        if abs(candidate - s) <= max(1, int(0.02 * s)):
            return s
    return max(128, candidate)

def parse_windows_arg(windows_arg: Optional[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Parse windows specification (supports negative numbers and open-ended ranges).

    Accepts strings like:
      - "" or None -> default windows
      - "0-10,10-25,25-40,40-"
      - "-1-2"  (from -1 minute to 2 minutes)
      - "40-"  (40 minutes and beyond)
    Returns list of (start_min, end_min_or_None)
    """
    if not windows_arg:
        return DEFAULT_WINDOWS_MIN
    toks = [t.strip() for t in windows_arg.split(",") if t.strip()]
    out: List[Tuple[Optional[int], Optional[int]]] = []
    for tok in toks:
        # open-ended like "40-"
        m_open = re.match(r'^\s*(-?\d+)\s*-\s*$', tok)
        if m_open:
            a = int(m_open.group(1))
            out.append((a, None))
            continue
        # full range like "-1-2" or "0-10" or "-5--1" (negative end)
        m = re.match(r'^\s*(-?\d+)\s*-\s*(-?\d+)\s*$', tok)
        if m:
            a = int(m.group(1)); b = int(m.group(2))
            if b <= a:
                raise ValueError(f"Window end must be > start: {tok}")
            out.append((a, b))
            continue
        raise ValueError(f"Invalid window token: {tok}")
    return out

def collect_points_for_team(rows: List[Dict[str, Any]],
                            team_choice: Optional[str],
                            team_name_or_id: Optional[str],
                            time_start: Optional[float],
                            time_end: Optional[float],
                            player_slot: Optional[int],
                            only_types: Optional[List[str]] = None,
                            verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Enhanced collector with placement filtering and deduplication.
    - only_types: e.g. ['obs'] to only include obs wards (recommended).
    - It counts only rows whose 'type' field looks like a placement (contains '_log').
    - Dedup by ehandle (preferred), then by 'key' (if present), then by integer grid cell (per match).
    - When verbose=True it prints diagnostics for rows read / placement-like rows / unique placements.
    """
    pts: List[Dict[str, Any]] = []
    seen = set()  # unique identifiers to deduplicate placements
    raw_rows = 0
    placement_rows = 0
    per_match_counts = {}

    for r in rows:
        raw_rows += 1

        # team name/id match filter if provided
        if team_name_or_id is not None:
            matched = False
            try:
                tid = int(team_name_or_id)
                if 'team_id' in r and r.get('team_id') not in (None, "") and str(r.get('team_id')) == str(tid):
                    matched = True
            except Exception:
                pass
            if not matched:
                if 'team_name' in r and r.get('team_name') not in (None, ""):
                    if str(r.get('team_name')).strip().lower() == str(team_name_or_id).strip().lower():
                        matched = True
            if not matched:
                continue

        team_field = (str(r.get("radiant_or_dire") or r.get("team") or "")).lower()
        if team_choice and team_field != team_choice:
            continue

        # time filter (seconds)
        if (time_start is not None) or (time_end is not None):
            tval = None
            if "time" in r:
                try:
                    tval = float(r["time"])
                except Exception:
                    tval = None
            if tval is not None:
                if time_start is not None and tval < time_start:
                    continue
                if time_end is not None and tval > time_end:
                    continue

        # player slot filter
        if player_slot is not None:
            slot_val = r.get("player_slot")
            try:
                if int(slot_val) != player_slot:
                    continue
            except Exception:
                continue

        # ensure placement-like
        row_type = str(r.get("type") or r.get("event") or "").lower()
        if "_log" not in row_type:
            continue
        placement_rows += 1

        ward_type_raw = (str(r.get("ward_type") or r.get("type") or "")).lower()
        if ward_type_raw.startswith("obs"):
            wtype = "obs"
        elif ward_type_raw.startswith("sen"):
            wtype = "sen"
        else:
            wtype = "unknown"

        if only_types and wtype not in only_types:
            continue

        xy = parse_xy_from_row(r)
        if not xy:
            continue
        x, y = xy

        ehandle = r.get("ehandle") or r.get("entityleft") or ""
        key_field = r.get("key") or r.get("pos_key") or ""
        match_id = str(r.get("match_id") or r.get("match") or "")

        if ehandle not in (None, ""):
            uniq = ("ehandle", match_id, str(ehandle))
        elif key_field not in (None, ""):
            uniq = ("key", match_id, str(key_field).strip())
        else:
            try:
                gx = int(round(float(x)))
                gy = int(round(float(y)))
            except Exception:
                gx = int(math.floor(float(x))) if x is not None else 0
                gy = int(math.floor(float(y))) if y is not None else 0
            uniq = ("cell", match_id, wtype, gx, gy)

        if uniq in seen:
            per_match_counts.setdefault(match_id, {"raw": 0, "placement": 0, "unique": 0})
            continue
        seen.add(uniq)

        x -= 64
        y -= 64

        pts.append({
            "x": x,
            "y": y,
            "ward_type": wtype,
            "team": team_field,
            "time": float(r["time"]) if "time" in r and r["time"] not in (None, "") else None,
            "match": match_id,
            "account_id": r.get("account_id") or "",
            "player_slot": r.get("player_slot") or "",
            "uniq_id": uniq
        })

        per_match_counts.setdefault(match_id, {"raw": 0, "placement": 0, "unique": 0})
        per_match_counts[match_id]["placement"] += 1
        per_match_counts[match_id]["unique"] += 1

    if verbose:
        print(f"[diag] rows read={raw_rows}, placement-like rows={placement_rows}, unique placements after dedup={len(pts)}")
        if per_match_counts:
            print("[diag] per-match placement summary (placement_count, unique_count) for matches seen:")
            placement_per_match = {}
            for r in rows:
                m = str(r.get("match_id") or r.get("match") or "")
                row_type = str(r.get("type") or r.get("event") or "").lower()
                if "_log" not in row_type:
                    continue
                placement_per_match[m] = placement_per_match.get(m, 0) + 1
            for mid, counts in per_match_counts.items():
                placements = placement_per_match.get(mid, 0)
                print(f"  match {mid or '<unknown>'}: placement_rows={placements}, unique_after_dedup={counts.get('unique',0)}")
    return pts

def map_to_image_coords(x: float, y: float, grid_size: int, img_w: int, img_h: int, flip_y: bool) -> Tuple[float, float]:
    if grid_size <= 1:
        return (img_w / 2.0, img_h / 2.0)
    max_index = grid_size - 1
    gx = float(x); gy = float(y)
    gx = max(0.0, min(gx, float(max_index)))
    gy = max(0.0, min(gy, float(max_index)))
    px = (gx / max_index) * (img_w - 1)
    py = (gy / max_index) * (img_h - 1)
    if flip_y:
        py = img_h - py
    return px, py

def plot_aggregate_map(img_path: Path, points_by_window: List[List[Dict[str, Any]]],
                       out_path: Path, grid_size: int, flip_y: bool,
                       point_size: int = 30, alpha: float = 0.6, figsize=(8,8), title: Optional[str]=None,
                       draw_circle: bool = True):
    """
    points_by_window: list where each element is a list of point dicts for that window.
    Draws all windows on the same map with distinct colors and a legend.
    Overlays a red circle (outline) on each ward for easy inspection.
    """
    img = Image.open(img_path).convert("RGBA")
    img_w, img_h = img.size
    dpi = 128
    plt.figure(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
    ax = plt.gca()
    ax.imshow(img, extent=[0, img_w, 0, img_h])

    cmap = plt.get_cmap("tab10")
    nwin = len(points_by_window)
    colors = [cmap(i % 10) for i in range(nwin)]

    for i, pts in enumerate(points_by_window):
        if not pts:
            continue
        xs = []
        ys = []
        for p in pts:
            px, py = map_to_image_coords(p["x"], p["y"], grid_size, img_w, img_h, flip_y)
            xs.append(px); ys.append(py)
        xs = np.array(xs); ys = np.array(ys)

        ax.scatter(xs, ys, c=[colors[i]], s=point_size, marker="o",
                   edgecolors="k", linewidths=0.3, alpha=alpha, label=f"window {i+1} ({len(xs)})")

        from matplotlib.patches import Circle
        if draw_circle and xs.size > 0:
            outline_px = 10
            outline_lw = 1.8
            outline_alpha = 0.95
            for xi, yi in zip(xs, ys):
                circ = Circle((xi, yi), radius=outline_px,
                            facecolor='none', edgecolor='red',
                            linewidth=outline_lw, alpha=outline_alpha, zorder=5)
                ax.add_patch(circ)

    ax.set_xlim(0, img_w); ax.set_ylim(0, img_h)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    ax.legend(loc="lower left", fontsize="small", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def _write_summary_txt(path: Path, points_by_window: List[List[Dict[str,Any]]], windows_min: List[Tuple[Optional[int], Optional[int]]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"Aggregated points per window:\n")
        for idx, lst in enumerate(points_by_window):
            a_min, b_min = windows_min[idx]
            label = f"{a_min}-{b_min}" if b_min is not None else f"{a_min}+"
            fh.write(f" window {idx+1} ({label} min): {len(lst)} points\n")

def aggregate_folder_and_plot_by_teamname(in_dir: Path, map_path: Path, out_dir: Path, team_name_or_id: str,
                                          grid_size: int, flip_y: bool, windows_min: List[Tuple[Optional[int], Optional[int]]],
                                          ward_types: List[str], player_slot: Optional[int], verbose: bool):
    """
    For given team_name_or_id, produce per-side images and per-player images, and print per-file diagnostics.
    """
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.name.startswith("match_") and p.name.endswith("_wards.csv")])
    if verbose:
        print(f"[diag] found {len(files)} files in {in_dir}")

    team_safe = safe_name(team_name_or_id)
    # For each side produce windows lists
    for side in ("radiant", "dire"):
        # team-level aggregation
        points_by_window: List[List[Dict[str,Any]]] = [[] for _ in windows_min]
        # per-player aggregation (account_id -> list of points)
        player_points: Dict[str, List[Dict[str,Any]]] = {}
        file_count = 0
        total_points_before = 0
        total_unique_points = 0

        for f in files:
            rows = read_csv_rows(f)
            raw_rows = len(rows)
            placement_like = 0
            for rr in rows:
                if "_log" in (str(rr.get("type") or rr.get("event") or "")).lower():
                    placement_like += 1

            pts = collect_points_for_team(rows, team_choice=side, team_name_or_id=team_name_or_id,
                                          time_start=None, time_end=None, player_slot=player_slot,
                                          only_types=ward_types if ward_types else None, verbose=False)
            if not pts:
                if verbose:
                    print(f"[diag] file {f.name}: placement_like_rows={placement_like}, unique_pts_for_team_as_{side}=0 (skipping)")
                continue

            # normalize times if needed (detect minutes vs seconds) - keep as-is if seconds
            times = [float(p['time']) for p in pts if p.get('time') is not None]
            if times:
                mx = max(times); med = float(np.median(times))
                if mx <= 200 and med >= 1.0:
                    if verbose:
                        print(f"[time-norm] converting ward times in {f.name} from minutes->seconds (max={mx}, med={med})")
                    for p in pts:
                        if p.get('time') is not None:
                            p['time'] = float(p['time']) * 60.0

            file_count += 1
            total_points_before += placement_like
            total_unique_points += len(pts)

            # append to team-level windows and also to per-player buckets
            for p in pts:
                # collect per-player
                aid = str(p.get("account_id") or p.get("player_slot") or "unknown")
                player_points.setdefault(aid, []).append(p)

                # assign to windows (minutes)
                t = p.get("time")
                assigned = False
                for idx, (a_min, b_min) in enumerate(windows_min):
                    if t is None:
                        if idx == 0:
                            points_by_window[idx].append(p); assigned = True; break
                        else:
                            continue
                    t_min = float(t) / 60.0
                    if a_min is not None and t_min < a_min:
                        continue
                    if b_min is None:
                        if t_min >= a_min:
                            points_by_window[idx].append(p); assigned = True; break
                    else:
                        if t_min >= a_min and t_min < b_min:
                            points_by_window[idx].append(p); assigned = True; break
                if not assigned:
                    if windows_min and windows_min[-1][1] is None:
                        points_by_window[-1].append(p)
                    else:
                        # skip out-of-windows points
                        continue

            if verbose:
                print(f"[diag] file {f.name}: placement_like_rows={placement_like}, unique_pts_for_team_as_{side}={len(pts)}")

        total_points = sum(len(l) for l in points_by_window)
        if total_points == 0:
            if verbose:
                print(f"[diag] team {team_name_or_id} has no points as {side} across checked files (skipping).")
            continue

        # write team-level outputs
        base_out = out_dir / f"wards_agg_{team_safe}_{side}"
        side_out = base_out
        side_out.mkdir(parents=True, exist_ok=True)

        summary_path = base_out.with_suffix(".summary.txt")
        _write_summary_txt(summary_path, points_by_window, windows_min)
        if verbose:
            print(f"[diag] wrote summary {summary_path}")

        # per-window images for team (user-specified windows)
        for idx, lst in enumerate(points_by_window):
            a_min, b_min = windows_min[idx]
            label = f"{a_min}-{b_min}" if b_min is not None else f"{a_min}plus"
            win_out = base_out.with_name(base_out.name + f"_window{idx+1}_{label}.png")
            if verbose:
                print(f"[diag] plotting team {team_name_or_id} as {side} window {idx+1} ({label}) with {len(lst)} points -> {win_out}")
            plot_aggregate_map(map_path, [lst], win_out, grid_size=grid_size, flip_y=flip_y, point_size=28, alpha=0.8, title=f"{team_name_or_id} as {side} ({label} min)")

        # combined map for side (by-window overlay)
        combined_out = base_out.with_name(base_out.name + f"_combined.png")
        if verbose:
            print(f"[diag] plotting combined map for {team_name_or_id} as {side} -> {combined_out}")
        plot_aggregate_map(map_path, points_by_window, combined_out, grid_size=grid_size, flip_y=flip_y, point_size=28, alpha=0.7, title=f"{team_name_or_id} as {side} (combined)")

        # ALSO output default-window maps (if different from user windows)
        if windows_min != DEFAULT_WINDOWS_MIN:
            if verbose:
                print(f"[diag] also plotting default windows for team {team_name_or_id} as {side}")
            default_points_by_window: List[List[Dict[str,Any]]] = [[] for _ in DEFAULT_WINDOWS_MIN]
            for f in files:
                rows = read_csv_rows(f)
                pts = collect_points_for_team(rows, team_choice=side, team_name_or_id=team_name_or_id,
                                              time_start=None, time_end=None, player_slot=player_slot,
                                              only_types=ward_types if ward_types else None, verbose=False)
                # normalize times same as before
                times = [float(p['time']) for p in pts if p.get('time') is not None]
                if times:
                    mx = max(times); med = float(np.median(times))
                    if mx <= 200 and med >= 1.0:
                        for p in pts:
                            if p.get('time') is not None:
                                p['time'] = float(p['time']) * 60.0
                for p in pts:
                    t = p.get("time")
                    if t is None:
                        default_points_by_window[0].append(p)
                        continue
                    t_min = float(t) / 60.0
                    for idx, (a_min, b_min) in enumerate(DEFAULT_WINDOWS_MIN):
                        if a_min is not None and t_min < a_min:
                            continue
                        if b_min is None:
                            if t_min >= a_min:
                                default_points_by_window[idx].append(p); break
                        else:
                            if t_min >= a_min and t_min < b_min:
                                default_points_by_window[idx].append(p); break
            # write default-window images
            base_out_def = out_dir / f"wards_agg_{team_safe}_{side}_default_windows"
            for idx, lst in enumerate(default_points_by_window):
                a_min, b_min = DEFAULT_WINDOWS_MIN[idx]
                label = f"{a_min}-{b_min}" if b_min is not None else f"{a_min}plus"
                win_out = base_out_def.with_name(base_out_def.name + f"_window{idx+1}_{label}.png")
                if verbose:
                    print(f"[diag] plotting default-window team {team_name_or_id} as {side} window {idx+1} ({label}) with {len(lst)} points -> {win_out}")
                plot_aggregate_map(map_path, [lst], win_out, grid_size=grid_size, flip_y=flip_y, point_size=28, alpha=0.8, title=f"{team_name_or_id} as {side} ({label} min)")

        # PER-PLAYER output: one folder per account_id (use safe_name on account id)
        players_out_base = side_out / "players"
        for aid, plist in player_points.items():
            aid_safe = safe_name(str(aid)) or "unknown"
            player_dir = players_out_base / aid_safe
            player_dir.mkdir(parents=True, exist_ok=True)
            # bucket per windows_min
            p_points_by_window: List[List[Dict[str,Any]]] = [[] for _ in windows_min]
            for p in plist:
                t = p.get("time")
                assigned = False
                for idx, (a_min, b_min) in enumerate(windows_min):
                    if t is None:
                        if idx == 0:
                            p_points_by_window[idx].append(p); assigned = True; break
                        else:
                            continue
                    t_min = float(t) / 60.0
                    if a_min is not None and t_min < a_min:
                        continue
                    if b_min is None:
                        if t_min >= a_min:
                            p_points_by_window[idx].append(p); assigned = True; break
                    else:
                        if t_min >= a_min and t_min < b_min:
                            p_points_by_window[idx].append(p); assigned = True; break
                if not assigned:
                    if windows_min and windows_min[-1][1] is None:
                        p_points_by_window[-1].append(p)
                    else:
                        continue
            # write per-window images for this player
            for idx, lst in enumerate(p_points_by_window):
                a_min, b_min = windows_min[idx]
                label = f"{a_min}-{b_min}" if b_min is not None else f"{a_min}plus"
                out_png = player_dir / f"player_{aid_safe}_window{idx+1}_{label}.png"
                if verbose:
                    print(f"[diag] plotting player {aid_safe} as {side} window {idx+1} ({label}) with {len(lst)} points -> {out_png}")
                plot_aggregate_map(map_path, [lst], out_png, grid_size=grid_size, flip_y=flip_y, point_size=20, alpha=0.9, title=f"player {aid_safe} as {side} ({label})")

            # combined_by_window (existing behaviour) + combined_all (new: all points flattened)
            combined_player_by_window = player_dir / f"player_{aid_safe}_combined_by_window.png"
            if verbose:
                print(f"[diag] plotting player {aid_safe} combined (by-window) -> {combined_player_by_window}")
            plot_aggregate_map(map_path, p_points_by_window, combined_player_by_window,
                               grid_size=grid_size, flip_y=flip_y, point_size=20, alpha=0.9,
                               title=f"player {aid_safe} as {side} (combined by window)")

            # NEW: flatten all points into a single layer for "all times" combined view
            all_points = [pt for win in p_points_by_window for pt in win]
            combined_player_all = player_dir / f"player_{aid_safe}_combined_all.png"
            if verbose:
                print(f"[diag] plotting player {aid_safe} combined (all) -> {combined_player_all} (total {len(all_points)} pts)")
            plot_aggregate_map(map_path, [all_points], combined_player_all,
                               grid_size=grid_size, flip_y=flip_y, point_size=20, alpha=0.9,
                               title=f"player {aid_safe} as {side} (all times)")

        if verbose:
            print(f"[write] per-window maps, default-window maps and per-player maps saved to {base_out.parent}")

def process_single_csv_for_teamname(csv_path: Path, map_path: Path, out_dir: Path, team_name_or_id: str,
                                    grid_size: int, flip_y: bool, windows_min: List[Tuple[Optional[int], Optional[int]]],
                                    ward_types: List[str], player_slot: Optional[int], verbose: bool):
    """Produce per-side images for a single CSV for the given team name/id."""
    rows = read_csv_rows(csv_path)
    if grid_size == 0:
        inferred = infer_grid_size_from_rows(rows)
        grid_size_to_use = inferred
        if verbose:
            print(f"[diagnostic] inferred grid_size={inferred} from CSV")
    else:
        grid_size_to_use = grid_size

    for side in ("radiant","dire"):
        pts = collect_points_for_team(rows, team_choice=side, team_name_or_id=team_name_or_id, time_start=None, time_end=None, player_slot=player_slot, only_types=ward_types if ward_types else None)
        if not pts:
            if verbose:
                print(f"[diag] no points for team {team_name_or_id} as {side} in {csv_path.name}")
            continue

        # normalize times if needed
        times = [float(p['time']) for p in pts if p.get('time') is not None]
        if times:
            mx = max(times); med = float(np.median(times))
            if mx <= 200 and med >= 1.0:
                if verbose:
                    print(f"[time-norm] converting ward times in {csv_path.name} from minutes->seconds (max={mx}, med={med})")
                for p in pts:
                    if p.get('time') is not None:
                        p['time'] = float(p['time']) * 60.0

        points_by_window: List[List[Dict[str,Any]]] = [[] for _ in windows_min]
        player_points: Dict[str, List[Dict[str,Any]]] = {}

        for p in pts:
            aid = str(p.get("account_id") or p.get("player_slot") or "unknown")
            player_points.setdefault(aid, []).append(p)
            t = p.get("time")
            assigned = False
            for idx, (a_min, b_min) in enumerate(windows_min):
                if t is None:
                    if idx == 0:
                        points_by_window[idx].append(p); assigned = True; break
                    else:
                        continue
                t_min = float(t) / 60.0
                if a_min is not None and t_min < a_min:
                    continue
                if b_min is None:
                    if t_min >= a_min:
                        points_by_window[idx].append(p); assigned = True; break
                else:
                    if t_min >= a_min and t_min < b_min:
                        points_by_window[idx].append(p); assigned = True; break
            if not assigned:
                if windows_min and windows_min[-1][1] is None:
                    points_by_window[-1].append(p)
                else:
                    continue

        team_safe = safe_name(team_name_or_id)
        base_out = out_dir / f"{csv_path.stem}_{team_safe}_{side}"
        base_out.mkdir(parents=True, exist_ok=True)

        # per-window and combined for team
        for idx, lst in enumerate(points_by_window):
            a_min, b_min = windows_min[idx]
            label = f"{a_min}-{b_min}" if b_min is not None else f"{a_min}plus"
            win_out = base_out.with_name(base_out.name + f"_window{idx+1}_{label}.png")
            plot_aggregate_map(map_path, [lst], win_out, grid_size=grid_size_to_use, flip_y=flip_y, point_size=40, alpha=0.85, title=f"{csv_path.stem} {team_name_or_id} as {side} ({label})")
        combined_out = base_out.with_name(base_out.name + "_combined.png")
        plot_aggregate_map(map_path, points_by_window, combined_out, grid_size=grid_size_to_use, flip_y=flip_y, point_size=40, alpha=0.8, title=f"{csv_path.stem} {team_name_or_id} as {side} (combined)")

        # default windows too if different
        if windows_min != DEFAULT_WINDOWS_MIN:
            default_points_by_window: List[List[Dict[str,Any]]] = [[] for _ in DEFAULT_WINDOWS_MIN]
            for p in pts:
                t = p.get("time")
                if t is None:
                    default_points_by_window[0].append(p); continue
                t_min = float(t) / 60.0
                for idx, (a_min, b_min) in enumerate(DEFAULT_WINDOWS_MIN):
                    if a_min is not None and t_min < a_min:
                        continue
                    if b_min is None:
                        if t_min >= a_min:
                            default_points_by_window[idx].append(p); break
                    else:
                        if t_min >= a_min and t_min < b_min:
                            default_points_by_window[idx].append(p); break
            base_out_def = base_out.with_name(base_out.name + "_default_windows")
            for idx, lst in enumerate(default_points_by_window):
                a_min, b_min = DEFAULT_WINDOWS_MIN[idx]
                label = f"{a_min}-{b_min}" if b_min is not None else f"{a_min}plus"
                win_out = base_out_def.with_name(base_out_def.name + f"_window{idx+1}_{label}.png")
                plot_aggregate_map(map_path, [lst], win_out, grid_size=grid_size_to_use, flip_y=flip_y, point_size=40, alpha=0.85, title=f"{csv_path.stem} {team_name_or_id} as {side} ({label})")

        # per-player outputs
        players_out_base = base_out / "players"
        for aid, plist in player_points.items():
            aid_safe = safe_name(str(aid)) or "unknown"
            player_dir = players_out_base / aid_safe
            player_dir.mkdir(parents=True, exist_ok=True)
            p_points_by_window: List[List[Dict[str,Any]]] = [[] for _ in windows_min]
            for p in plist:
                t = p.get("time")
                assigned = False
                for idx, (a_min, b_min) in enumerate(windows_min):
                    if t is None:
                        if idx == 0:
                            p_points_by_window[idx].append(p); assigned = True; break
                        else:
                            continue
                    t_min = float(t) / 60.0
                    if a_min is not None and t_min < a_min:
                        continue
                    if b_min is None:
                        if t_min >= a_min:
                            p_points_by_window[idx].append(p); assigned = True; break
                    else:
                        if t_min >= a_min and t_min < b_min:
                            p_points_by_window[idx].append(p); assigned = True; break
                if not assigned:
                    if windows_min and windows_min[-1][1] is None:
                        p_points_by_window[-1].append(p)
                    else:
                        continue
            for idx, lst in enumerate(p_points_by_window):
                a_min, b_min = windows_min[idx]
                label = f"{a_min}-{b_min}" if b_min is not None else f"{a_min}plus"
                out_png = player_dir / f"player_{aid_safe}_window{idx+1}_{label}.png"
                plot_aggregate_map(map_path, [lst], out_png, grid_size=grid_size_to_use, flip_y=flip_y, point_size=20, alpha=0.9, title=f"player {aid_safe} as {side} ({label})")

            # combined_by_window (existing behaviour)
            combined_player_by_window = player_dir / f"player_{aid_safe}_combined_by_window.png"
            plot_aggregate_map(map_path, p_points_by_window, combined_player_by_window, grid_size=grid_size_to_use, flip_y=flip_y, point_size=20, alpha=0.9, title=f"player {aid_safe} as {side} (combined by window)")

            # NEW: combined_all -> flatten all points into a single layer
            all_points = [pt for win in p_points_by_window for pt in win]
            combined_player_all = player_dir / f"player_{aid_safe}_combined_all.png"
            plot_aggregate_map(map_path, [all_points], combined_player_all, grid_size=grid_size_to_use, flip_y=flip_y, point_size=20, alpha=0.9, title=f"player {aid_safe} as {side} (all times)")

def main():
    p = argparse.ArgumentParser(description="Plot team's ward placements — single match or aggregated by time windows across many matches.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=Path, help="Single match CSV to plot")
    group.add_argument("--in", dest="in_dir", type=Path, help="Folder containing match_*_wards.csv files to aggregate")
    p.add_argument("--map", dest="map_image", type=Path, required=True, help="Path to minimap image PNG/JPG")
    p.add_argument("--team", type=str, choices=["radiant", "dire"], help="(optional) restrict to side when used (radiant/dire).")
    p.add_argument("--team-name", type=str, default=None, help="Team name or team_id to filter (e.g., 'Xtreme Gaming'). If provided with --in and --aggregate, produces per-side windowed maps for that team.")
    p.add_argument("--player-slot", type=int, default=None, help="Only plot wards placed by this player slot (optional)")
    p.add_argument("--out", dest="out_dir", type=Path, default=Path("./ward_maps"), help="Output folder for generated map images")
    p.add_argument("--grid-size", type=int, default=256, help="Grid size used by OpenDota coords (default 256). Pass 0 to auto-detect per file.")
    p.add_argument("--flip-y", action="store_true", help="Flip Y axis when mapping coords to image")
    p.add_argument("--start", type=float, default=None, help="Start time (seconds) inclusive to filter events for single match")
    p.add_argument("--end", type=float, default=None, help="End time (seconds) inclusive to filter events for single match")
    p.add_argument("--aggregate", action="store_true", help="When using --in, aggregate across matches into one combined map (default no)")
    p.add_argument("--windows", type=str, default="", help="Comma-separated windows in minutes, e.g. '0-10,10-25,25-40,40-' (empty uses defaults). Quote if starting with '-'")
    p.add_argument("--heatmap", action="store_true", help="(single-match) Overlay a kernel density instead of plain scatter (kept for compatibility)")
    p.add_argument("--point-size", type=int, default=60, help="Base point size for obs wards (auto-scaled for large images)")
    p.add_argument("--ward-types", type=str, default="obs", help="Comma-separated ward types to include (default obs). Use '' for all")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    windows_min = parse_windows_arg(args.windows)
    ward_types = [w.strip() for w in args.ward_types.split(",") if w.strip()] if args.ward_types else []

    # Single CSV mode
    if args.csv:
        if args.team_name:
            process_single_csv_for_teamname(args.csv, args.map_image, out_dir, args.team_name, args.grid_size, args.flip_y, windows_min, ward_types, args.player_slot, args.verbose)
            return
        else:
            rows = read_csv_rows(args.csv)
            grid_size_to_use = args.grid_size if args.grid_size != 0 else infer_grid_size_from_rows(rows)
            points = collect_points_for_team(rows, team_choice=args.team, team_name_or_id=None, time_start=args.start, time_end=args.end, player_slot=args.player_slot)
            if not points:
                print("No ward points found for given filters.")
                return
            plot_points_on_map_image = globals().get("plot_points_on_map_image")
            out_path = out_dir / (args.csv.stem + f"_{args.team or 'all'}_map.png")
            if plot_points_on_map_image is None:
                plot_aggregate_map(args.map_image, [points], out_path, grid_size=grid_size_to_use, flip_y=args.flip_y, point_size=args.point_size, alpha=0.9)
            else:
                plot_points_on_map_image(args.map_image, points, out_path, grid_size=grid_size_to_use, flip_y=args.flip_y, show_heatmap=args.heatmap, point_size=args.point_size)
            print(f"Saved {out_path}")
            return

    # Folder mode (--in)
    in_dir = args.in_dir
    if not in_dir.exists():
        print(f"Input folder {in_dir} does not exist.")
        return

    if args.aggregate:
        if args.team_name:
            aggregate_folder_and_plot_by_teamname(in_dir, args.map_image, out_dir, args.team_name, args.grid_size, args.flip_y, windows_min, ward_types, args.player_slot, args.verbose)
            return
        out_path = out_dir / f"wards_agg_all_{args.team or 'all'}_combined.png"
        files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.name.startswith("match_") and p.name.endswith("_wards.csv")])
        if args.verbose:
            print(f"[diag] found {len(files)} files in {in_dir}")
        points_by_window: List[List[Dict[str,Any]]] = [[] for _ in windows_min]
        for f in files:
            rows = read_csv_rows(f)
            pts = collect_points_for_team(rows, team_choice=args.team, team_name_or_id=None, time_start=None, time_end=None, player_slot=args.player_slot)
            if not pts:
                continue
            for p in pts:
                t = p.get("time")
                assigned = False
                for idx, (a_min, b_min) in enumerate(windows_min):
                    if t is None:
                        if idx == 0:
                            points_by_window[idx].append(p); assigned = True; break
                        else:
                            continue
                    t_min = float(t) / 60.0
                    if a_min is not None and t_min < a_min:
                        continue
                    if b_min is None:
                        if t_min >= a_min:
                            points_by_window[idx].append(p); assigned = True; break
                    else:
                        if t_min >= a_min and t_min < b_min:
                            points_by_window[idx].append(p); assigned = True; break
                if not assigned:
                    if windows_min and windows_min[-1][1] is None:
                        points_by_window[-1].append(p)
                    else:
                        continue
        summary_path = out_path.with_suffix(".summary.txt")
        with summary_path.open("w", encoding="utf-8") as fh:
            fh.write(f"Aggregated {sum(len(l) for l in points_by_window)} points from {len(files)} files\n")
            for idx, lst in enumerate(points_by_window):
                a_min, b_min = windows_min[idx]
                label = f"{a_min}-{b_min}" if b_min is not None else f"{a_min}+"
                fh.write(f" window {idx+1} ({label} min): {len(lst)} points\n")
        if args.verbose:
            print(f"[diag] wrote summary {summary_path}")
        base_out = out_dir / f"wards_agg_all_{args.team or 'all'}"
        for idx, lst in enumerate(points_by_window):
            a_min, b_min = windows_min[idx]
            label = f"{a_min}-{b_min}" if b_min is not None else f"{a_min}plus"
            win_out = base_out.with_name(base_out.name + f"_window{idx+1}_{label}.png")
            if args.verbose:
                print(f"[diag] plotting window {idx+1} ({label}) with {len(lst)} points -> {win_out}")
            plot_aggregate_map(args.map_image, [lst], win_out, grid_size=args.grid_size, flip_y=args.flip_y, point_size=28, alpha=0.8)
        if args.verbose:
            print(f"[diag] plotting combined map -> {out_path}")
        plot_aggregate_map(args.map_image, points_by_window, out_path, grid_size=args.grid_size, flip_y=args.flip_y, point_size=28, alpha=0.7)
        print(f"Aggregated map written to {out_path}")
        return

    csv_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.name.startswith("match_") and p.name.endswith("_wards.csv")])
    for f in csv_files:
        rows = read_csv_rows(f)
        grid_size_to_use = args.grid_size if args.grid_size != 0 else infer_grid_size_from_rows(rows)
        points = collect_points_for_team(rows, team_choice=args.team, team_name_or_id=None, time_start=args.start, time_end=args.end, player_slot=args.player_slot)
        out_path = out_dir / (f.stem + f"_{args.team or 'all'}_map.png")
        if not points:
            if args.verbose:
                print(f"[diag] no points for file {f.name} with current filters")
            continue
        plot_aggregate_map(args.map_image, [points], out_path, grid_size=grid_size_to_use, flip_y=args.flip_y, point_size=args.point_size, alpha=0.9)
        if args.verbose:
            print(f"[write] {out_path}")

if __name__ == "__main__":
    main()