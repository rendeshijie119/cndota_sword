#!/usr/bin/env python3
"""
plot_wards_on_map_team.py

Read per-match ward CSV files (match_<id>_wards.csv) in a folder or a single CSV,
filter to one team (radiant or dire), and plot that team's ward placements on a
Dota 2 minimap image.

Improvements:
- Uses (grid_size - 1) as denominator so 0..(grid_size-1) maps exactly to 0..image pixels.
- Auto-detects grid_size from CSV rows when --grid-size 0 is provided (snaps to 256/512/1024).
- Prints diagnostics (max_x, max_y, chosen grid_size) for each CSV processed.
- Keeps auto-scaling of marker sizes for large images and improved legend placement.

Usage examples:
    python3 plot_wards_on_map_team.py --csv ./obs_logs/match_8460339590_wards.csv --map minimap_900.png --team dire
    python3 plot_wards_on_map_team.py --in ./obs_logs --map minimap_900.png --team radiant --grid-size 0

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
    # fallback to 'key' like "[124,124]" or "124,124"
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

def infer_grid_size_from_rows(rows: List[Dict[str, Any]], snap_to=(256,512,1024)) -> int:
    """
    Inspect x/y values in rows and return a sensible grid_size.
    Strategy:
      - compute observed max coordinate (ignore negatives)
      - set candidate = ceil(max_coord) + 1
      - snap to common sizes if within ~2%
      - otherwise return max(256, candidate)
    """
    max_coord = 0.0
    for r in rows:
        try:
            # prefer numeric columns if present
            xval = r.get("x") or r.get("pos_x") or r.get("posx") or r.get("position_x")
            yval = r.get("y") or r.get("pos_y") or r.get("posy") or r.get("position_y")
            if xval is None or yval is None:
                # try key
                xy = parse_xy_from_row(r)
                if xy:
                    gx, gy = xy
                else:
                    continue
            else:
                gx = float(xval)
                gy = float(yval)
            max_coord = max(max_coord, abs(gx), abs(gy))
        except Exception:
            continue
    if max_coord <= 0:
        return 256
    candidate = int(math.ceil(max_coord)) + 1
    for s in snap_to:
        if abs(candidate - s) <= max(1, int(0.02 * s)):
            return s
    return max(256, candidate)

# Update collect_points_for_team to accept player_slot
def collect_points_for_team(rows: List[Dict[str, Any]],
                            team_choice: str,
                            time_start: Optional[float],
                            time_end: Optional[float],
                            player_slot: Optional[int]) -> List[Dict[str, Any]]:
    pts = []
    for r in rows:
        # filter by chosen team
        team = (str(r.get("radiant_or_dire") or r.get("team") or "")).lower()
        if team_choice and team != team_choice:
            continue

        # optional time filtering
        if time_start is not None or time_end is not None:
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

        # optional player_slot filtering
        if player_slot is not None:
            slot_val = r.get("player_slot")
            try:
                if int(slot_val) != player_slot:
                    continue
            except Exception:
                continue

        xy = parse_xy_from_row(r)
        if not xy:
            continue
        x, y = xy

        x -= 64
        y -= 64

        ward_type = (str(r.get("ward_type") or r.get("type") or "")).lower()
        if ward_type.startswith("obs"):
            wtype = "obs"
        elif ward_type.startswith("sen"):
            wtype = "sen"
        else:
            wtype = "unknown"
        pts.append({
            "x": x,
            "y": y,
            "ward_type": wtype,
            "team": team,
            "account_id": r.get("account_id") or "",
            "player_slot": r.get("player_slot") or "",
            "time": r.get("time") or ""
        })
    return pts


def map_to_image_coords(x: float, y: float, grid_size: int, img_w: int, img_h: int, flip_y: bool) -> Tuple[float, float]:
    """
    Convert grid coords to image pixel coords.
    Use (grid_size - 1) as denominator so values 0 .. grid_size-1 map to 0 .. img_w/img_h.
    Clamp to the valid grid range [0, grid_size-1] first.
    """
    # img_w = 1024
    # img_h = 1024
    if grid_size <= 1:
        return (img_w / 2.0, img_h / 2.0)
    max_index = grid_size - 1
    try:
        gx = float(x)
    except Exception:
        gx = 0.0
    try:
        gy = float(y)
    except Exception:
        gy = 0.0
    # clamp
    if gx < 0:
        gx = 0.0
    if gy < 0:
        gy = 0.0
    if gx > max_index:
        gx = float(max_index)
    if gy > max_index:
        gy = float(max_index)
    px = (gx / max_index) * (img_w - 1)
    py = (gy / max_index) * (img_h - 1)
    if flip_y:
        py = img_h - py
    
    print(f"Mapping grid ({x},{y}) to image coords ({px:.1f},{py:.1f}) with grid_size={grid_size}, img_size=({img_w},{img_h}), flip_y={flip_y}")
    
    return px, py

def plot_points_on_map_image(img_path: Path, points: List[Dict[str, Any]], out_path: Path,
                             grid_size: int = 256, flip_y: bool = False, show_heatmap: bool = False,
                             point_size: int = 60, alpha: float = 0.9, figsize=(8,8)):
    """
    Auto-scales marker sizes when input image dimensions are large (e.g., 900x900).
    """
    img = Image.open(img_path).convert("RGBA")
    img_w, img_h = img.size

    print(f"Plotting {len(points)} wards on map image {img_path.name} ({img_w}x{img_h}), grid_size={grid_size}, flip_y={flip_y}, heatmap={show_heatmap}")
    # compute scale factor based on image size (512 is baseline)
    scale_factor = max(1.0, min(img_w, img_h) / 512.0)
    effective_point_size = int(point_size * scale_factor)

    xs = []
    ys = []
    teams = []
    wtypes = []
    for p in points:
        px, py = map_to_image_coords(p["x"], p["y"], grid_size, img_w, img_h, flip_y=flip_y)
        xs.append(px)
        ys.append(py)
        teams.append(p["team"])
        wtypes.append(p["ward_type"])

    dpi=128
    plt.figure(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)

    ax = plt.gca()
    ax.imshow(img, extent=[0, img_w, 0, img_h])

    if show_heatmap and len(xs) > 1:
        try:
            sns.kdeplot(x=xs, y=ys, levels=5, fill=True, cmap="magma", alpha=0.4, thresh=0.05, bw_method="scott", ax=ax)
        except Exception:
            pass

    # Group by ward_type for plotting markers
    groups = {}
    for i in range(len(xs)):
        key = (wtypes[i])
        groups.setdefault(key, []).append((xs[i], ys[i], teams[i]))

    # place legend to bottom-left for wide images
    legend_loc = "lower left" if img_w >= 800 else "upper right"

    for wtype, coords in groups.items():
        coords_np = np.array([(c[0], c[1]) for c in coords])
        print(coords_np)
        if coords_np.size == 0:
            continue
        # use the team color from the first point (all points same team because we filtered earlier)
        color = TEAM_COLORS.get(coords[0][2], TEAM_COLORS["unknown"])
        marker = WARD_MARKERS.get(wtype, "o")
        size = effective_point_size if wtype == "obs" else max(8, int(effective_point_size * 0.6))
        ax.scatter(coords_np[:,0], coords_np[:,1], c=color, s=size, marker=marker,
                   edgecolors="k", linewidths=0.4, alpha=alpha, label=f"{wtype}")

    ax.set_xlim(0, img_w)
    ax.set_ylim(0, img_h)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(out_path.stem)
    ax.legend(loc=legend_loc, fontsize="small", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Saved map with {len(points)} wards to {out_path}")


def process_single_csv_for_team(csv_path: Path, map_path: Path, out_path: Path,
                                team_choice: str, grid_size: int, flip_y: bool,
                                time_start: Optional[float], time_end: Optional[float],
                                show_heatmap: bool, point_size: int,
                                player_slot: Optional[int]):
    rows = read_csv_rows(csv_path)
    # infer grid if requested (grid_size == 0)
    if grid_size == 0:
        inferred = infer_grid_size_from_rows(rows)
        print(f"[diagnostic] inferred grid_size={inferred} from CSV (max coords scanned).")
        grid_size_to_use = inferred
    else:
        grid_size_to_use = grid_size

    # show diagnostics of max coord
    max_x = max_y = 0.0
    for r in rows:
        xy = parse_xy_from_row(r)
        if xy:
            max_x = max(max_x, abs(xy[0])); max_y = max(max_y, abs(xy[1]))
    print(f"[diagnostic] CSV={csv_path.name} max_x={max_x:.1f} max_y={max_y:.1f} using grid_size={grid_size_to_use}")

    # Update call to collect_points_for_team in process_single_csv_for_team
    points = collect_points_for_team(rows, team_choice=team_choice,
                                    time_start=time_start, time_end=time_end,
                                    player_slot=player_slot)

    if not points:
        print(f"No ward points found for team '{team_choice}' in {csv_path} for the given time window.")
        return
    plot_points_on_map_image(map_path, points, out_path,
                             grid_size=grid_size_to_use, flip_y=flip_y, show_heatmap=show_heatmap,
                             point_size=point_size)

def main():
    p = argparse.ArgumentParser(description="Plot one team's ward events on a Dota minimap image")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=Path, help="Single match CSV to plot (match_<id>_wards.csv)")
    group.add_argument("--in", dest="in_dir", type=Path, help="Folder containing match_*_wards.csv files")
    p.add_argument("--map", dest="map_image", type=Path, required=True, help="Path to minimap image PNG/JPG")
    p.add_argument("--team", type=str, required=True, choices=["radiant", "dire"], help="Which team to plot: radiant or dire")
    p.add_argument("--player-slot", type=int, default=None, help="Only plot wards placed by this player slot (0–255)")
    p.add_argument("--out", dest="out_dir", type=Path, default=Path("./ward_maps"), help="Output folder for generated map images")
    p.add_argument("--grid-size", type=int, default=256, help="Grid size used by OpenDota coords (default 256). Pass 0 to auto-detect from CSV.")
    p.add_argument("--flip-y", action="store_true", help="Flip Y axis when mapping coords to image")
    p.add_argument("--start", type=float, default=None, help="Start time (seconds) inclusive to filter events")
    p.add_argument("--end", type=float, default=None, help="End time (seconds) inclusive to filter events")
    p.add_argument("--heatmap", action="store_true", help="Overlay a heatmap (density) instead of plain scatter")
    p.add_argument("--point-size", type=int, default=60, help="Base point size for obs wards (auto-scaled for large images)")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        csv_path = args.csv
        if not csv_path.exists():
            print(f"CSV file {csv_path} not found.")
            return
        out_path = out_dir / (csv_path.stem + f"_{args.team}_map.png")
        process_single_csv_for_team(csv_path, args.map_image, out_path, args.team, args.grid_size, args.flip_y, args.start, args.end, args.heatmap, args.point_size, args.player_slot)
        return

    in_dir = args.in_dir
    if not in_dir.exists():
        print(f"Input folder {in_dir} does not exist.")
        return
    csv_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.name.startswith("match_") and p.name.endswith("_wards.csv")])
    if not csv_files:
        print("No match_..._wards.csv files found in", in_dir)
        return

    for f in csv_files:
        out_path = out_dir / (f.stem + f"_{args.team}_map.png")
        process_single_csv_for_team(f, args.map_image, out_path, args.team, args.grid_size, args.flip_y, args.start, args.end, args.heatmap, args.point_size)

if __name__ == "__main__":
    main()