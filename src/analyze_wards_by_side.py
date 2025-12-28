#!/usr/bin/env python3
"""
analyze_wards_by_side.py

Analyze ward placement for a team, producing separate outputs for when the team
is Radiant and when the team is Dire.

This version matches the plotting/coordinate behavior of
plot_wards_on_map_team_aggregate.py (collector subtracts 64 once, mapping uses
(grid_size-1) denominator). For now heatmap/density overlay is disabled — script
only creates scatter/outlined-circle maps per window and per side.

Usage example:
  python3 analyze_wards_by_side.py \
    --ward-dir data/obs_logs --gold-dir data/gold_timeline \
    --team-name "Xtreme Gaming" --out ./analysis_out --map images/map_1024.jpg \
    --grid-size 128 --normalize-grid --verbose
"""
from pathlib import Path
import argparse
import csv
import re
import math
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Colors/markers
TEAM_COLORS = {"radiant": "#00FF66", "dire": "#FF3333", "unknown": "#999999"}
WARD_MARKERS = {"obs": "o", "sen": "s", "unknown": "x"}

DEFAULT_WINDOWS = [(0, 10), (10, 25), (25, 40), (40, None)]


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-_\.]+", "_", str(s)).strip("_")


def parse_windows_arg(wstr: Optional[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    if not wstr:
        return DEFAULT_WINDOWS
    toks = [t.strip() for t in wstr.split(",") if t.strip()]
    out = []
    for tok in toks:
        if tok.endswith("-"):
            out.append((int(tok[:-1]), None))
        else:
            a, b = tok.split("-")
            out.append((int(a), int(b)))
    return out


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if pd:
        try:
            df = pd.read_csv(path)
            return df.to_dict(orient="records")
        except Exception:
            pass
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def parse_xy_from_row(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
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
                        return x, y
                    except Exception:
                        continue
    key = row.get("key") or row.get("pos_key") or ""
    if isinstance(key, str) and key:
        m = re.search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", key)
        if m:
            try:
                return float(m.group(1)), float(m.group(2))
            except Exception:
                pass
        m2 = re.search(r"\[(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\]", key)
        if m2:
            try:
                return float(m2.group(1)), float(m2.group(2))
            except Exception:
                pass
    return None


def infer_grid_size_from_rows(rows: List[Dict[str, Any]], snap_to=(128, 256, 512, 1024)) -> int:
    max_coord = 0.0
    for r in rows:
        xy = parse_xy_from_row(r)
        if not xy:
            continue
        gx, gy = xy
        max_coord = max(max_coord, abs(gx), abs(gy))
    if max_coord <= 0:
        return 128
    candidate = int(math.ceil(max_coord)) + 1
    for s in snap_to:
        if abs(candidate - s) <= max(1, int(0.02 * s)):
            return s
    return max(128, candidate)


# --- team diff helper (was missing) ---
def team_diff_for_match_and_side(gold_df: Optional[pd.DataFrame], t_sec: float, side: str) -> Optional[float]:
    """
    Given standardized gold dataframe for a match and a time (seconds),
    return the team-specific gold difference (team - opponent) for the given side.
    gold_df is expected to have either:
      - 'gold_diff' column (radiant - dire), or
      - an adv-like column such as 'radiant_gold_adv'
    If gold_df has 'gold_diff' we interpolate that and return value (positive means Radiant leads).
    For side == 'radiant' return gold_diff; for side == 'dire' return -gold_diff.
    If only adv column is present (radiant advantage), treat that as gold_diff.
    Returns None when interpolation not possible.
    """
    if gold_df is None or gold_df.empty:
        return None
    if 'time' not in gold_df.columns:
        return None
    try:
        times = gold_df['time'].to_numpy(dtype=float)
    except Exception:
        return None

    # Prefer 'gold_diff' (radiant - dire)
    if 'gold_diff' in gold_df.columns:
        try:
            arr = gold_df['gold_diff'].to_numpy(dtype=float)
            val = float(np.interp(float(t_sec), times, arr))
            return val if side == 'radiant' else -val
        except Exception:
            return None

    # Try common advantage columns
    adv_col = None
    for c in gold_df.columns:
        c_low = c.lower()
        if 'radiant' in c_low and 'adv' in c_low:
            adv_col = c
            break
        if 'adv' in c_low and 'gold' in c_low:
            adv_col = c
            break
    if adv_col is not None:
        try:
            arr = gold_df[adv_col].to_numpy(dtype=float)
            val = float(np.interp(float(t_sec), times, arr))
            return val if side == 'radiant' else -val
        except Exception:
            return None

    # Fallback: look for any column with 'adv' or 'gold_diff' substring
    for c in gold_df.columns:
        if 'adv' in c.lower() or 'gold_diff' in c.lower():
            try:
                arr = gold_df[c].to_numpy(dtype=float)
                val = float(np.interp(float(t_sec), times, arr))
                return val if side == 'radiant' else -val
            except Exception:
                continue

    return None


# --- Collector (based on plot_wards_on_map_team_aggregate.py) ---
def collect_points_for_team(rows: List[Dict[str, Any]],
                            team_choice: Optional[str],
                            team_name_or_id: Optional[str],
                            time_start: Optional[float],
                            time_end: Optional[float],
                            player_slot: Optional[int],
                            only_types: Optional[List[str]] = None,
                            verbose: bool = False) -> List[Dict[str, Any]]:
    pts: List[Dict[str, Any]] = []
    seen = set()
    raw_rows = 0
    placement_rows = 0
    per_match_counts = {}

    for r in rows:
        raw_rows += 1

        # team name/id filter (if provided)
        if team_name_or_id is not None:
            matched = False
            # try id
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

        # placement-like indicator
        row_type = str(r.get("type") or r.get("event") or "").lower()
        if "_log" not in row_type:
            continue
        placement_rows += 1

        # ward_type normalization
        ward_type_raw = (str(r.get("ward_type") or r.get("type") or "")).lower()
        if ward_type_raw.startswith("obs"):
            wtype = "obs"
        elif ward_type_raw.startswith("sen"):
            wtype = "sen"
        else:
            wtype = "unknown"
        if only_types and wtype not in only_types:
            continue

        # parse xy
        xy = parse_xy_from_row(r)
        if not xy:
            continue
        x, y = xy

        # dedupe key
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

        # subtract 64 once here (collector)
        try:
            x = float(x) - 64.0
            y = float(y) - 64.0
        except Exception:
            continue

        pts.append({
            "x": x,
            "y": y,
            "ward_type": wtype,
            "team": team_field,
            "time": float(r["time"]) if "time" in r and r["time"] not in (None, "") else None,
            "match": match_id,
            "uniq_id": uniq
        })

        per_match_counts.setdefault(match_id, {"raw": 0, "placement": 0, "unique": 0})
        per_match_counts[match_id]["placement"] += 1
        per_match_counts[match_id]["unique"] += 1

    if verbose:
        print(f"[diag] rows read={raw_rows}, placement-like rows={placement_rows}, unique placements after dedup={len(pts)}")
    return pts


def map_to_image_coords(x: float, y: float, grid_size: int, img_w: int, img_h: int, flip_y: bool) -> Tuple[float, float]:
    if grid_size <= 1:
        return img_w / 2.0, img_h / 2.0
    max_index = grid_size - 1
    try:
        gx = float(x)
    except Exception:
        gx = 0.0
    try:
        gy = float(y)
    except Exception:
        gy = 0.0
    gx = max(0.0, min(gx, float(max_index)))
    gy = max(0.0, min(gy, float(max_index)))
    px = (gx / max_index) * (img_w - 1)
    py = (gy / max_index) * (img_h - 1)
    if flip_y:
        py = img_h - py
    return px, py


def plot_points_on_map_image(img_path: Path, points: List[Dict[str, Any]], out_path: Path,
                             grid_size: int = 128, flip_y: bool = False,
                             point_size: int = 40, alpha: float = 0.9):
    img = Image.open(img_path).convert("RGBA")
    img_w, img_h = img.size

    scale_factor = max(1.0, min(img_w, img_h) / 512.0)
    effective_point_size = int(point_size * scale_factor)

    xs = []
    ys = []
    teams = []
    wtypes = []
    for p in points:
        px, py = map_to_image_coords(p["x"], p["y"], grid_size, img_w, img_h, flip_y=flip_y)
        xs.append(px); ys.append(py); teams.append(p.get("team")); wtypes.append(p.get("ward_type"))

    dpi = 128
    plt.figure(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
    ax = plt.gca()
    ax.imshow(img, extent=[0, img_w, 0, img_h])

    # Group by ward_type
    groups = {}
    for i in range(len(xs)):
        key = wtypes[i]
        groups.setdefault(key, []).append((xs[i], ys[i], teams[i]))

    legend_loc = "lower left" if img_w >= 800 else "upper right"

    for wtype, coords in groups.items():
        coords_np = np.array([(c[0], c[1]) for c in coords])
        if coords_np.size == 0:
            continue
        color = TEAM_COLORS.get(coords[0][2], TEAM_COLORS["unknown"])
        marker = WARD_MARKERS.get(wtype, "o")
        size = effective_point_size if wtype == "obs" else max(8, int(effective_point_size * 0.6))
        ax.scatter(coords_np[:, 0], coords_np[:, 1], c=color, s=size, marker=marker,
                   edgecolors="k", linewidths=0.4, alpha=alpha, label=f"{wtype}")

        # overlay hollow red circles
        from matplotlib.patches import Circle
        for xi, yi in zip(coords_np[:, 0], coords_np[:, 1]):
            circ = Circle((xi, yi), radius=10, facecolor='none', edgecolor='red', linewidth=1.4, alpha=0.95, zorder=5)
            ax.add_patch(circ)

    ax.set_xlim(0, img_w)
    ax.set_ylim(0, img_h)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc=legend_loc, fontsize="small", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def aggregate_hotspots(ward_df, out_dir: Path, top_n: int = 20):
    out_rows = []
    for (w, side, state), group in ward_df.groupby(['window', 'side', 'state']):
        if group.empty:
            continue
        grp = group.copy()
        grp['gx'] = grp['x'].round().astype(int)
        grp['gy'] = grp['y'].round().astype(int)
        counts = grp.groupby(['gx', 'gy']).size().reset_index(name='count').sort_values('count', ascending=False)
        top = counts.head(top_n)
        for _, r in top.iterrows():
            out_rows.append({
                "window": int(w),
                "side": side,
                "state": state,
                "gx": int(r['gx']),
                "gy": int(r['gy']),
                "count": int(r['count'])
            })
    out_df = pd.DataFrame(out_rows)
    out_path = out_dir / "hotspots_window_side_state.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def extract_and_annotate_all(ward_dir: Path, gold_map: Dict[str, pd.DataFrame], team_name: str,
                             windows: List[Tuple[Optional[int], Optional[int]]], grid_size_arg: int,
                             normalize_grid: bool, normalize_pixel: bool, map_img_path: Optional[Path],
                             pct_low: float, pct_high: float, only_type: Optional[str], verbose: bool) -> pd.DataFrame:
    rows = []
    ward_files = sorted([p for p in ward_dir.iterdir() if p.is_file() and p.name.startswith("match_") and p.name.endswith("_wards.csv")])
    for f in ward_files:
        try:
            raw_rows = read_csv_rows(f)
        except Exception:
            continue

        # compute placement-like count for diagnostics
        placement_like = 0
        for rr in raw_rows:
            if "_log" in (str(rr.get("type") or rr.get("event") or "")).lower():
                placement_like += 1

        # for each side separately collect points (collector handles team_name_or_id)
        for side in ("radiant", "dire"):
            pts = collect_points_for_team(raw_rows, team_choice=side, team_name_or_id=team_name,
                                          time_start=None, time_end=None, player_slot=None,
                                          only_types=[only_type] if only_type else None, verbose=False)
            if not pts:
                if verbose:
                    print(f"[diag] file {f.name} as {side}: placement_like_rows={placement_like}, unique_pts_for_team_as_{side}=0 (skipping)")
                continue

            # choose effective grid size (if user asked auto per-file)
            effective_grid = grid_size_arg
            if effective_grid == 0:
                effective_grid = infer_grid_size_from_rows(raw_rows)
                if verbose:
                    print(f"[diag] inferred grid_size={effective_grid} for {f.name}")

            mid = str(pts[0].get("match") or f.stem.split("_")[1])

            gold_df = gold_map.get(mid)
            gold_present = (mid in gold_map)

            # diagnostics: before processing per-point, print file-level summary
            if verbose:
                print(f"[diag] processing file {f.name} (match {mid}) as {side}: raw_rows={len(raw_rows)}, placement_like={placement_like}, collected_pts={len(pts)}, gold_present={gold_present}")

            # if pixel normalization requested, get image dims
            img_w = img_h = None
            if normalize_pixel and map_img_path and map_img_path.exists():
                img = Image.open(map_img_path)
                img_w, img_h = img.size

            # counters for reasons of skipping
            skipped_no_time = 0
            skipped_no_gold = 0
            appended = 0

            for p in pts:
                t = p.get("time")
                if t is None:
                    skipped_no_time += 1
                    continue
                td = None
                if gold_df is not None:
                    td = team_diff_for_match_and_side(gold_df, t, side)
                if td is None:
                    skipped_no_gold += 1
                    continue

                mins = t / 60.0
                # window assignment
                widx = 0
                assigned = False
                for idx, (a, b) in enumerate(windows):
                    if b is None:
                        if mins >= a:
                            widx = idx; assigned = True; break
                    else:
                        if mins >= a and mins < b:
                            widx = idx; assigned = True; break
                if not assigned:
                    widx = len(windows) - 1

                x_n, y_n = p['x'], p['y']
                # normalization (mirror) if requested
                if side == 'dire' and (normalize_grid or normalize_pixel):
                    if normalize_pixel and img_w is not None:
                        px, py = map_to_image_coords(x_n, y_n, effective_grid, img_w, img_h, flip_y=False)
                        px_m = img_w - px
                        py_m = img_h - py
                        gx = (px_m / (img_w - 1)) * (effective_grid - 1)
                        gy = (py_m / (img_h - 1)) * (effective_grid - 1)
                        x_n, y_n = gx, gy
                    elif normalize_grid:
                        max_idx = effective_grid - 1
                        x_n = max_idx - x_n
                        y_n = max_idx - y_n

                rows.append({
                    "match_id": mid,
                    "time": t,
                    "mins": mins,
                    "side": side,
                    "x": x_n,
                    "y": y_n,
                    "team_diff": td,
                    "window": widx
                })
                appended += 1

            if verbose:
                print(f"[diag] file {f.name} as {side}: appended={appended}, skipped_no_time={skipped_no_time}, skipped_no_gold={skipped_no_gold}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_thresholds_by_window_and_side(ward_df: pd.DataFrame, windows: List[Tuple[Optional[int], Optional[int]]],
                                          pct_low: float, pct_high: float, method: str = "percentile",
                                          abs_thresholds: Optional[List[float]] = None) -> Dict[Tuple[int, str], Tuple[float, float]]:
    thresholds = {}
    for w in range(len(windows)):
        for side in ('radiant', 'dire'):
            sub = ward_df[(ward_df['window'] == w) & (ward_df['side'] == side)]
            arr = sub['team_diff'].dropna()
            if method == "absolute":
                if abs_thresholds and len(abs_thresholds) > w:
                    thr = abs_thresholds[w]
                    thresholds[(w, side)] = (-thr, thr)
                else:
                    thresholds[(w, side)] = (-0.0, 0.0)
            else:
                if len(arr) >= 5:
                    low = float(np.quantile(arr, pct_low))
                    high = float(np.quantile(arr, pct_high))
                else:
                    all_side = ward_df[ward_df['side'] == side]['team_diff'].dropna()
                    if len(all_side) >= 5:
                        low = float(np.quantile(all_side, pct_low))
                        high = float(np.quantile(all_side, pct_high))
                    else:
                        low = -1000.0
                        high = 1000.0
                thresholds[(w, side)] = (low, high)
    return thresholds


def run_pipeline(args):
    windows = parse_windows_arg(args.windows)
    gold_map = {}
    # load gold timelines
    gold_dir_path = Path(args.gold_dir)
    for p in gold_dir_path.glob("match_*_gold_std.csv"):
        mid = p.stem.split("_")[1]
        try:
            gold_map[mid] = pd.read_csv(p)
        except Exception:
            continue
    if args.verbose:
        print(f"[diag] loaded {len(gold_map)} gold timelines from {args.gold_dir}")
        # list a sample of keys
        if len(gold_map) > 0:
            sample_keys = list(gold_map.keys())[:50]
            print(f"[diag] sample gold match ids: {sample_keys if len(sample_keys)<=50 else sample_keys[:50]}")
    # check ward files against expected gold files (verbose)
    ward_files_list = sorted([p for p in Path(args.ward_dir).iterdir() if p.is_file() and p.name.startswith("match_") and p.name.endswith("_wards.csv")])
    missing_gold = []
    for f in ward_files_list:
        try:
            mid = f.stem.split("_")[1]
        except Exception:
            continue
        expected = gold_dir_path / f"match_{mid}_gold_std.csv"
        if not expected.exists():
            missing_gold.append((f.name, str(expected)))
    if args.verbose:
        if missing_gold:
            print(f"[warn] missing gold files for {len(missing_gold)} ward files (showing up to 50):")
            for wf, gp in missing_gold[:50]:
                print(f"  ward file {wf} -> expected gold file {gp} missing")
        else:
            print("[diag] all expected gold files present for ward files")

    ward_df = extract_and_annotate_all(Path(args.ward_dir), gold_map, args.team_name, windows,
                                       grid_size_arg=args.grid_size,
                                       normalize_grid=args.normalize_grid,
                                       normalize_pixel=args.normalize_pixel,
                                       map_img_path=args.map, pct_low=args.pct_low, pct_high=args.pct_high,
                                       only_type=args.ward_only_type if args.ward_only_type else None,
                                       verbose=args.verbose)
    if ward_df.empty:
        print("[info] no ward data found for that team. Exiting.")
        return

    thresholds = compute_thresholds_by_window_and_side(ward_df, windows, args.pct_low, args.pct_high,
                                                       method=args.method,
                                                       abs_thresholds=[float(x) for x in args.abs_thresholds.split(",")] if args.abs_thresholds else None)
    if args.verbose:
        print("[diag] thresholds per (window,side):")
        for k, v in thresholds.items():
            print(f"  {k}: low={v[0]:.1f}, high={v[1]:.1f}")

    # classify state
    def classify_row(r):
        low, high = thresholds.get((int(r['window']), r['side']), (-1000, 1000))
        return "disadv" if r['team_diff'] <= low else ("adv" if r['team_diff'] >= high else "even")
    ward_df['state'] = ward_df.apply(classify_row, axis=1)

    # per-side outputs
    base_out = Path(args.out) / safe_name(args.team_name)
    for side in ('radiant', 'dire'):
        sub = ward_df[ward_df['side'] == side].copy()
        if sub.empty:
            if args.verbose:
                print(f"[info] no data for {args.team_name} as {side}")
            continue
        side_out = base_out / side
        side_out.mkdir(parents=True, exist_ok=True)
        ann_csv = side_out / f"{safe_name(args.team_name)}_wards_annotated_{side}.csv"
        sub.to_csv(ann_csv, index=False)
        if args.verbose:
            print(f"[write] annotated CSV -> {ann_csv} ({len(sub)} rows)")

        # hotspots CSV
        hot_csv = aggregate_hotspots(sub, side_out, top_n=args.top_n)
        if args.verbose:
            print(f"[write] hotspots CSV -> {hot_csv}")

        # plotting per (window, state)
        if args.map and Path(args.map).exists():
            for (w, state), group in sub.groupby(['window', 'state']):
                out_png = side_out / f"{safe_name(args.team_name)}_window{w}_{side}_{state}.png"
                if group.empty:
                    continue
                points = group.to_dict(orient='records')
                plot_points_on_map_image(Path(args.map), points, out_png, grid_size=args.grid_size, flip_y=args.flip_y, point_size=40, alpha=0.9)
                if args.verbose:
                    print(f"[plot] wrote {out_png}")

    print("Done. outputs under:", base_out)


def main():
    p = argparse.ArgumentParser(description="Analyze ward hotspots by team side and time window.")
    p.add_argument("--ward-dir", type=Path, required=True)
    p.add_argument("--gold-dir", type=Path, required=True)
    p.add_argument("--team-name", type=str, required=True)
    p.add_argument("--out", type=Path, default=Path("./analysis_out"))
    p.add_argument("--map", type=Path, default=None, help="Optional minimap image for overlays")
    p.add_argument("--grid-size", type=int, default=128, help="Grid size used by coordinates (default 128). Pass 0 to auto-detect per file.")
    p.add_argument("--flip-y", action="store_true", help="Flip Y axis when mapping coords to image")
    p.add_argument("--normalize-grid", action="store_true", help="Mirror coordinates in grid space (no map required)")
    p.add_argument("--normalize-pixel", action="store_true", help="Mirror coordinates in pixel space (requires --map)")
    p.add_argument("--pct-low", type=float, default=0.33)
    p.add_argument("--pct-high", type=float, default=0.67)
    p.add_argument("--method", type=str, choices=["percentile", "absolute"], default="percentile")
    p.add_argument("--abs-thresholds", type=str, default="", help="Comma list for absolute thresholds per window")
    p.add_argument("--ward-only-type", type=str, default="obs", help="Filter by ward type (obs/sen) or empty for all")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--windows", type=str, default="", help="Windows like '0-10,10-25,25-40,40-'")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.normalize_pixel and (args.map is None or not Path(args.map).exists()):
        print("[error] --normalize-pixel requires a valid --map image path")
        return

    # allow shorthand
    if args.normalize_grid is None:
        args.normalize_grid = False

    # parse windows
    args.windows = args.windows if args.windows is not None else ""
    run_pipeline(args)


if __name__ == "__main__":
    main()