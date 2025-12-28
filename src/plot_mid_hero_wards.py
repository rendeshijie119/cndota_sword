#!/usr/bin/env python3
"""
plot_mid_hero_wards.py

Aggregate and plot ward placements for a specific team's mid player,
grouped by the hero that player picked, and separated by team side
(radiant / dire).

This version trusts the canonical per-match ward CSVs (produced by
download_obs_logs_from_matches.py) to contain an explicit lifetime column
(lifetime, duration, lifetime_s, etc). The plotting code will prefer
that explicit lifetime and will NOT attempt to recompute lifetime from
removed_time/time pairs.

v28 changes:
- outer red ring for observer wards scales per-point with the scatter size so
  the ring remains visually consistent with lifetime-scaled point size.
- added set_chinese_font_prefer() to pick a system CJK-capable font so Chinese
  titles/labels render correctly (call early in main()).
- added options to use configured max-lifetime defaults (instead of inferring
  lifetimes from the data) and optional "clock" (wedge) style for observer
  inner fill.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
except Exception:
    np = None
    plt = None
    Image = None

# Ward plotting styles (distinct icons/colors for obs vs sen vs unknown)
WARD_STYLES = {
    "obs": {"facecolor": "yellow", "edgecolor": "k", "marker": "o"},
    "sen": {"facecolor": "cyan", "edgecolor": "k", "marker": "s"},
    "unknown": {"facecolor": "orange", "edgecolor": "k", "marker": "x"},
}

# Defaults
DEFAULT_MATCHES_DIR = Path("data/matches")
DEFAULT_WARDS_DIR = Path("data/obs_logs")
DEFAULT_HERO_MAP = Path("data/heros_mapping/hero_map.csv")
DEFAULT_TIME_START = -60
DEFAULT_TIME_END = 7200

# Default max lifetimes (seconds) used when NOT auto-computing range from data
# (you can override via CLI args)
DEFAULT_OBS_MAX_LIFETIME = 360.0   # default: observer ward 6 minutes (common)
DEFAULT_SEN_MAX_LIFETIME = 0.0     # sentry: usually 'permanent until destroyed' -> 0.0 means no auto-scale

# ----------------- helpers -----------------
def safe_name(s: Optional[str]) -> str:
    if s is None:
        return "unknown"
    s2 = str(s).strip()
    # allow unicode letters (keeps Chinese), replace problematic
    return re.sub(r"[^\w\-\.\s]+", "_", s2).strip("_")[:200]

def read_json_optional(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """
    Read CSV with utf-8-sig to tolerate BOM produced by some editors/tools.
    Returns list of dict rows (empty list on error).
    """
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
                out.append(r)
    except Exception:
        pass
    return out

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

def _extract_numeric_from_row_keys(row: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in row and row[k] not in (None, "", "nan"):
            try:
                return float(row[k])
            except Exception:
                # try to extract numeric substring
                try:
                    s = str(row[k])
                    m = re.search(r"-?\d+\.?\d*", s)
                    if m:
                        return float(m.group(0))
                except Exception:
                    pass
    return None

# ----------------- lifetime helpers (explicit-only) -----------------
def _extract_lifetime_from_row(row: Dict[str, Any]) -> Optional[float]:
    """
    Prefer explicit lifetime-like columns only.
    Accepts: lifetime, lifetime_s, life, duration, time_alive, entity_life.
    Returns numeric seconds or None. DOES NOT compute removed_time - time.
    """
    for primary in ("lifetime", "lifetime_s", "life", "duration", "time_alive", "entity_life"):
        if primary in row and row.get(primary) not in (None, "", "nan"):
            try:
                return float(row.get(primary))
            except Exception:
                try:
                    m = re.search(r"-?\d+\.?\d*", str(row.get(primary)))
                    if m:
                        return float(m.group(0))
                except Exception:
                    pass
    return None

# ----------------- Chinese font support -----------------
def set_chinese_font_prefer():
    """
    Try to set a font that supports Chinese so matplotlib can render Chinese text.
    Returns the chosen font name or None if none found.
    """
    try:
        import matplotlib.font_manager as fm
    except Exception:
        return None

    # common font name candidates (cross-platform)
    candidates = [
        "Noto Sans CJK SC",     # common Google Noto (Linux / installed by user)
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "PingFang SC",          # macOS newer
        "STHeiti",              # macOS older
        "Heiti SC",
        "Microsoft YaHei",      # Windows
        "SimHei",               # Windows common
        "WenQuanYi Micro Hei",  # many Linux distros
        "AR PL UKai CN",        # some Chinese fonts on linux
        "Apple LiGothic",       # macOS
        "Arial Unicode MS",     # old cross-platform one (if present)
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    for cand in candidates:
        for a in available:
            if cand.lower() in a.lower() or a.lower() in cand.lower():
                chosen = a
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [chosen] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['axes.unicode_minus'] = False
                return chosen

    for a in available:
        if any(x in a.lower() for x in ("noto", "cjk", "hei", "song", "fang", "yahei", "heiti", "wqy")):
            chosen = a
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [chosen] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['axes.unicode_minus'] = False
            return chosen

    return None

# ----------------- hero map loader -----------------
def load_hero_map_full(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """
    Robust loader for hero_map CSV.
    Returns mapping: hero_id_str -> { 'hero_name':..., 'hero_name_cn':... }
    """
    out: Dict[str, Dict[str, str]] = {}
    if not path:
        return out
    try:
        rows = read_csv_rows(path)
        if rows:
            sample = rows[0]
            id_keys = [k for k in sample.keys() if k and k.strip().lower() in ("id", "hero_id", "hero", "heroid")]
            name_keys = [k for k in sample.keys() if k and k.strip().lower() in ("hero_name", "localized_name", "name")]
            cn_keys = [k for k in sample.keys() if k and k.strip().lower() in ("hero_name_cn", "name_cn", "cn")]
            if id_keys:
                idk = id_keys[0]
                namek = name_keys[0] if name_keys else None
                cnk = cn_keys[0] if cn_keys else None
                for r in rows:
                    hid = r.get(idk)
                    if hid is None or str(hid).strip() == "":
                        continue
                    hid_s = str(hid).strip()
                    hero_name = (r.get(namek) or "").strip() if namek else ""
                    hero_name_cn = (r.get(cnk) or "").strip() if cnk else ""
                    if not hero_name:
                        for v in r.values():
                            if v and str(v).strip():
                                hero_name = str(v).strip(); break
                    out[hid_s] = {"hero_name": hero_name or hid_s, "hero_name_cn": hero_name_cn or (hero_name or hid_s)}
                return out
            else:
                first_key = next(iter(sample.keys()))
                for r in rows:
                    hid = r.get(first_key)
                    if hid is None or str(hid).strip() == "":
                        continue
                    hid_s = str(hid).strip()
                    keys = list(r.keys())
                    hero_name = r.get(keys[1]).strip() if len(keys) > 1 and r.get(keys[1]) else hid_s
                    hero_name_cn = r.get(keys[2]).strip() if len(keys) > 2 and r.get(keys[2]) else hero_name
                    out[hid_s] = {"hero_name": hero_name or hid_s, "hero_name_cn": hero_name_cn or (hero_name or hid_s)}
                return out
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rdr = csv.reader(fh)
            for row in rdr:
                if not row:
                    continue
                hid_s = str(row[0]).strip()
                hero_name = row[1].strip() if len(row) > 1 else hid_s
                hero_name_cn = row[2].strip() if len(row) > 2 else hero_name
                out[hid_s] = {"hero_name": hero_name or hid_s, "hero_name_cn": hero_name_cn or (hero_name or hid_s)}
    except Exception:
        pass
    return out

# ----------------- ward extraction / CSV saving (from download_obs_logs_from_matches) -----------------
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

def collect_ward_events(match_id: Optional[int], match_data: dict, time_start: int, time_end: int,
                        account_name_map: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    players = match_data.get("players", []) or []
    team_meta = get_team_meta(match_data)
    for p in players:
        p_slot = p.get("player_slot")
        if p_slot is None:
            continue
        side = team_for_slot(p_slot)
        account_id = p.get("account_id", "")
        personal_name = get_personal_name_from_player(p)
        if account_name_map is not None and account_id not in (None, ""):
            try:
                account_name_map[str(account_id)] = personal_name
            except Exception:
                pass

        def process_log_list(ev_list: list, ward_type: str):
            for ev in ev_list or []:
                try:
                    ev_time = ev.get("time")
                    if ev_time is None:
                        continue
                    ev_time_val = float(ev_time)
                except Exception:
                    continue
                if ev_time_val < time_start or ev_time_val > time_end:
                    continue
                row: Dict[str, Any] = {
                    "match_id": match_id,
                    "radiant_or_dire": side,
                    "team_id": team_meta.get(side, {}).get("team_id", ""),
                    "team_name": team_meta.get(side, {}).get("team_name", ""),
                    "ward_type": ward_type,
                    "account_id": account_id,
                    "personal_name": personal_name,
                    "player_slot": p_slot,
                }
                for k, v in ev.items():
                    row[k] = v
                rows.append(row)

        process_log_list(p.get("obs_log", []), "obs")
        process_log_list(p.get("sen_log", []), "sen")
        # also include left logs if present (so canonical CSV contains both)
        process_log_list(p.get("obs_left_log", []) or p.get("obs_left_log", []), "obs_left")
        process_log_list(p.get("sen_left_log", []) or p.get("sen_left_log", []), "sen_left")

    return rows

def infer_match_id(match_data: dict, filename: Optional[str] = None) -> Optional[int]:
    mid = match_data.get("match_id") or match_data.get("matchid") or match_data.get("matchId")
    try:
        if mid is not None:
            return int(mid)
    except Exception:
        pass
    if filename:
        fn = Path(filename).name
        m = re.search(r"(\d{6,})", fn)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def save_csv_canonical_and_team_copies(match_id: Optional[int], rows: List[Dict[str, Any]], base_out: Path,
                                       group_by_team: bool, match_data: dict, verbose: bool = False) -> Tuple[Path, List[Path]]:
    """
    Save canonical CSV to base_out/all/ and, if requested, copy into base_out/teams/<team>/.
    """
    all_dir = base_out / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    if match_id is not None:
        csv_name = f"match_{match_id}_wards.csv"
    else:
        csv_name = f"{int(time.time())}_match_unknown_wards.csv"
    canonical = all_dir / csv_name

    # build columns
    default_fields = [
        "match_id", "radiant_or_dire", "team_id", "team_name", "ward_type",
        "account_id", "personal_name", "player_slot", "time", "type",
        "x", "y", "z", "entityleft", "ehandle", "key",
    ]
    cols = list(default_fields)
    extra = set()
    for r in rows:
        extra.update(r.keys())
    for k in sorted(extra):
        if k not in cols:
            cols.append(k)

    # write canonical CSV
    try:
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
        print(f"[extract] Wrote canonical CSV: {canonical}")

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
                        print(f"[extract] Created team dir: {team_dir}")
                team_csv = team_dir / canonical.name
                shutil.copy2(canonical, team_csv)
                team_paths.append(team_csv)
                if verbose:
                    print(f"[extract] Wrote team CSV: {team_csv}")
            except Exception as e:
                print(f"[extract] Failed to write/copy team CSV {team_dir}: {e}")

    return canonical, team_paths

# ----------------- placement collection (CSV-driven, lifetime from CSV) -----------------
def collect_placements_from_rows(rows: List[Dict[str, Any]], team_side: Optional[str], team_display: Optional[str],
                                ward_types: Optional[List[str]], verbose: bool=False) -> List[Dict[str, Any]]:
    """
    Simplified: treat input rows as canonical CSV rows (already may contain 'lifetime').
    - Only consider placement rows (type contains '_log' but not 'left').
    - Prefer explicit lifetime-like columns via _extract_lifetime_from_row.
    - Do NOT attempt to match removal events to compute lifetime.
    """
    pts: List[Dict[str, Any]] = []
    def _get_matchid(r):
        return str(r.get("match_id") or r.get("match") or "")

    def _get_time(r):
        try:
            if r.get("time") not in (None, ""):
                return float(r.get("time"))
        except Exception:
            pass
        return None

    seen = set()
    for r in rows:
        # side/team filter
        rside = str(r.get("radiant_or_dire") or r.get("team") or "").lower()
        if team_side in ("radiant", "dire") and rside and rside != team_side:
            continue
        tn = str(r.get("team_name") or r.get("team") or "")
        if tn and team_display and tn.strip().lower() != team_display.strip().lower():
            continue

        row_type = (str(r.get("type") or r.get("event") or "")).lower()
        # only placements (obs_log/sen_log), skip left logs
        if "_log" not in row_type or "left" in row_type:
            continue

        tval = _get_time(r)
        typ = (str(r.get("ward_type") or r.get("type") or "")).lower()
        wtype = "obs" if "obs" in typ else ("sen" if "sen" in typ else "unknown")
        if ward_types and wtype not in ward_types:
            continue

        xy = parse_xy_from_row(r)
        if not xy:
            continue
        x, y = xy
        matchid = _get_matchid(r)
        ehandle = r.get("ehandle") or r.get("entityleft") or ""
        key = (r.get("key") or r.get("pos_key") or "").strip()

        # unique id (prefer ehandle, then key, then cell)
        if ehandle not in (None, "", False):
            uniq = ("ehandle", matchid, str(ehandle))
        elif key:
            uniq = ("key", matchid, key)
        else:
            try:
                gx = int(round(float(x))); gy = int(round(float(y)))
            except Exception:
                gx = int(math.floor(float(x))) if x is not None else 0
                gy = int(math.floor(float(y))) if y is not None else 0
            uniq = ("cell", matchid, wtype, gx, gy)

        if uniq in seen:
            continue
        seen.add(uniq)

        # Prefer explicit lifetime from CSV row; DO NOT compute from removed_time here.
        lifetime_raw = _extract_lifetime_from_row(r)

        # Also accept duration-like fields as a fallback (redundant)
        duration = _extract_numeric_from_row_keys(r, ["duration", "life", "lifetime", "time_alive", "entity_life"])
        if duration is not None:
            lifetime_raw = float(duration)

        # map coordinate transform consistent with plotting (subtract 64)
        try:
            px = float(x) - 64.0
            py = float(y) - 64.0
        except Exception:
            continue

        pts.append({
            "match": matchid,
            "x": px,
            "y": py,
            "time_raw": tval,
            "removed_time_raw": r.get("removed_time") or r.get("removed") or "",
            "lifetime_raw": lifetime_raw,
            "ward_type": wtype,
            "side": rside,
            "uniq_id": uniq
        })

    if verbose:
        total = len(pts)
        with_life = sum(1 for p in pts if p.get("lifetime_raw") is not None)
        print(f"[diag] collected placements={total}, with_lifetime={with_life}")
    return pts

# ----------------- time normalization -----------------
def normalize_times_guess(points: List[Dict[str, Any]], verbose: bool=False):
    """
    Normalize time fields from minutes->seconds when appropriate.
    Converts:
      - time_raw -> time_s
      - lifetime_raw -> lifetime_s (if present)
    """
    times = [float(p["time_raw"]) for p in points if p.get("time_raw") is not None]
    if not times:
        for p in points:
            p["lifetime_s"] = float(p["lifetime_raw"]) if p.get("lifetime_raw") is not None else None
            p["time_s"] = float(p["time_raw"]) if p.get("time_raw") is not None else None
        return

    mx = max(times); med = sorted(times)[len(times)//2]
    if mx <= 200 and med >= 1.0:
        if verbose:
            print(f"[time-norm] converting {len(times)} times minutes->seconds (max={mx}, med={med})")
        for p in points:
            if p.get("time_raw") is not None:
                p["time_s"] = float(p["time_raw"]) * 60.0
            else:
                p["time_s"] = None
            if p.get("lifetime_raw") is not None:
                p["lifetime_s"] = float(p["lifetime_raw"]) * 60.0
            else:
                p["lifetime_s"] = None
    else:
        for p in points:
            p["time_s"] = float(p["time_raw"]) if p.get("time_raw") is not None else None
            p["lifetime_s"] = float(p["lifetime_raw"]) if p.get("lifetime_raw") is not None else None

# ----------------- plotting -----------------
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

def plot_points_on_minimap(img_path: Path, points: List[Dict[str, Any]], out_png: Path,
                           grid_size: int=256, flip_y: bool=True, point_size: int=28, alpha: float=0.9,
                           title: Optional[str]=None, scale_by_lifetime: bool=False,
                           lifetime_min_mul: float=0.6, lifetime_max_mul: float=2.4,
                           lifetime_auto_range: bool=False,
                           obs_max_lifetime: float=DEFAULT_OBS_MAX_LIFETIME,
                           sen_max_lifetime: float=DEFAULT_SEN_MAX_LIFETIME,
                           ward_clock: bool=False):
    """
    Plot points grouped by ward_type. If scale_by_lifetime True, scale s by lifetime_s.
    lifetime_auto_range: if True compute lif_min/lif_max from the points (legacy behavior).
                         if False (default) use configured max lifetimes (obs_max_lifetime / sen_max_lifetime)
                         and set lif_min = 0.0.
    ward_clock: if True, draw observer inner fill as a clock-like wedge proportional to lifetime fraction.
                When ward_clock is active, the scatter marker itself is drawn without fill to avoid
                drawing both a filled circle and the wedge. Also the clock inner radius is fixed
                (not scaled by lifetime) so the wedge angle represents lifetime fraction while the
                physical clock size remains uniform.
    """
    if Image is None or plt is None:
        print("Plot libs missing; cannot draw points:", out_png)
        return
    if not points:
        print("No points to plot for", out_png)
        return
    try:
        img = Image.open(img_path).convert("RGBA")
    except Exception as e:
        print(f"Failed to open map image {img_path}: {e}")
        return
    img_w, img_h = img.size
    dpi = 128
    plt.figure(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
    ax = plt.gca()
    ax.imshow(img, extent=[0, img_w, 0, img_h])

    # Determine lifetime range: either auto from data OR default maxima
    if lifetime_auto_range:
        lifetime_values = [p.get("lifetime_s") for p in points if p.get("lifetime_s") is not None]
        lif_min = min(lifetime_values) if lifetime_values else None
        lif_max = max(lifetime_values) if lifetime_values else None
    else:
        # use configured maxima per ward type; lif_min = 0.0
        lif_min = 0.0
        max_list = []
        for p in points:
            wt = p.get("ward_type") or "obs"
            if wt == "obs":
                if obs_max_lifetime and obs_max_lifetime > 0.0:
                    max_list.append(float(obs_max_lifetime))
            elif wt == "sen":
                if sen_max_lifetime and sen_max_lifetime > 0.0:
                    max_list.append(float(sen_max_lifetime))
            else:
                # fallback to obs max
                if obs_max_lifetime and obs_max_lifetime > 0.0:
                    max_list.append(float(obs_max_lifetime))
        lif_max = max(max_list) if max_list else None

    def size_for_point(p):
        base = point_size
        if not scale_by_lifetime or p.get("lifetime_s") is None or lif_min is None or lif_max is None:
            return base
        if lif_max <= lif_min:
            norm = 0.5
        else:
            norm = (p["lifetime_s"] - lif_min) / (lif_max - lif_min)
            norm = max(0.0, min(1.0, norm))
        mul = lifetime_min_mul + norm * (lifetime_max_mul - lifetime_min_mul)
        return max(1.0, base * mul)

    pts_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for p in points:
        wt = p.get("ward_type") or "unknown"
        pts_by_type.setdefault(wt, []).append(p)

    legend_handles = []
    for wt in ("obs", "sen", "unknown"):
        pts_list = pts_by_type.get(wt, [])
        if not pts_list:
            continue
        xs = []; ys = []; sizes = []
        # compute sizes, but if ward_clock is enabled for obs, use fixed point_size
        for p in pts_list:
            px, py = map_to_image_coords(p["x"], p["y"], grid_size, img_w, img_h, flip_y)
            xs.append(px); ys.append(py)
            if wt == "obs" and ward_clock:
                # keep clock physical size constant when using wedge representation
                sizes.append(point_size)
            else:
                sizes.append(size_for_point(p))
        xs = np.array(xs); ys = np.array(ys); sizes = np.array(sizes)
        style = WARD_STYLES.get(wt, WARD_STYLES["unknown"])

        # When ward_clock is enabled for observer wards, avoid drawing a filled scatter
        # marker under the wedge (that would create a filled circle + wedge). Draw the
        # scatter marker without face fill and without edge so only the wedge + outer ring
        # represent the ward. For other cases keep original styling.
        if wt == "obs" and ward_clock:
            facecolor_param = 'none'
            edgecolor_param = 'none'
            sc = ax.scatter(xs, ys, facecolors=facecolor_param, s=sizes, marker=style["marker"],
                            edgecolors=edgecolor_param, linewidths=0.6, alpha=alpha, zorder=6, label=wt)
        else:
            sc = ax.scatter(xs, ys, c=style["facecolor"], s=sizes, marker=style["marker"],
                            edgecolors=style["edgecolor"], linewidths=0.6, alpha=alpha, zorder=6, label=wt)

        legend_handles.append(sc)
        if wt == "obs":
            from matplotlib.patches import Circle, Wedge
            # Outer ring: fixed size in pixels (image coordinates). Keep it small so it does not
            # overwhelm the map. This is constant for all observer wards.
            outer_ring_radius_px = 10.0   # <-- tunable: try 8..14
            outer_ring_linewidth = 1.0

            # Inner fill: radius is proportional to normalized lifetime.
            # If lifetime is None -> no fill (show only outer ring).
            # If lif_min == lif_max -> treat as fully-filled.
            inner_max_mul = 0.9  # inner circle max radius = outer_ring_radius_px * inner_max_mul

            for xi, yi, sz, lt in zip(xs, ys, sizes, [p.get("lifetime_s") for p in pts_list]):
                # edge color for ring: keep it red to mark observer
                ring_edge = "red"

                # Compute fill fraction from lifetime:
                if lt is None or lif_min is None or lif_max is None:
                    frac = 0.0
                else:
                    if lif_max <= lif_min:
                        frac = 1.0
                    else:
                        frac = (float(lt) - float(lif_min)) / (float(lif_max) - float(lif_min))
                        frac = max(0.0, min(1.0, frac))

                # Outer ring (constant radius)
                circ = Circle((xi, yi), radius=outer_ring_radius_px, facecolor="none",
                            edgecolor=ring_edge, linewidth=outer_ring_linewidth, alpha=0.9, zorder=7)
                ax.add_patch(circ)

                # Inner filled circle or clock-like wedge
                if frac > 0.0:
                    # For clock-style wedge we want a fixed inner radius (so the wedge always
                    # fills the same physical circle area) and only the wedge angle varies.
                    if ward_clock:
                        inner_radius = outer_ring_radius_px * inner_max_mul
                    else:
                        # Non-clock: radius proportional to lifetime fraction (keeps previous behavior)
                        inner_radius = outer_ring_radius_px * inner_max_mul * frac
                        # make sure inner is at least a few px so very small lives are visible
                        inner_radius = max(2.0, inner_radius)

                    if ward_clock:
                        # Draw a wedge (clock-like) from top (90 deg) clockwise proportionally to frac
                        # Matplotlib wedge draws counter-clockwise from theta1 to theta2, so to get
                        # clockwise progress we set theta1 = 90 and theta2 = 90 - frac*360
                        theta1 = 90.0
                        theta2 = 90.0 - (frac * 360.0)
                        # filled portion: opaque yellow (use facecolor, full opacity)
                        wedge = Wedge((xi, yi), inner_radius, theta2, theta1,
                                      facecolor=style.get("facecolor", "yellow"),
                                      edgecolor="k", linewidth=0.6, alpha=1.0, zorder=8)
                        ax.add_patch(wedge)
                        # Do NOT draw a background inner circle - unfilled part remains transparent.
                    else:
                        inner = Circle((xi, yi), radius=inner_radius, facecolor=style.get("facecolor", "yellow"),
                                    edgecolor="k", linewidth=0.6, alpha=0.95, zorder=8)
                        ax.add_patch(inner)
                else:
                    # tiny dot for unknown/very short-lived
                    # If using ward_clock and frac==0, leave inner transparent (no tiny dot)
                    if ward_clock:
                        # skip inner marker for fully-empty (transparent) clock
                        pass
                    else:
                        tiny_r = 2.5
                        tiny = Circle((xi, yi), radius=tiny_r, facecolor=style.get("facecolor", "yellow"),
                                    edgecolor="k", linewidth=0.6, alpha=0.95, zorder=8)
                        ax.add_patch(tiny)

    label_map = {"obs": "Observer ward (obs)", "sen": "Sentry ward (sen)", "unknown": "unknown"}
    if legend_handles:
        labels = [label_map.get(h.get_label(), h.get_label()) for h in legend_handles]
        ax.legend(legend_handles, labels, loc="lower left", fontsize="small", framealpha=0.9)

    if scale_by_lifetime and (lif_min is not None and lif_max is not None):
        lif_min_s = lif_min if lif_min is not None else 0.0
        lif_max_s = lif_max if lif_max is not None else 0.0
        info_text = f"lifetime range: {lif_min_s:.1f}s - {lif_max_s:.1f}s (scaled x{lifetime_min_mul:.2f}-{lifetime_max_mul:.2f})"
        ax.text(0.99, 0.01, info_text, ha="right", va="bottom", transform=ax.transAxes, fontsize=8, color="black", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    if title:
        ax.set_title(title)
    ax.set_xlim(0, img_w); ax.set_ylim(0, img_h)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    print("Wrote points map:", out_png)

def plot_scatter_points(points: List[Dict[str, Any]], out_png: Path, grid_size: int=256, flip_y: bool=True,
                        scale_by_lifetime: bool=False, lifetime_min_mul: float=0.6, lifetime_max_mul: float=2.4,
                        lifetime_auto_range: bool=False,
                        obs_max_lifetime: float=DEFAULT_OBS_MAX_LIFETIME,
                        sen_max_lifetime: float=DEFAULT_SEN_MAX_LIFETIME):
    if plt is None:
        print("Plot libs missing; cannot plot scatter:", out_png); return
    max_idx = grid_size - 1
    fig, ax = plt.subplots(figsize=(8, 8))
    legend_handles = []

    # Determine lifetime range (same policy as minimap)
    if lifetime_auto_range:
        lifetime_values = [p.get("lifetime_s") for p in points if p.get("lifetime_s") is not None]
        lif_min = min(lifetime_values) if lifetime_values else None
        lif_max = max(lifetime_values) if lifetime_values else None
    else:
        lif_min = 0.0
        max_list = []
        for p in points:
            wt = p.get("ward_type") or "obs"
            if wt == "obs":
                if obs_max_lifetime and obs_max_lifetime > 0.0:
                    max_list.append(float(obs_max_lifetime))
            elif wt == "sen":
                if sen_max_lifetime and sen_max_lifetime > 0.0:
                    max_list.append(float(sen_max_lifetime))
            else:
                if obs_max_lifetime and obs_max_lifetime > 0.0:
                    max_list.append(float(obs_max_lifetime))
        lif_max = max(max_list) if max_list else None

    def size_for_point_local(p, base=20):
        if not scale_by_lifetime or p.get("lifetime_s") is None or lif_min is None or lif_max is None:
            return base
        if lif_max <= lif_min:
            norm = 0.5
        else:
            norm = (p["lifetime_s"] - lif_min) / (lif_max - lif_min)
            norm = max(0.0, min(1.0, norm))
        mul = lifetime_min_mul + norm * (lifetime_max_mul - lifetime_min_mul)
        return max(1.0, base * mul)

    for wt in ("obs", "sen", "unknown"):
        xs = []; ys = []; sizes = []
        for p in points:
            if (p.get("ward_type") or "unknown") != wt:
                continue
            gx = float(p["x"]); gy = float(p["y"])
            gx = max(0.0, min(gx, float(max_idx))); gy = max(0.0, min(gy, float(max_idx)))
            xs.append(gx); ys.append(gy)
            sizes.append(size_for_point_local(p, base=20))
        if not xs:
            continue
        xs = np.array(xs); ys = np.array(ys); sizes = np.array(sizes)
        style = WARD_STYLES.get(wt, WARD_STYLES["unknown"])
        sc = ax.scatter(xs, ys, c=style["facecolor"], s=sizes, marker=style["marker"], edgecolors=style["edgecolor"], alpha=0.7, label=wt)
        legend_handles.append(sc)
    if legend_handles:
        label_map = {"obs": "Observer ward (obs)", "sen": "Sentry ward (sen)", "unknown": "unknown"}
        labels = [label_map.get(h.get_label(), h.get_label()) for h in legend_handles]
        ax.legend(legend_handles, labels, fontsize="small", framealpha=0.9)
    ax.set_xlim(0, grid_size); ax.set_ylim(0, grid_size)
    if flip_y:
        ax.invert_yaxis()
    ax.set_title(out_png.stem)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150); plt.close(fig)
    print("Wrote scatter:", out_png)

# ----------------- filesystem helpers used by extraction -----------------
def find_match_json_files(matches_dir: Path) -> List[Path]:
    files: List[Path] = []
    if not matches_dir.exists():
        return files
    for p in matches_dir.rglob("*.json"):
        files.append(p)
    return sorted(files)

def process_match_file_for_extraction(match_file: Path, out_dir: Path, time_start: int, time_end: int,
                                      skip_existing: bool, group_by_team: bool, verbose: bool = False) -> Tuple[int, bool, List[Path]]:
    try:
        with match_file.open("r", encoding="utf-8") as fh:
            match_data = json.load(fh)
    except Exception as e:
        if verbose:
            print(f"[extract] Failed to load JSON {match_file}: {e}")
        return (0, False, [])
    match_id = infer_match_id(match_data, filename=str(match_file))
    canonical_path = out_dir / "all" / (f"match_{match_id}_wards.csv" if match_id is not None else f"{match_file.stem}_wards.csv")
    if skip_existing and canonical_path.exists():
        created = []
        if group_by_team:
            try:
                _, created = create_team_copies_from_existing_csv(out_dir, match_id, match_data, group_by_team, verbose)
            except Exception:
                created = []
        paths = [canonical_path] + created
        if verbose:
            print(f"[extract] skipped existing -> {paths}")
        return (match_id or 0, False, paths)
    rows = collect_ward_events(match_id, match_data, time_start, time_end, {})
    try:
        canonical, team_paths = save_csv_canonical_and_team_copies(match_id, rows, out_dir, group_by_team, match_data, verbose)
        paths = [canonical] + team_paths
        return (match_id or 0, True, paths)
    except Exception as e:
        if verbose:
            print(f"[extract] Failed to save CSV for {match_file}: {e}")
        return (match_id or 0, False, [])

# ----------------- main flow -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--team-name", required=True)
    p.add_argument("--mid-account", required=True)
    p.add_argument("--matches-dir", type=Path, default=DEFAULT_MATCHES_DIR)
    p.add_argument("--wards-dir", type=Path, default=DEFAULT_WARDS_DIR)
    p.add_argument("--hero-map", type=Path, default=DEFAULT_HERO_MAP)
    p.add_argument("--map", dest="map_image", type=Path, help="Minimap image (required to overlay points)")
    p.add_argument("--out", dest="out_dir", type=Path, default=Path("results/mid_hero_wards"))
    p.add_argument("--start", type=float, default=0.0, help="Start minute (will be converted to seconds)")
    p.add_argument("--end", type=float, default=10.0, help="End minute (will be converted to seconds)")
    p.add_argument("--ward-types", type=str, default="obs", help="Comma-separated: obs,sen (empty for all)")
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--flip-y", action="store_true")
    p.add_argument("--min-matches", type=int, default=1)
    p.add_argument("--side", type=str, choices=["radiant", "dire"], help="Optional: only produce outputs for this side")
    p.add_argument("--verbose", action="store_true")

    # extraction options
    p.add_argument("--extract-wards", action="store_true", help="Extract obs/sen logs from match JSONs into wards CSVs before plotting")
    p.add_argument("--group-wards-by-team", action="store_true", help="When extracting, also copy canonical CSV into per-team folders")
    p.add_argument("--skip-existing-wards", action="store_true", help="Skip generating wards CSV if canonical CSV exists")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers for extraction")
    p.add_argument("--scale-by-lifetime", action="store_true", help="Scale marker sizes by ward lifetime (requires lifetime data)")
    p.add_argument("--lifetime-min-mul", type=float, default=0.6, help="Minimum multiplier for lifetime scaling")
    p.add_argument("--lifetime-max-mul", type=float, default=2.4, help="Maximum multiplier for lifetime scaling")

    # NEW: lifetime defaults vs auto-range
    p.add_argument("--lifetime-auto-range", action="store_true",
                   help="If set, compute lifetime min/max from data (legacy). If not set (default), use provided max-lifetime defaults.")
    p.add_argument("--obs-max-lifetime", type=float, default=DEFAULT_OBS_MAX_LIFETIME,
                   help=f"Default max lifetime (seconds) for observer wards when not using --lifetime-auto-range (default {DEFAULT_OBS_MAX_LIFETIME}s).")
    p.add_argument("--sen-max-lifetime", type=float, default=DEFAULT_SEN_MAX_LIFETIME,
                   help=f"Default max lifetime (seconds) for sentry wards when not using --lifetime-auto-range (default {DEFAULT_SEN_MAX_LIFETIME}s).")

    # NEW: clock-like wedge option for observer inner fill
    p.add_argument("--ward-clock", action="store_true", help="Draw clock-style wedge for observer inner fill (clock-like remaining lifetime arc)")

    args = p.parse_args()

    # enable Chinese font early (before any plt.figure)
    chosen_font = None
    try:
        chosen_font = set_chinese_font_prefer()
    except Exception:
        chosen_font = None
    if args.verbose:
        print("[font] chosen font:", chosen_font)

    team_display = args.team_name
    team_safe = safe_name(team_display)
    try:
        mid_acc = int(args.mid_account)
    except Exception:
        mid_acc = args.mid_account

    matches_dir = args.matches_dir
    wards_dir = args.wards_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # debug log file
    debug_log = out_dir / "run_debug.log"
    def dbg(msg: str):
        if args.verbose:
            print(msg)
        try:
            with debug_log.open("a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass

    # If requested, extract wards CSVs from matches
    if args.extract_wards:
        dbg(f"[extract] Starting extraction from {matches_dir} -> {wards_dir}")
        matches_files = find_match_json_files(matches_dir)
        if not matches_files:
            dbg(f"[extract] No match JSON files found under {matches_dir}")
        else:
            workers = max(1, int(args.workers))
            generated = 0; skipped = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_map = {ex.submit(process_match_file_for_extraction, mf, wards_dir, DEFAULT_TIME_START, DEFAULT_TIME_END, args.skip_existing_wards, args.group_wards_by_team, args.verbose): mf for mf in matches_files}
                for fut in as_completed(future_map):
                    mf = future_map[fut]
                    try:
                        mid, did_gen, paths = fut.result()
                    except Exception as e:
                        dbg(f"[extract] worker failed for {mf}: {e}")
                        continue
                    if did_gen:
                        generated += 1
                        dbg(f"[extract] generated for {mid}: {len(paths)} path(s)")
                    else:
                        skipped += 1
            dbg(f"[extract] done. generated={generated}, skipped_existing={skipped}")

    # load hero map
    hero_map_full = load_hero_map_full(args.hero_map) if args.hero_map else {}
    if args.verbose:
        dbg(f"[debug] loaded hero_map entries: {len(hero_map_full)}")
        sample_keys = list(hero_map_full.keys())[:8]
        for k in sample_keys:
            v = hero_map_full.get(k, {})
            dbg(f"  id={k} -> en='{v.get('hero_name')}', cn='{v.get('hero_name_cn')}'")

    start_s = float(args.start) * 60.0 if args.start is not None else None
    end_s = float(args.end) * 60.0 if args.end is not None else None
    ward_types = [w.strip().lower() for w in args.ward_types.split(",") if w.strip()] if args.ward_types else []

    # Find matches and build hero->matches mapping (same as previous logic)
    hero_to_matches: Dict[str, Dict[str, List[str]]] = {}
    match_files = list(matches_dir.rglob("*.json"))
    if args.verbose:
        dbg(f"[scan] scanning {len(match_files)} match json files for account {mid_acc} and team {team_display}")
    for mf in match_files:
        mj = read_json_optional(mf)
        if not mj:
            continue
        mid = str(mj.get("match_id") or "")
        team_side = None
        try:
            team_side = (
                "radiant" if (str(team_display).strip().lower() == str(mj.get("radiant_name") or "").strip().lower()) else
                ("dire" if (str(team_display).strip().lower() == str(mj.get("dire_name") or "").strip().lower()) else None)
            )
            if team_side is None:
                # nested
                if isinstance(mj.get("radiant_team"), dict) and safe_name(team_display) == safe_name(mj.get("radiant_team").get("name", "")):
                    team_side = "radiant"
                if isinstance(mj.get("dire_team"), dict) and safe_name(team_display) == safe_name(mj.get("dire_team").get("name", "")):
                    team_side = "dire"
        except Exception:
            team_side = None

        players = mj.get("players") or []
        for pl in players:
            acc = pl.get("account_id")
            try:
                acc_int = int(acc) if acc not in (None, "") else None
            except Exception:
                acc_int = acc
            if acc_int == mid_acc or str(acc) == str(mid_acc):
                slot = pl.get("player_slot")
                try:
                    s = int(slot); pside = "radiant" if s < 128 else "dire"
                except Exception:
                    pside = "unknown"
                if team_side and pside and pside != team_side:
                    # conflict, skip
                    continue
                hero = pl.get("hero_id") or pl.get("hero")
                if hero is None:
                    continue
                eff_side = team_side if team_side else (pside if pside else "unknown")
                hero_to_matches.setdefault(str(hero), {}).setdefault(eff_side, []).append(mid)
                if args.verbose:
                    dbg(f"  match {mid}: account {mid_acc} hero {hero} side {eff_side}")
                break

    if not hero_to_matches:
        print(f"No matches found for account {mid_acc} on team {team_display}")
        return

    # For each hero+side collect ward points and plot
    for hero_id, side_map in sorted(hero_to_matches.items(), key=lambda kv: -sum(len(v) for v in kv[1].values())):
        info = hero_map_full.get(str(hero_id), {})
        hero_name_en = info.get("hero_name") or str(hero_id)
        hero_name_cn = info.get("hero_name_cn") or hero_name_en
        safe_en = safe_name(hero_name_en)
        safe_cn = safe_name(hero_name_cn)
        hero_label = f"{hero_id}_{safe_en}_{safe_cn}"

        for side_key, matches in sorted(side_map.items(), key=lambda kv: -len(kv[1])):
            if args.side and side_key != args.side:
                continue
            if len(matches) < args.min_matches:
                if args.verbose:
                    dbg(f"Skipping hero {hero_id} {side_key}: only {len(matches)} matches")
                continue
            if args.verbose:
                dbg(f"Processing hero {hero_id} ({hero_name_en}/{hero_name_cn}) side={side_key} matches={len(matches)}")
            collected: List[Dict[str, Any]] = []
            for mid in matches:
                team_csv = wards_dir / "teams" / team_safe / f"match_{mid}_wards.csv"
                fallback = wards_dir / "all" / f"match_{mid}_wards.csv"
                rows: List[Dict[str, Any]] = []
                if team_csv.exists():
                    rows = read_csv_rows(team_csv)
                    dbg(f" Using team CSV: {team_csv}")
                elif fallback.exists():
                    rows = read_csv_rows(fallback)
                    dbg(f" Using fallback CSV: {fallback}")
                else:
                    dbg(f"  No ward CSV for match {mid}")
                    continue
                pts = collect_placements_from_rows(rows, side_key if side_key in ("radiant", "dire") else None, team_display, ward_types if ward_types else None, verbose=args.verbose)
                normalize_times_guess(pts, verbose=args.verbose)
                in_window: List[Dict[str, Any]] = []
                for p in pts:
                    t = p.get("time_s")
                    if t is None:
                        if start_s is None or start_s <= 0.0:
                            in_window.append(p)
                        continue
                    if start_s is not None and t < start_s:
                        continue
                    if end_s is not None and t > end_s:
                        continue
                    in_window.append(p)
                if args.verbose:
                    dbg(f"  match {mid}: collected {len(pts)} placements, {len(in_window)} in window")
                collected.extend(in_window)

            # dedupe across matches
            seen = set(); unique: List[Dict[str, Any]] = []
            for p in collected:
                uniq = p.get("uniq_id") if p.get("uniq_id") is not None else (p.get("match"), round(p.get("x", 0)), round(p.get("y", 0)))
                if uniq in seen:
                    continue
                seen.add(uniq); unique.append(p)

            team_out = out_dir / team_safe
            team_out.mkdir(parents=True, exist_ok=True)
            match_list = team_out / f"hero_{hero_label}_{side_key}_matches.txt"
            with match_list.open("w", encoding="utf-8") as fh:
                for m in matches:
                    fh.write(m + "\n")
            csv_out = team_out / f"hero_{hero_label}_{side_key}_wards_n{len(unique)}.csv"
            with csv_out.open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh); w.writerow(["match", "x", "y", "time_s", "lifetime_s", "ward_type", "side"])
                for p in unique:
                    w.writerow([p.get("match"), p.get("x"), p.get("y"), p.get("time_s"), p.get("lifetime_s"), p.get("ward_type"), p.get("side")])
            print(f"Wrote {csv_out} ({len(unique)} points) and {match_list}")

            # plotting
            if args.map_image and unique:
                out_png = team_out / f"hero_{hero_label}_{side_key}_wards.png"
                title = f"{hero_name_cn} ({hero_name_en}, {hero_id}) — {team_display} as {side_key} — {len(matches)} matches"
                plot_points_on_minimap(
                    args.map_image, unique, out_png, grid_size=args.grid_size, flip_y=args.flip_y, point_size=28,
                    title=title, scale_by_lifetime=args.scale_by_lifetime,
                    lifetime_min_mul=args.lifetime_min_mul, lifetime_max_mul=args.lifetime_max_mul,
                    lifetime_auto_range=args.lifetime_auto_range,
                    obs_max_lifetime=args.obs_max_lifetime,
                    sen_max_lifetime=args.sen_max_lifetime,
                    ward_clock=args.ward_clock
                )
            elif unique:
                out_png = team_out / f"hero_{hero_label}_{side_key}_scatter.png"
                plot_scatter_points(unique, out_png, grid_size=args.grid_size, flip_y=args.flip_y,
                                    scale_by_lifetime=args.scale_by_lifetime,
                                    lifetime_min_mul=args.lifetime_min_mul, lifetime_max_mul=args.lifetime_max_mul,
                                    lifetime_auto_range=args.lifetime_auto_range,
                                    obs_max_lifetime=args.obs_max_lifetime,
                                    sen_max_lifetime=args.sen_max_lifetime)
            else:
                print(f"No points to plot for hero {hero_id} side={side_key}")

    print("Done.")

if __name__ == "__main__":
    main()