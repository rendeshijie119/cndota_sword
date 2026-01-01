from __future__ import annotations

import os
import re
from typing import Optional, List, Tuple, Dict, Union
from pathlib import Path
from datetime import datetime, timezone, date

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import json

# 正确推断 sword 根目录 -> 默认数据根：sword/data
SWORD_ROOT = Path(__file__).resolve().parents[2]      # .../src/sword
DATA_ROOT = SWORD_ROOT / "data"
DEFAULT_WARDS_DIR = DATA_ROOT / "obs_logs"            # .../src/sword/data/obs_logs
DEFAULT_MATCHES_DIR = DATA_ROOT / "matches"           # .../src/sword/data/matches

# 新：配置目录（非下载数据，手工维护的配置文件）
CONFIG_ROOT = SWORD_ROOT / "config"
# 支持环境覆盖：CAMP_BOXES_FILE=/absolute/path/to/camp_boxes_xxx.json
_env_camp = os.environ.get("CAMP_BOXES_FILE")
CAMP_BOXES_FILE = Path(_env_camp).expanduser().resolve() if _env_camp else (CONFIG_ROOT / "camp_boxes_256.json")

# v9 坐标策略：GRID=128 且 x/y - 64
GRID_SIZE = 128
MAX_MAP_WIDTH = 1800

# 默认生命周期（用于过滤与尺寸/填充比例）
DEFAULT_OBS_MAX_LIFETIME = 360.0   # 观察守卫默认 6 分钟
DEFAULT_SEN_MAX_LIFETIME = 420.0   # 真眼默认 7 分钟

ASSET_DIRS = [
    Path(__file__).resolve().parent.parent / "assets",   # frontend/assets
    Path(__file__).resolve().parent / "assets",          # pages/assets
    Path.cwd() / "frontend" / "assets",
    Path.cwd() / "assets",
]
PREFERRED_NAMES = ("dota2_map.png", "dota2_map.jpg", "map.png", "map.jpg")
GLOB_PATTERNS = ("dota2_map.*", "map.*", "*.png", "*.jpg", "*.jpeg")


def find_default_map() -> Optional[Path]:
    env = os.environ.get("WARD_MAP_IMAGE")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    for d in ASSET_DIRS:
        if not d.exists():
            continue
        for fn in PREFERRED_NAMES:
            p = d / fn
            if p.exists():
                return p
        for pat in GLOB_PATTERNS:
            for p in d.glob(pat):
                if p.is_file():
                    return p
    return None


def load_map_resized(img_path: Path) -> Tuple[Image.Image, Tuple[int, int]]:
    img = Image.open(img_path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    w, h = img.size
    if w > MAX_MAP_WIDTH:
        new_w = MAX_MAP_WIDTH
        new_h = int(h * (new_w / w))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = new_w, new_h
    return img, (w, h)


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ ." else "_" for c in (s or "").strip())


def discover_team_csvs(team_name: str) -> List[Path]:
    """
    仅从 sword/data/obs_logs/teams/<team_safe>/ 读取 CSV。
    文件名匹配优先级：match_*_wards.csv -> *_wards.csv -> *.csv
    """
    team_safe = safe_name(team_name)
    team_dir = DEFAULT_WARDS_DIR / "teams" / team_safe
    patterns = ["match_*_wards.csv", "*_wards.csv", "*.csv"]
    found: List[Path] = []
    if team_dir.exists():
        for pat in patterns:
            found.extend(sorted(team_dir.glob(pat)))
    uniq, seen = [], set()
    for p in found:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq


def extract_match_id_from_name(name: str) -> str:
    m = re.search(r"(\d{6,})", name)
    return m.group(1) if m else name


@st.cache_data(show_spinner=False)
def load_wards_csv(team_name: str) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    discovered = discover_team_csvs(team_name)
    if not discovered:
        return pd.DataFrame(), [], [], []

    dfs, errors = [], []
    for f in discovered:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            df["__src__"] = f.name
            if "match_id" not in df.columns:
                df["match_id"] = extract_match_id_from_name(f.name)
            dfs.append(df)
        except Exception as e:
            errors.append(f"{f}: {e}")

    match_ids = sorted({extract_match_id_from_name(p.name) for p in discovered})
    if not dfs:
        return pd.DataFrame(), [str(p) for p in discovered], errors, match_ids

    data = pd.concat(dfs, ignore_index=True)

    # 仅保留 obs_log/sen_log（不含 *_left）
    tcol = data.get("type")
    if tcol is not None:
        mask = tcol.astype(str).str.contains("_log", case=False) & (~tcol.astype(str).str.contains("left", case=False))
        data = data[mask]

    # 统一 ward_type
    data["ward_type"] = (
        data.get("ward_type")
        .fillna(data.get("type"))
        .astype(str)
        .str.lower()
        .map(lambda s: "obs" if "obs" in s else ("sen" if "sen" in s else "unknown"))
    )

    # 坐标（v9：x/y - 64 -> 0..128）
    to_num = lambda col: pd.to_numeric(data.get(col), errors="coerce")
    x = to_num("x"); y = to_num("y")
    data["x_grid"] = x - 64.0
    data["y_grid"] = y - 64.0

    # 时间与生命周期（秒）
    data["time_s"] = pd.to_numeric(data.get("time"), errors="coerce")
    lifetime_cols = ["lifetime_s", "lifetime", "duration", "life", "time_alive", "entity_life"]
    lif = None
    for c in lifetime_cols:
        if c in data.columns:
            lif = pd.to_numeric(data.get(c), errors="coerce"); break
    data["lifetime_s"] = lif

    return data, [str(p) for p in discovered], errors, match_ids


def _abbr_from_name(name: str) -> str:
    """根据队伍名称生成简写（优先取大写字母，否则取每个词首字母）"""
    if not name:
        return ""
    letters = "".join([c for c in name if c.isupper()])
    if len(letters) >= 2:
        return letters[:6]
    tokens = re.findall(r"[A-Za-z]+", name)
    if tokens:
        return "".join(t[0].upper() for t in tokens)[:6]
    return name[:6]


@st.cache_data(show_spinner=False)
def load_matches_metadata_for_ids(match_ids: List[str]) -> pd.DataFrame:
    """
    读取元数据，包含比赛开始时间、赛事名称和两队简称
    返回列：match_id,start_ts,start_dt,radiant,dire,radiant_tag,dire_tag,league_name,json_path
    """
    rows: List[Dict] = []
    for mid in match_ids:
        json_path = None
        try:
            candidates = list(DEFAULT_MATCHES_DIR.rglob(f"*{mid}*.json"))
            if candidates:
                json_path = candidates[0]
        except Exception:
            json_path = None

        start_ts = None
        radiant = ""; dire = ""; league_name = ""; rtag = ""; dtag = ""
        if json_path and Path(json_path).exists():
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    j = json.load(fh)
                start_ts = j.get("start_time")
                radiant = j.get("radiant_name") or ""
                dire = j.get("dire_name") or ""
                # 赛事名称
                lj = j.get("league") or {}
                if isinstance(lj, dict):
                    league_name = lj.get("name") or ""
                # 简称（优先 tag）
                rteam = j.get("radiant_team") or {}
                dteam = j.get("dire_team") or {}
                rtag = (rteam.get("tag") or "").strip()
                dtag = (dteam.get("tag") or "").strip()
                if not rtag:
                    rtag = _abbr_from_name(radiant)
                if not dtag:
                    dtag = _abbr_from_name(dire)
                # 兼容嵌套队名
                if not radiant and isinstance(rteam, dict):
                    radiant = rteam.get("name") or radiant
                if not dire and isinstance(dteam, dict):
                    dire = dteam.get("name") or dire
            except Exception:
                pass

        if start_ts:
            try:
                start_dt = datetime.fromtimestamp(int(start_ts), tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
            except Exception:
                start_dt = ""
        else:
            start_dt = ""

        rows.append({
            "match_id": str(mid),
            "start_ts": float(start_ts) if start_ts is not None else np.nan,
            "start_dt": start_dt,
            "radiant": radiant,
            "dire": dire,
            "radiant_tag": rtag,
            "dire_tag": dtag,
            "league_name": league_name,
            "json_path": str(json_path) if json_path else "",
        })

    # 使用 ascending=False 排序（兼容旧版 pandas）
    df_meta = pd.DataFrame(rows).sort_values(by=["start_ts"], ascending=False, na_position="last").reset_index(drop=True)
    return df_meta


def make_background_figure_image(img: Image.Image, w: int, h: int, width_px: int, flip_y: bool):
    fig = go.Figure()
    fig.add_trace(go.Image(z=np.array(img)))
    if flip_y: fig.update_yaxes(range=[h, 0], visible=False)
    else:      fig.update_yaxes(range=[0, h], visible=False)
    fig.update_xaxes(range=[0, w], visible=False)
    fig.update_layout(
        width=width_px, height=int(width_px * (h / w)),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0.2)")
    )
    x_scale = w / GRID_SIZE
    y_scale = h / GRID_SIZE
    return fig, x_scale, y_scale


def _fmt_ms(series: pd.Series) -> pd.Series:
    """
    将秒格式化为 'Xm Ys'（支持负数）。NaN -> '-'
    """
    def f(v):
        if pd.isna(v): return "-"
        try:
            x = float(v)
        except Exception:
            return "-"
        sign = "-" if x < 0 else ""
        x = abs(x)
        m = int(x // 60)
        s = int(round(x % 60))
        if s == 60:
            m += 1
            s = 0
        return f"{sign}{m}m {s:02d}s"
    return series.map(f)


def add_obs_with_ring(fig: go.Figure, df_obs: pd.DataFrame, x_scale: float, y_scale: float, flip_y: bool, h_px: int):
    if df_obs.empty:
        return
    x_px = df_obs["x_grid"] * x_scale
    y_px = df_obs["y_grid"] * y_scale
    if flip_y: y_px = (h_px - y_px)

    life = pd.to_numeric(df_obs.get("lifetime_s"), errors="coerce")
    frac = np.clip(life / DEFAULT_OBS_MAX_LIFETIME, 0.0, 1.0).fillna(0.0)

    time_str = _fmt_ms(df_obs.get("time_s"))
    life_str = _fmt_ms(df_obs.get("lifetime_s"))

    ring_size = 18.0
    fig.add_trace(go.Scatter(
        x=x_px, y=y_px, mode="markers", name="Observer (obs)",
        marker=dict(symbol="circle", size=ring_size, color="rgba(0,0,0,0)", line=dict(color="red", width=1.2)),
        opacity=0.95,
        hovertemplate="Match:%{customdata[0]}<br>x:%{x:.1f} y:%{y:.1f}<br>t:%{customdata[1]}<br>life:%{customdata[2]}",
        customdata=np.stack([df_obs.get("match_id", df_obs.get("__src__", "")), time_str, life_str], axis=-1),
        legendgroup="obs"
    ))

    inner_sizes = np.maximum(2.0, ring_size * 0.9 * frac.values)
    fig.add_trace(go.Scatter(
        x=x_px, y=y_px, mode="markers", name="obs life fill",
        marker=dict(symbol="circle", size=inner_sizes, color="yellow", line=dict(color="black", width=0.6)),
        opacity=0.95, hoverinfo="skip", showlegend=False, legendgroup="obs"
    ))


def add_sen_scaled(fig: go.Figure, df_sen: pd.DataFrame, x_scale: float, y_scale: float, flip_y: bool, h_px: int):
    if df_sen.empty:
        return
    x_px = df_sen["x_grid"] * x_scale
    y_px = df_sen["y_grid"] * y_scale
    if flip_y: y_px = (h_px - y_px)

    life = pd.to_numeric(df_sen.get("lifetime_s"), errors="coerce")
    frac = np.clip(life / DEFAULT_SEN_MAX_LIFETIME, 0.0, 1.0).fillna(1.0)

    time_str = _fmt_ms(df_sen.get("time_s"))
    life_str = _fmt_ms(df_sen.get("lifetime_s"))

    base_size = 16.0
    sizes = np.maximum(6.0, base_size * frac.values)

    fig.add_trace(go.Scatter(
        x=x_px, y=y_px, mode="markers", name="Sentry (sen)",
        marker=dict(symbol="square", size=sizes, color="cyan", line=dict(color="black", width=0.6)),
        opacity=0.95,
        hovertemplate="Match:%{customdata[0]}<br>x:%{x:.1f} y:%{y:.1f}<br>t:%{customdata[1]}<br>life:%{customdata[2]}",
        customdata=np.stack([df_sen.get("match_id", df_sen.get("__src__", "")), time_str, life_str], axis=-1),
        legendgroup="sen"
    ))


def add_heatmap_image(fig: go.Figure, df: pd.DataFrame, name: str,
                      x_scale: float, y_scale: float, flip_y: bool, h_px: int):
    x_px = df["x_grid"] * x_scale
    y_px = df["y_grid"] * y_scale
    if flip_y: y_px = (h_px - y_px)
    fig.add_trace(go.Histogram2d(
        x=x_px, y=y_px, nbinsx=32, nbinsy=32,
        colorscale="YlOrRd", showscale=True, name=name,
    ))


# ---------- Neutral camp boxes (blocking) ----------
def _detect_src_grid(boxes_list: List[Dict[str, Union[str, float, dict]]]) -> int:
    """
    简单检测源坐标的网格：若任何坐标 > 128 则视为 256，否则 128。
    """
    try:
        for it in boxes_list:
            b = it.get("box") or {}
            vals = [b.get("min_x"), b.get("max_x"), b.get("min_y"), b.get("max_y")]
            vals = [float(v) for v in vals if v is not None]
            if any(v > 128.0 for v in vals):
                return 256
    except Exception:
        pass
    return 128


def _scale_boxes_df(df: pd.DataFrame, src_grid: int, dst_grid: int = GRID_SIZE) -> pd.DataFrame:
    """
    将源网格坐标缩放到内部 GRID_SIZE（默认 128）。
    简单线性缩放：new = old * (dst_grid / src_grid)
    """
    if src_grid == dst_grid or df.empty:
        return df
    k = float(dst_grid) / float(src_grid)
    out = df.copy()
    for col in ("min_x", "max_x", "min_y", "max_y"):
        out[col] = out[col] * k
    return out


@st.cache_data(show_spinner=False)
def load_camp_boxes() -> Tuple[pd.DataFrame, int]:
    """
    读取野点刷怪区域矩形框。
    支持两种 JSON 格式：
      1) 对象：{"grid_size": 256, "boxes": [ {...} ]}
      2) 列表：[{...}, {...}] （自动检测坐标是否为 256/128）
    返回 (df, src_grid)
    df 坐标会转换到内部 GRID_SIZE（默认 128）。
    """
    if not CAMP_BOXES_FILE.exists():
        return pd.DataFrame(columns=["id","label","side","type","min_x","max_x","min_y","max_y"]), 0

    try:
        with open(CAMP_BOXES_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return pd.DataFrame(columns=["id","label","side","type","min_x","max_x","min_y","max_y"]), 0

    boxes_list: List[Dict] = []
    declared_grid = 128
    if isinstance(data, dict) and "boxes" in data:
        # Trust but verify: read declared grid, then override if values exceed 128
        try:
            declared_grid = int(data.get("grid_size") or 128)
        except Exception:
            declared_grid = 128
        boxes_list = data.get("boxes") or []
        # Auto-detect if any coordinate suggests 256-grid
        detected_grid = _detect_src_grid(boxes_list)  # returns 256 if any value > 128
        src_grid = detected_grid if detected_grid != declared_grid else declared_grid
    elif isinstance(data, list):
        boxes_list = data
        src_grid = _detect_src_grid(boxes_list)
    else:
        return pd.DataFrame(columns=["id","label","side","type","min_x","max_x","min_y","max_y"]), 0

    # Build dataframe
    rows = []
    for it in boxes_list:
        b = (it.get("box") or {})
        try:
            min_x = float(b.get("min_x"))
            max_x = float(b.get("max_x"))
            min_y = float(b.get("min_y"))
            max_y = float(b.get("max_y"))
        except Exception:
            continue
        if not (np.isfinite(min_x) and np.isfinite(max_x) and np.isfinite(min_y) and np.isfinite(max_y)):
            continue
        if max_x <= min_x or max_y <= min_y:
            continue
        rows.append({
            "id": it.get("id") or "",
            "label": it.get("label") or it.get("id") or "",
            "side": (it.get("side") or "").lower(),
            "type": (it.get("type") or "").lower(),
            "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y
        })
    df = pd.DataFrame(rows)

    # Scale to internal GRID_SIZE=128
    df_scaled = _scale_boxes_df(df, src_grid, GRID_SIZE)
    return df_scaled, src_grid


def _point_in_box(x: float, y: float, row: pd.Series) -> bool:
    return (row["min_x"] <= x <= row["max_x"]) and (row["min_y"] <= y <= row["max_y"])


def add_camp_boxes_overlay(fig: go.Figure, boxes: pd.DataFrame, x_scale: float, y_scale: float, flip_y: bool, h_px: int,
                           color="rgba(255,0,0,0.15)", line_color="red", width=2.0, show_labels=True):
    """
    在地图上绘制所有刷怪框（半透明矩形）。
    """
    for r in boxes.itertuples():
        x0 = r.min_x * x_scale
        x1 = r.max_x * x_scale
        y0 = r.min_y * y_scale
        y1 = r.max_y * y_scale
        if flip_y:
            y0, y1 = (h_px - y0), (h_px - y1)
        fig.add_shape(type="rect", x0=min(x0,x1), x1=max(x0,x1), y0=min(y0,y1), y1=max(y0,y1),
                      fillcolor=color, line=dict(color=line_color, width=width), layer="above")
        if show_labels:
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            fig.add_annotation(x=cx, y=cy, text=r.label, showarrow=False,
                               font=dict(size=10, color="red"), bgcolor="rgba(255,255,255,0.6)")


def compute_camp_blocks(wards: pd.DataFrame, boxes: pd.DataFrame,
                        minute_range: Tuple[float,float],
                        use_default_lifetime: bool,
                        consider_types: List[str],
                        side_choice: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算在给定时间范围内（按整分钟 tick）被守卫阻断的刷怪次数。
    返回： (camp_stats_df, blocking_points_df)
    camp_stats_df: columns=[id,label,side,type,blocked_ticks,total_ticks,ratio]
    blocking_points_df: 从 wards 中筛出在框内且覆盖至少一个 tick 的守卫生效点
    """
    if wards.empty or boxes.empty:
        return pd.DataFrame(columns=["id","label","side","type","blocked_ticks","total_ticks","ratio"]), pd.DataFrame()

    if consider_types:
        wards = wards[wards["ward_type"].isin(consider_types)].copy()

    if side_choice in ("radiant","dire"):
        boxes = boxes[boxes["side"] == side_choice].copy()

    start_s = max(0.0, float(minute_range[0]) * 60.0)
    end_s = max(start_s, float(minute_range[1]) * 60.0)
    first_tick = int(np.ceil(start_s / 60.0)) * 60
    last_tick = int(np.floor(end_s / 60.0)) * 60
    if last_tick < first_tick:
        ticks = np.array([], dtype=int)
    else:
        ticks = np.arange(first_tick, last_tick + 1, 60, dtype=int)
    total_ticks = len(ticks)

    wards = wards.copy()
    wards["lifetime_eff"] = pd.to_numeric(wards.get("lifetime_s"), errors="coerce")
    if use_default_lifetime:
        is_obs = (wards["ward_type"] == "obs")
        is_sen = (wards["ward_type"] == "sen")
        wards.loc[is_obs & wards["lifetime_eff"].isna(), "lifetime_eff"] = DEFAULT_OBS_MAX_LIFETIME
        wards.loc[is_sen & wards["lifetime_eff"].isna(), "lifetime_eff"] = DEFAULT_SEN_MAX_LIFETIME

    wards["time_eff"] = pd.to_numeric(wards.get("time_s"), errors="coerce")
    wards = wards.dropna(subset=["x_grid","y_grid","time_eff","lifetime_eff"])

    camp_counts: Dict[str,int] = {r.id: 0 for r in boxes.itertuples()}
    blocking_rows: List[Dict] = []

    for w in wards.itertuples():
        wx, wy = float(w.x_grid), float(w.y_grid)
        w_start = float(w.time_eff)
        w_end = w_start + float(w.lifetime_eff)
        if total_ticks == 0:
            continue
        covered_ticks = ticks[(ticks >= w_start) & (ticks < w_end)]
        if covered_ticks.size == 0:
            continue
        for b in boxes.itertuples():
            if _point_in_box(wx, wy, pd.Series({"min_x":b.min_x,"max_x":b.max_x,"min_y":b.min_y,"max_y":b.max_y})):
                camp_counts[b.id] += int(covered_ticks.size)
                blocking_rows.append({
                    "match_id": getattr(w, "match_id", ""),
                    "ward_type": getattr(w, "ward_type", ""),
                    "x_grid": wx, "y_grid": wy,
                    "time_s": w_start, "lifetime_s": getattr(w, "lifetime_s", np.nan),
                    "blocked_tick_count": int(covered_ticks.size),
                    "camp_id": b.id, "camp_label": b.label, "camp_side": b.side, "camp_type": b.type
                })

    stats_rows = []
    for b in boxes.itertuples():
        blocked = int(camp_counts.get(b.id, 0))
        ratio = (blocked / total_ticks) if total_ticks > 0 else 0.0
        stats_rows.append({"id": b.id, "label": b.label, "side": b.side, "type": b.type,
                           "blocked_ticks": blocked, "total_ticks": total_ticks, "ratio": ratio})
    stats_df = pd.DataFrame(stats_rows).sort_values(by=["blocked_ticks","label"], ascending=[False, True]).reset_index(drop=True)
    blocking_df = pd.DataFrame(blocking_rows)
    return stats_df, blocking_df


def add_blocking_points(fig: go.Figure, blocking_df: pd.DataFrame, x_scale: float, y_scale: float, flip_y: bool, h_px: int):
    """
    将判定为封野的守卫点高亮（红色叉号），并在 hover 中显示封野 tick 次数与营地标签。
    """
    if blocking_df.empty:
        return
    x_px = blocking_df["x_grid"] * x_scale
    y_px = blocking_df["y_grid"] * y_scale
    if flip_y:
        y_px = (h_px - y_px)

    time_str = _fmt_ms(blocking_df.get("time_s"))
    life_str = _fmt_ms(blocking_df.get("lifetime_s"))

    fig.add_trace(go.Scatter(
        x=x_px, y=y_px,
        mode="markers",
        name="Camp-blocking wards",
        marker=dict(symbol="x", size=14, color="red", line=dict(color="black", width=0.6)),
        opacity=0.95,
        hovertemplate="Camp:%{customdata[0]}<br>ticks blocked:%{customdata[1]}<br>t:%{customdata[2]}<br>life:%{customdata[3]}",
        customdata=np.stack([blocking_df.get("camp_label",""), blocking_df.get("blocked_tick_count",0), time_str, life_str], axis=-1),
        legendgroup="blocking"
    ))


# ---------- Hotspots ----------
def compute_hotspots(df: pd.DataFrame, bins: int, min_count: int, top_k: int) -> pd.DataFrame:
    """
    将过滤后的 df 做二维直方图，返回热点中心与计数。
    输入坐标域是 0..GRID_SIZE（此处 0..128）。
    """
    if df.empty:
        return pd.DataFrame(columns=["x_grid", "y_grid", "count"])
    xs = pd.to_numeric(df["x_grid"], errors="coerce")
    ys = pd.to_numeric(df["y_grid"], errors="coerce")
    xs = xs.clip(0, GRID_SIZE - 1); ys = ys.clip(0, GRID_SIZE - 1)

    H, x_edges, y_edges = np.histogram2d(xs, ys, bins=bins, range=[[0, GRID_SIZE], [0, GRID_SIZE]])
    idx = np.argwhere(H >= max(1, int(min_count)))
    if idx.size == 0:
        return pd.DataFrame(columns=["x_grid", "y_grid", "count"])

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    rows = [{"x_grid": float(x_centers[i]), "y_grid": float(y_centers[j]), "count": int(H[i, j])} for i, j in idx]
    hotspots = pd.DataFrame(rows).sort_values(by="count", ascending=False)
    if top_k > 0:
        hotspots = hotspots.head(top_k)
    return hotspots.reset_index(drop=True)


def add_hotspots(fig: go.Figure, hs: pd.DataFrame, x_scale: float, y_scale: float, flip_y: bool, h_px: int, show_labels: bool):
    """
    在像素坐标上叠加热点标记：圆点大小与颜色按 count 映射，可选显示计数标签。
    """
    if hs.empty:
        return
    x_px = hs["x_grid"] * x_scale
    y_px = hs["y_grid"] * y_scale
    if flip_y: y_px = (h_px - y_px)

    counts = hs["count"].astype(float)
    cmin, cmax = (counts.min(), counts.max()) if len(counts) else (1.0, 1.0)
    sizes = 10.0 + (counts - cmin) / (max(1e-6, (cmax - cmin))) * (30.0 - 10.0)

    fig.add_trace(go.Scatter(
        x=x_px, y=y_px, mode="markers+text" if show_labels else "markers",
        name="Hotspots",
        marker=dict(symbol="circle", size=sizes, color=counts, colorscale="Turbo", showscale=True, line=dict(color="black", width=0.8)),
        text=[str(int(c)) for c in counts] if show_labels else None,
        textposition="top center",
        opacity=0.95,
        hovertemplate="Hotspot count:%{marker.color:.0f}<br>x(px):%{x:.1f} y(px):%{y:.1f}",
        legendgroup="hotspots"
    ))


def main():
    # 样式（侧边栏字体小一些）
    st.markdown("""
    <style>
    section.main > div {max-width: 1800px;}
    [data-testid="stSidebar"] {min-width: 360px; width: 360px;}
    [data-testid="stSidebar"] label {font-size: 0.90rem;}
    </style>
    """, unsafe_allow_html=True)

    # 标题
    st.title("Sword Ward Analysis")

    try:
        import plotly, PIL
        st.caption(f"Plotly={plotly.__version__} · Pillow={PIL.__version__} · Streamlit={st.__version__}")
    except Exception:
        pass

    left, right = st.columns([0.9, 3.1])

    with left:
        st.subheader("Control")
        team_name = st.text_input("Team Name", value="Team Falcons")
        mid_account = st.text_input("Mid Player Account ID", value="", placeholder="可留空不过滤")

        # 阵营过滤（Radiant / Dire / Both）
        side_choice = st.radio("Side", ["Both", "Radiant", "Dire"], index=0, horizontal=True,
                               help="同一支队伍在不同阵营的插眼位置可能不同，可选择只看某一阵营")

        # 生命周期过滤
        only_countered = st.checkbox("Countered only", value=False, help="仅保留寿命低于默认阈值的守卫（obs<360s / sen<420s）")
        only_natural   = st.checkbox("Natural expire only", value=False, help="仅保留寿命不低于默认阈值的守卫（obs≥360s / sen≥420s）")

        # 绝对时间窗口
        time_minute = st.slider("Event time window (minutes)", 0.0, 60.0, (0.0, 60.0), step=0.5)

        # 分段筛选：新增 Start（-1' 到 5'）
        st.write("Segments")
        c1, c2, c3, c4 = st.columns(4)
        with c1: seg_start = st.checkbox("Start (-1' to 5')", value=False)
        with c2: seg_early = st.checkbox("Early (<12')", value=True)
        with c3: seg_mid   = st.checkbox("Mid (<30')", value=True)
        with c4: seg_late  = st.checkbox("Late (>30')", value=True)

        st.markdown("---")
        # 图层
        st.write("Layers")
        w_obs = st.checkbox("Observer (obs)", value=True)
        w_sen = st.checkbox("Sentry (sen)", value=True)
        w_hot = st.checkbox("Hotspots", value=False, help="仅显示热点或与点位叠加显示")

        view_mode = st.radio("View for points", ["Scatter", "Heatmap"], index=0)

        # 野点与封野分析配置
        st.markdown("---")
        st.subheader("Neutral camp blocking (experimental)")
        show_camp_boxes = st.checkbox("Show neutral camp boxes", value=True)
        analyze_blocking = st.checkbox("Analyze camp blocking by wards", value=False,
                                       help="根据守卫位置与寿命，统计给定时间范围内的刷怪被阻断次数，并高亮阻断点")
        block_minute_range = st.slider("Blocking minute range", 0, 60, (0, 20), step=1,
                                       help="按整分钟 tick 统计（例如 0..20 分钟，每分钟一次）")
        use_default_lifetime = st.checkbox("Use default ward lifetimes when missing", value=True,
                                           help="缺少 lifetime_s 时，obs=360s，sen=420s")
        st.caption("Note: camp boxes 文件可为 128 或 256 网格；页面会自动缩放到内部 128 网格。")

        # Hotspots 控制
        st.markdown("---")
        if w_hot:
            st.subheader("Hotspots parameters")
            bins = st.slider("Bins per axis (hotspots)", 8, 64, 32, step=4)
            min_count = st.slider("Min count per bin", 1, 50, 5, step=1)
            top_k = st.slider("Top-K hotspots", 1, 100, 20, step=1)
            show_labels = st.checkbox("Show hotspot labels", value=True)
        else:
            bins = 32; min_count = 5; top_k = 20; show_labels = False

        # 主图尺寸与坐标
        st.markdown("---")
        max_width = st.slider("View width (px)", 600, 1600, 1000, step=20)
        inner_margin_pct = st.slider("Inner margin (%)", 0, 20, 6, step=1)
        flip_y = st.checkbox("Flip Y axis", value=True)

        st.markdown("---")
        st.subheader("Match filtering")
        filter_mode = st.radio("Mode", ["Show last N recent", "By date range"], horizontal=True, index=0)
        n_recent = st.slider("N (recent)", 1, 200, 20, step=1, disabled=(filter_mode != "Show last N recent"))
        today = datetime.now().date()
        date_range = st.date_input(
            "Date range",
            (today.replace(day=1), today),
            help="Filter matches by start time from matches JSON",
            disabled=(filter_mode != "By date range")
        )

        st.markdown("---")
        team_dir = DEFAULT_WARDS_DIR / "teams" / safe_name(team_name)
        st.caption("Paths")
        st.write("Wards root:", str(DEFAULT_WARDS_DIR))
        st.write("Team dir:", str(team_dir), "exists:", team_dir.exists())
        st.write("Config root:", str(CONFIG_ROOT), "exists:", CONFIG_ROOT.exists())
        st.write("Camp boxes file:", str(CAMP_BOXES_FILE), "exists:", CAMP_BOXES_FILE.exists())

    # 背景图：始终渲染
    map_path = find_default_map()
    if not map_path:
        right.error("Default map not found. Put image in frontend/assets (dota2_map.jpg/.png) or set WARD_MAP_IMAGE.")
        return
    try:
        img, (w, h) = load_map_resized(map_path)
    except Exception as e:
        right.error(f"Failed to read map: {e}")
        return

    fig, x_scale, y_scale = make_background_figure_image(img, w, h, width_px=max_width, flip_y=flip_y)

    margin = inner_margin_pct / 100.0
    x_min_px = w * margin
    x_max_px = w * (1.0 - margin)
    y_min_px = h * margin
    y_max_px = h * (1.0 - margin)

    if flip_y: fig.update_yaxes(range=[y_max_px, y_min_px], visible=False)
    else:      fig.update_yaxes(range=[y_min_px, y_max_px], visible=False)
    fig.update_xaxes(range=[x_min_px, x_max_px], visible=False)

    # 加载数据与比赛元信息
    df, discovered_paths, read_errors, match_ids = load_wards_csv(team_name)
    meta_df = load_matches_metadata_for_ids(match_ids) if match_ids else pd.DataFrame(columns=["match_id","start_ts","start_dt","radiant","dire","radiant_tag","dire_tag","league_name","json_path"])

    # 侧边栏：match 复选框标签（时间 + 联赛 + 简称对）
    st.sidebar.markdown("### Matches")
    checkbox_prefix = f"match_checkbox::{safe_name(team_name)}::"
    col_sa, col_ca = st.sidebar.columns(2)
    with col_sa:
        if st.sidebar.button("Select all"):
            for mid in match_ids:
                st.session_state[checkbox_prefix + mid] = True
    with col_ca:
        if st.sidebar.button("Clear all"):
            for mid in match_ids:
                st.session_state[checkbox_prefix + mid] = False

    checked_ids: List[str] = []
    if not meta_df.empty:
        for r in meta_df.itertuples():
            pair = f"{(r.radiant_tag or _abbr_from_name(r.radiant))}vs{(r.dire_tag or _abbr_from_name(r.dire))}"
            label = f"{(r.start_dt or '--')} · {(r.league_name or '').strip() or 'Unknown'} · {pair} · {r.match_id}"
            key = checkbox_prefix + r.match_id
            default_val = st.session_state.get(key, True)
            if st.sidebar.checkbox(label, value=default_val, key=key):
                checked_ids.append(r.match_id)
    else:
        for mid in match_ids:
            key = checkbox_prefix + mid
            default_val = st.session_state.get(key, True)
            if st.sidebar.checkbox(mid, value=default_val, key=key):
                checked_ids.append(mid)

    st.sidebar.caption(f"Checked: {len(checked_ids)}/{len(match_ids)}")

    # 时间筛选
    time_filtered_ids: List[str] = []
    if not meta_df.empty:
        if filter_mode == "Show last N recent":
            time_filtered_ids = meta_df["match_id"].tolist()[:n_recent]
        else:
            try:
                start_d: date = date_range[0]
                end_d: date = date_range[1]
            except Exception:
                start_d = today.replace(day=1)
                end_d = today
            start_dt = datetime.combine(start_d, datetime.min.time()).astimezone()
            # 修复：datetime.max.time()（原错误为 datetime.max time()）
            end_dt = datetime.combine(end_d, datetime.max.time()).astimezone()
            mask = meta_df["start_ts"].apply(lambda x: (pd.notna(x)) and (start_dt.timestamp() <= float(x) <= end_dt.timestamp()))
            time_filtered_ids = meta_df.loc[mask, "match_id"].tolist()
    else:
        time_filtered_ids = match_ids if filter_mode == "By date range" else match_ids[:n_recent]

    final_selected_ids = sorted(set(checked_ids) & set(time_filtered_ids)) if checked_ids else time_filtered_ids

    if read_errors:
        right.warning("Errors while reading CSVs:")
        for e in read_errors[:8]:
            right.write(e)
        if len(read_errors) > 8:
            right.write(f"... and {len(read_errors)-8} more")

    # 应用比赛过滤
    df = df[df["match_id"].astype(str).isin(set(final_selected_ids))]

    # 阵营过滤
    if side_choice != "Both":
        side_key = "radiant" if side_choice == "Radiant" else "dire"
        if "radiant_or_dire" in df.columns:
            df = df[df["radiant_or_dire"].astype(str).str.lower() == side_key]
        else:
            st.info("CSV缺少 radiant_or_dire 列，无法按阵营过滤。")

    # 玩家过滤
    if mid_account:
        acc_series = df.get("account_id")
        if acc_series is not None:
            df = df[acc_series.astype(str) == str(mid_account)]
        else:
            st.info("CSV missing account_id; cannot filter by player.")

    # 事件时间窗口（绝对窗口）
    t_lo_s = time_minute[0] * 60.0
    t_hi_s = time_minute[1] * 60.0
    df = df[(df["time_s"].isna()) | ((df["time_s"] >= t_lo_s) & (df["time_s"] <= t_hi_s))]

    # 分段筛选：Start/early/mid/late（Start 定义为 -60s 到 300s）
    seg_mask = np.zeros(len(df), dtype=bool)
    if seg_start: seg_mask |= ((df["time_s"] >= -60.0) & (df["time_s"] <= 5.0 * 60.0))
    if seg_early: seg_mask |= (df["time_s"] <= 12.0 * 60.0)
    if seg_mid:   seg_mask |= (df["time_s"] <= 30.0 * 60.0)
    if seg_late:  seg_mask |= (df["time_s"] > 30.0 * 60.0)
    if not (seg_start and seg_early and seg_mid and seg_late):
        df = df[seg_mask | df["time_s"].isna()]

    # 类型过滤（点位层用）
    scope_types: List[str] = []
    if w_obs: scope_types.append("obs")
    if w_sen: scope_types.append("sen")
    df_points = df[df["ward_type"].isin(scope_types)] if scope_types else df

    # 生命周期过滤（按默认阈值）
    life = pd.to_numeric(df_points.get("lifetime_s"), errors="coerce")
    thresholds = np.where(
        df_points["ward_type"].astype(str) == "obs",
        DEFAULT_OBS_MAX_LIFETIME,
        np.where(
            df_points["ward_type"].astype(str) == "sen",
            DEFAULT_SEN_MAX_LIFETIME,
            DEFAULT_OBS_MAX_LIFETIME
        )
    )
    valid = life.notna()
    if only_countered and not only_natural:
        df_points = df_points[valid & (life < thresholds)]
    elif only_natural and not only_countered:
        df_points = df_points[valid & (life >= thresholds)]

    df_points = df_points.dropna(subset=["x_grid", "y_grid"])

    # 绘制：刷怪框与封野分析
    camp_boxes_df, src_grid = load_camp_boxes()
    if show_camp_boxes and not camp_boxes_df.empty:
        add_camp_boxes_overlay(fig, camp_boxes_df if side_choice == "Both" else camp_boxes_df[camp_boxes_df["side"] == side_choice.lower()],
                               x_scale, y_scale, flip_y, h, show_labels=True)
        right.caption(f"Camp boxes loaded (src grid={src_grid}, internal={GRID_SIZE}).")
    elif show_camp_boxes and camp_boxes_df.empty:
        right.info("Camp boxes file not found or empty. Provide config/camp_boxes_256.json or 128.json.")

    blocking_df = pd.DataFrame()
    if analyze_blocking:
        if camp_boxes_df.empty:
            right.warning("Cannot analyze blocking: camp boxes file missing or empty.")
        else:
            stats_df, blocking_df = compute_camp_blocks(
                df_points, camp_boxes_df,
                minute_range=block_minute_range,
                use_default_lifetime=use_default_lifetime,
                consider_types=scope_types,
                side_choice=("both" if side_choice == "Both" else side_choice.lower())
            )
            if not stats_df.empty:
                right.caption("Camp blocking summary (top 10 by blocked ticks)")
                top = stats_df.head(10)
                lines = [f"{r.label} [{r.side}/{r.type}] · blocked {r.blocked_ticks}/{r.total_ticks} ({r.ratio*100:.1f}%)"
                         for r in top.itertuples()]
                right.write(lines)
            add_blocking_points(fig, blocking_df, x_scale, y_scale, flip_y, h)

    # 绘制：点位层
    if view_mode == "Scatter":
        if w_obs:
            add_obs_with_ring(fig, df_points[df_points["ward_type"] == "obs"], x_scale, y_scale, flip_y, h)
        if w_sen:
            add_sen_scaled(fig, df_points[df_points["ward_type"] == "sen"], x_scale, y_scale, flip_y, h)
    else:
        if scope_types:
            add_heatmap_image(fig, df_points, name="Heatmap", x_scale=x_scale, y_scale=y_scale, flip_y=flip_y, h_px=h)

    # Hotspots 图层
    if w_hot:
        hs = compute_hotspots(df_points if scope_types else df.dropna(subset=["x_grid", "y_grid"]),
                              bins=bins, min_count=min_count, top_k=top_k)
        if hs.empty:
            right.info("No hotspots under current filters.")
        else:
            add_hotspots(fig, hs, x_scale, y_scale, flip_y, h, show_labels)

    # 输出
    right.plotly_chart(fig, use_container_width=True)
    right.markdown("---")
    if not meta_df.empty:
        r0 = meta_df.iloc[0]
        pair = f"{(r0['radiant_tag'] or _abbr_from_name(r0['radiant']))}vs{(r0['dire_tag'] or _abbr_from_name(r0['dire']))}"
        right.caption(f"{pair} · {r0.get('league_name','')}")
    right.caption(f"Team: {team_name} · Side: {side_choice} · Account: {mid_account or '(no filter)'}")