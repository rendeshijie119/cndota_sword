#!/usr/bin/env python3
"""
generate_team_side_insights.py

聚类/点位驱动的 ward-insights（按队、按阵营、按时间窗、可按 player-slot）——
重点识别“点位热点”（cluster centers），而不是模糊的整张热力图。

主要功能（高层说明）
- 扫描目录中 match_<id>_wards.csv 文件（按文件名中的 match id 降序，id 越大越新）
- 对给定队伍，分别挑选最近 N 场在 Radiant 与 Dire 位置的比赛（--n-per-side）
- 对每个 side 与每个时间窗（默认 0-10 / 10-25 / 25-40 分钟）：
  - 收集指定 ward types（默认只看 obs）与可选 player_slot 的点位
  - 对坐标做用户指定的平移（--subtract-x/--subtract-y，保持你现有的 -64 修正）
  - 将 grid 坐标（CSV 中的 x,y）映射到地图像素坐标（使用 (grid_size - 1) 作为分母）
  - 在像素空间上用 DBSCAN 做聚类以识别“点位热点”
  - 输出每个聚类的中心、点数、占比，并将聚类标注到一张带注释的 PNG（方便可视化）
  - 可选：将聚类中心归类到用户提供的语义区域（regions JSON），从而输出类似“70% 的眼在自家三角区”的结论

为什么这样做（动机）
- 你的目标是“点位洞察”（他们把眼插在哪里）而不是整张地图的模糊密度；
  DBSCAN 可以把密集的点簇识别出来并给出“该簇覆盖了多少插眼点 / 占比多少”；
  然后结合 regions（语义多边形），可以得到直观的文本结论（例如“70% 在三角区”）。

注意事项（使用前阅读）
- grid_size：表示 CSV 中坐标的逻辑网格范围 (常见 256)。不要把它设为图片像素尺寸。
  如果 set 为 0，脚本会基于选中的比赛自动推断合适的 grid_size（并优先匹配 256/512/1024）。
- subtract-x / subtract-y：默认 64（保留你之前的 -64 行为）。这是在将 grid->pixel 之前对原始 CSV 坐标做的平移修正。
  如果你不希望对坐标变换，设置为 0。
- eps（DBSCAN 半径）：单位是像素（不是 grid 单位），在 900px 左右图片上 20-50 常用；
  如果地图像素大小不同，请相应调整。
- 若提供 regions JSON（多边形使用 GRID 坐标），脚本会在报告里尝试用 region 名称来标注簇。
  注意：regions 的坐标与 CSV 的 grid 坐标一致（脚本内部会把聚类中心从像素反算回 grid，然后加回 subtract-offset 再做点在多边形判断）。

输出（文件与打印）
- 控制台打印：每个 side+time-window 的聚类摘要（包括 region 判断）
- CSV 报表：每个 side+time-window 一份，列出簇排名 / 点数 / 占比 / 中心像素 / 近似 grid 坐标
- 注释 PNG：每个 side+time-window 一张图（聚类中心用圈和编号标注），存放到 --out 文件夹

依赖
    pip install pandas numpy matplotlib pillow scikit-learn seaborn

示例
    python3 generate_team_side_insights.py \
      --in ./output --team "Xtreme Gaming" --map images/map_738.jpg \
      --out ./insights --n-per-side 5 --grid-size 128 \
      --subtract-x 64 --subtract-y 64 --player-slot 4 --eps 32 --min-samples 3 --verbose

作者注：脚本力求稳健（对缺失列和异常值有容错处理），并在关键步骤打印诊断信息以便调参。
"""

from pathlib import Path
import argparse
import re
import math
import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# 匹配文件名的正则：match_<id>_wards.csv
FILENAME_RE = re.compile(r"match_(\d+)_wards\.csv$", re.IGNORECASE)


# ---------------------
# 文件 / 列读取相关
# ---------------------
def list_match_csvs(folder: Path) -> List[Tuple[int, Path]]:
    """
    列出输入目录中所有符合 match_<id>_wards.csv 格式的文件并按 match id 降序返回。
    返回：[(match_id, Path), ...]，id 越大表示越新的比赛（假设 id 单调递增）。
    """
    files = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = FILENAME_RE.search(p.name)
        if m:
            try:
                mid = int(m.group(1))
            except Exception:
                continue
            files.append((mid, p))
    # 降序（最新的在前）
    files.sort(key=lambda x: x[0], reverse=True)
    return files


def parse_xy_from_row(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    从一行数据（字典或 pandas.Series 转为 dict）解析 x,y 坐标。
    支持：
      - 直接的列名 'x','pos_x','posx','position_x' 及对应 y 列
      - 如果没有数值列，则回退解析 'key' 列（示例 "[130,126]"）
    返回 (x, y) 或 None（无法解析时）
    注：返回的坐标仍然是 grid 单位（例如 0..255），后续会做 subtract / map -> pixels。
    """
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
    # fallback: key 字段，格式可能是 "[x,y]" 或 "x,y"
    key = row.get("key")
    if isinstance(key, str) and key:
        mm = re.search(r"\[?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]?", key)
        if mm:
            try:
                return (float(mm.group(1)), float(mm.group(2)))
            except Exception:
                return None
    return None


# ---------------------
# 推断 grid 大小
# ---------------------
def infer_grid_size_from_rows(rows: List[Dict[str, Any]]) -> int:
    """
    从若干行数据里观察最大坐标并推断 grid_size。
    策略：
      - 计算观察到的最大绝对坐标 max_coord
      - candidate = ceil(max_coord) + 1
      - 若 candidate 距常见值 256/512/1024 很接近，则 snap 到该常见值
      - 否则返回 max(256, candidate)
    这样在坐标是 0..255 时会返回 256（常见情况）。
    """
    max_coord = 0.0
    for r in rows:
        xy = parse_xy_from_row(r)
        if xy:
            max_coord = max(max_coord, abs(xy[0]), abs(xy[1]))
    if max_coord <= 0:
        return 256
    candidate = int(math.ceil(max_coord)) + 1
    for s in (256, 512, 1024):
        if abs(candidate - s) <= max(1, int(0.02 * s)):  # within ~2%
            return s
    return max(256, candidate)


# ---------------------
# 按队挑选最近比赛（每阵营）
# ---------------------
def gather_recent_matches_for_team(folder: Path, team_identifier: str, n_per_side: int,
                                   verbose: bool = False, team_is_id: bool = False) -> Dict[str, List[Path]]:
    """
    查找目录中与给定 team_identifier（队名或 team_id）匹配的最近比赛文件，并分别返回
    最近 n_per_side 场 Radiant 和 Dire 的文件路径数组。
    返回字典：{"radiant": [Path,...], "dire": [Path,...]}
    """
    files = list_match_csvs(folder)
    sides = {"radiant": [], "dire": []}
    for mid, path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            if verbose:
                print(f"[warn] skipping unreadable CSV {path}")
            continue

        # team 匹配：优先按 id（如果用户传了数字），否则按 team_name 不区分大小写匹配
        if team_is_id:
            try:
                tid = int(team_identifier)
                mask = df["team_id"].notnull() & (df["team_id"].astype(str) == str(tid))
            except Exception:
                mask = df["team_id"].notnull() & (df["team_id"].astype(str) == str(team_identifier))
        else:
            mask = df["team_name"].notnull() & (df["team_name"].astype(str).str.lower() == str(team_identifier).lower())

        if not mask.any():
            continue

        # 该文件中此队可能既有 radiant 行也有 dire 行（通常只会有一边），把存在的 side 加入列表
        sides_present = df.loc[mask, "radiant_or_dire"].dropna().unique().tolist()
        for s in sides_present:
            s = str(s).lower()
            if s in ("radiant", "dire"):
                if len(sides[s]) < n_per_side:
                    sides[s].append(path)
        # 提前退出：当两边都凑齐时
        if len(sides["radiant"]) >= n_per_side and len(sides["dire"]) >= n_per_side:
            break

    if verbose:
        print(f"[diagnostic] found {len(sides['radiant'])} radiant and {len(sides['dire'])} dire matches for team '{team_identifier}'")
    return sides


# ---------------------
# 从文件收集 x,y 点 （按 side / time / player / ward types）
# ---------------------
def collect_xy_from_files(file_list: List[Path], side: str, time_start: Optional[float],
                          time_end: Optional[float], player_slot: Optional[int],
                          ward_types: List[str], verbose: bool = False) -> List[Tuple[float, float]]:
    """
    从给定文件列表中收集指定 side 的 (x,y) 点。
    过滤条件：
      - side: "radiant" 或 "dire"
      - 时间：time_start <= time <= time_end（单位：秒），若未提供则不过滤
      - player_slot：若提供则只保留该 player_slot 的事件
      - ward_types：只保留列 ward_type 属于该列表的事件（不区分大小写）
    返回：[(x,y), ...]（grid 单位）
    """
    pts: List[Tuple[float, float]] = []
    ward_types_normalized = [wt.strip().lower() for wt in ward_types if wt.strip()]
    for p in file_list:
        if verbose:
            print(f"[read] {p.name} for side={side}")
        try:
            df = pd.read_csv(p)
        except Exception:
            if verbose:
                print(f"[warn] could not read {p}")
            continue

        # side 过滤
        side_mask = df["radiant_or_dire"].notnull() & (df["radiant_or_dire"].astype(str).str.lower() == side)
        df_side = df.loc[side_mask].copy()

        # ward_type 过滤（如果该列存在）
        if ward_types_normalized and "ward_type" in df_side.columns:
            try:
                df_side = df_side[df_side["ward_type"].astype(str).str.lower().isin(ward_types_normalized)]
            except Exception:
                if verbose:
                    print(f"[warn] ward_type filtering failed on {p}")

        # player_slot 过滤（如果指定）
        if player_slot is not None and "player_slot" in df_side.columns:
            try:
                df_side = df_side[pd.to_numeric(df_side["player_slot"], errors="coerce") == int(player_slot)]
            except Exception:
                if verbose:
                    print(f"[warn] unable to filter by player_slot on {p}")

        # time 过滤（如果指定）
        if time_start is not None or time_end is not None:
            if "time" in df_side.columns:
                # 先把 time 字段转成数值
                df_side["t_sec"] = pd.to_numeric(df_side["time"], errors="coerce")
                if time_start is not None:
                    df_side = df_side[df_side["t_sec"] >= time_start]
                if time_end is not None:
                    df_side = df_side[df_side["t_sec"] <= time_end]

        # 提取 x,y
        for _, row in df_side.iterrows():
            rowd = row.to_dict() if not isinstance(row, dict) else row
            xy = parse_xy_from_row(rowd)
            if xy:
                pts.append(xy)
    return pts


# ---------------------
# grid -> image pixel 映射（数组版）
# ---------------------
def grid_to_image_pixels(xs: np.ndarray, ys: np.ndarray, grid_size: int, img_w: int, img_h: int, flip_y: bool):
    """
    把一组 grid 坐标 (xs, ys) 映射到图片像素坐标 (px, py)：
      - 使用 max_index = grid_size - 1 作为分母（保证 grid 范围两端 0 与 max_index 对应图片两端）
      - 使用 np.clip 限制在 [0, max_index]
      - px = (gx / max_index) * (img_w - 1)
      - py = (gy / max_index) * (img_h - 1)
      - 若 flip_y=True，垂直翻转 py（py = (img_h - 1) - py）
    返回 (px, py) 两个 numpy 数组（float）
    """
    if grid_size <= 1:
        raise ValueError("grid_size must be > 1")
    max_index = grid_size - 1
    gx = np.clip(xs.astype(float), 0.0, float(max_index))
    gy = np.clip(ys.astype(float), 0.0, float(max_index))
    px = (gx / max_index) * (img_w - 1)
    py = (gy / max_index) * (img_h - 1)
    if flip_y:
        py = (img_h - 1) - py
    return px, py


# ---------------------
# clustering与可视化辅助
# ---------------------
def cluster_and_report(px: np.ndarray, py: np.ndarray, top_k: int, eps: int, min_samples: int):
    """
    在像素空间上使用 DBSCAN 做聚类：
      - eps: 半径（像素）
      - min_samples: 最小样本数（簇阈值）
    返回：按簇大小降序排列的字典列表（每个包含 center, count, points）
    过滤掉 label == -1（噪声）
    """
    if len(px) == 0:
        return []
    XY = np.vstack([px, py]).T  # N x 2
    cl = DBSCAN(eps=eps, min_samples=min_samples).fit(XY)
    labels = cl.labels_
    clusters: Dict[int, Dict[str, Any]] = {}
    for lbl in set(labels):
        if lbl == -1:
            continue
        mask = labels == lbl
        coords = XY[mask]
        center = coords.mean(axis=0)
        count = coords.shape[0]
        clusters[lbl] = {"center": center, "count": int(count), "points": coords}
    ordered = sorted(clusters.values(), key=lambda c: c["count"], reverse=True)
    return ordered[:top_k]


def annotate_image_with_clusters(img_path: Path, clusters: List[Dict[str, Any]], out_path: Path):
    """
    在原始 minimap 上画出聚类中心和编号：
      - 圆的半径按簇大小（count）缩放，便于视觉区分
      - 在圆旁标出编号与点数（例如 "1: 12"）
    输出：保存 PNG 到 out_path
    """
    img = Image.open(img_path).convert("RGBA")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)
    # 选择字体（系统可用则用 DejaVuSans）
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    max_count = max((c["count"] for c in clusters), default=1)
    for idx, c in enumerate(clusters):
        cx, cy = float(c["center"][0]), float(c["center"][1])
        cnt = c["count"]
        # radius 与 sqrt(count) 成比例，避免单簇过大
        r = 8 + int(20 * math.sqrt(cnt / max_count))
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, outline=(255, 0, 0, 220), width=3)
        label = f"{idx+1}: {cnt}"
        draw.text((cx + r + 2, cy - 6), label, fill=(255, 255, 255, 240), font=font)
    img.save(out_path)


def safe_name(s: str) -> str:
    """把任意字符串转成文件名安全的形式（仅字母数字和下划线/短横线）"""
    return re.sub(r"[^\w\-]+", "_", str(s).strip()).strip("_")[:200]


# ---------------------
# regions 加载与点内判定
# ---------------------
def load_regions_if_any(regions_file: Optional[Path]) -> List[Dict[str, Any]]:
    """
    regions JSON 格式示例（用户提供，使用 GRID 坐标）：
    [
      {"name":"radiant_triangle","description":"...", "polygon":[[x1,y1],[x2,y2],...]},
      ...
    ]
    返回列表（若加载失败或未给出则返回空列表）
    """
    if not regions_file:
        return []
    try:
        with open(regions_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data
    except Exception as e:
        print(f"[warn] cannot load regions file: {e}")
        return []


def point_in_polygon(x, y, poly):
    """
    判断点 (x,y) 是否在多边形 poly 内（ray casting 算法）。
    poly: list of [x,y]，按 GRID 单位给出。
    注意：如果多边形顶点数量很少或自交，结果可能不可靠。
    """
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


# ---------------------
# 主流程：解析参数，执行收集/聚类/输出
# ---------------------
def main():
    p = argparse.ArgumentParser(description="Generate ward 'point' insights via clustering.")
    p.add_argument("--in", dest="in_dir", type=Path, required=True, help="Folder containing match_<id>_wards.csv files")
    p.add_argument("--team", required=True, help="Team name (string) or team id (integer)")
    p.add_argument("--map", dest="map_image", type=Path, required=True, help="Path to minimap image (PNG/JPG)")
    p.add_argument("--out", dest="out_dir", type=Path, default=Path("./insights"), help="Output folder for reports and annotated images")
    p.add_argument("--n-per-side", type=int, default=5, help="Number of recent matches per side to include")
    p.add_argument("--grid-size", type=int, default=256, help="Grid size used by CSV coords (default 256). Pass 0 to auto-detect.")
    p.add_argument("--subtract-x", type=float, default=64.0, help="Subtract this amount from each ward x coordinate before mapping (default 64)")
    p.add_argument("--subtract-y", type=float, default=64.0, help="Subtract this amount from each ward y coordinate before mapping (default 64)")
    p.add_argument("--player-slot", type=int, default=None, help="If provided, only include ward events placed by this player_slot")
    p.add_argument("--ward-types", type=str, default="obs", help="Comma-separated ward types to include (default 'obs')")
    p.add_argument("--eps", type=float, default=32.0, help="DBSCAN eps (pixels) — 聚类半径，单位为像素")
    p.add_argument("--min-samples", type=int, default=3, help="DBSCAN min samples — 簇中至少包含这个数量的点")
    p.add_argument("--top-k", type=int, default=5, help="只输出每窗 top K 个聚类")
    p.add_argument("--regions-file", type=Path, default=None, help="Optional JSON file defining named polygons in GRID coords")
    p.add_argument("--windows", type=str, default="0-10,10-25,25-40", help="Comma-separated minute windows like '0-10,10-25,25-40'")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # team 可以是数字（team_id）也可以是字符串（team name）
    team = args.team
    team_is_id = False
    try:
        int(team)
        team_is_id = True
    except Exception:
        team_is_id = False

    # 解析时间窗参数（分钟 -> 秒）
    windows: List[Tuple[int, int]] = []
    for tok in [t.strip() for t in args.windows.split(",")]:
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", tok)
        if not m:
            raise SystemExit(f"invalid window: {tok}")
        a = int(m.group(1)) * 60
        b = int(m.group(2)) * 60
        windows.append((a, b))

    # regions（可选）
    regions = load_regions_if_any(args.regions_file)

    # 按 side 挑选最近比赛
    sides_matches = gather_recent_matches_for_team(args.in_dir, team, args.n_per_side, verbose=args.verbose, team_is_id=team_is_id)
    if args.verbose:
        print("Selected matches per side:")
        for s, arr in sides_matches.items():
            print(f"  {s}: {[p.name for p in arr]}")

    # 规范 ward types 列表
    ward_types = [w.strip().lower() for w in args.ward_types.split(",") if w.strip()]

    # 对每个 side 与每个 time window 执行收集/聚类/输出
    for side, files in sides_matches.items():
        if not files:
            if args.verbose:
                print(f"[skip] no matches for side {side}")
            continue

        # grid_size 支持 auto-detect（合并所选文件的数据来估计）
        grid_size = args.grid_size
        if grid_size == 0:
            all_rows = []
            for f in files:
                df = pd.read_csv(f)
                all_rows.extend(df.to_dict(orient="records"))
            grid_size = infer_grid_size_from_rows(all_rows)
            if args.verbose:
                print(f"[diagnostic] auto-detected grid_size={grid_size} for side {side}")

        for (tstart, tend) in windows:
            if args.verbose:
                print(f"[collect] side={side} window={tstart}-{tend} secs ({tstart//60}-{tend//60} min)")

            # 收集 grid 单位坐标（x,y）
            pts = collect_xy_from_files(files, side, time_start=tstart, time_end=tend,
                                        player_slot=args.player_slot, ward_types=ward_types, verbose=args.verbose)
            if not pts:
                if args.verbose:
                    print(f"[info] no points collected for side={side} window={tstart}-{tend}")
                continue

            # 转成 numpy 数组并应用用户指定的 subtract 偏移（这是你要求保留的 -64 行为）
            xs = np.array([p[0] for p in pts], dtype=float)
            ys = np.array([p[1] for p in pts], dtype=float)
            xs = xs - float(args.subtract_x)
            ys = ys - float(args.subtract_y)

            # 把 grid 映射到图片像素（默认不翻转 y，若你的 map 需要翻转，可在这里改为 flip_y=True）
            img = Image.open(args.map_image)
            img_w, img_h = img.size
            px, py = grid_to_image_pixels(xs, ys, grid_size, img_w, img_h, flip_y=False)

            # 用 DBSCAN 做聚类（在像素空间）
            clusters = cluster_and_report(px, py, top_k=args.top_k, eps=int(args.eps), min_samples=args.min_samples)

            # 汇总报表：打印并写 CSV
            total = len(px)
            report_lines = []
            report_lines.append(f"Team={team} side={side} window={tstart//60}-{tend//60}min total_points={total} ward_types={ward_types}")
            for i, c in enumerate(clusters):
                cx, cy = float(c["center"][0]), float(c["center"][1])
                cnt = c["count"]
                pct = cnt * 100.0 / total if total > 0 else 0.0

                # 把聚类中心从像素近似反算到 grid（方便在 regions.json 中匹配）
                grid_max = max(1, grid_size - 1)
                gx = (cx / (img_w - 1)) * grid_max
                gy = (cy / (img_h - 1)) * grid_max

                # regions 判断：注意 regions 存的是 GRID 坐标，脚本把 center 反算回 grid 并加回 subtract，
                # 因此这里传入的是 (gx + subtract_x, gy + subtract_y) 去判断是否落在多边形内
                region_name = None
                for reg in regions:
                    poly = reg.get("polygon", [])
                    if poly and point_in_polygon(gx + args.subtract_x, gy + args.subtract_y, poly):
                        region_name = reg.get("name")
                        break

                if region_name:
                    report_lines.append(f"  Cluster {i+1}: count={cnt} ({pct:.1f}%) center_px=({cx:.1f},{cy:.1f}) approx_grid=({gx:.1f},{gy:.1f}) region={region_name}")
                else:
                    report_lines.append(f"  Cluster {i+1}: count={cnt} ({pct:.1f}%) center_px=({cx:.1f},{cy:.1f}) approx_grid=({gx:.1f},{gy:.1f})")

            # 打印报告到控制台
            print("\n".join(report_lines))

            # 写 CSV 汇总（如果有聚类）
            csv_rows = []
            for i, c in enumerate(clusters):
                cx, cy = float(c["center"][0]), float(c["center"][1])
                cnt = c["count"]
                pct = cnt * 100.0 / total if total > 0 else 0.0
                gx = (cx / (img_w - 1)) * max(1, grid_size - 1)
                gy = (cy / (img_h - 1)) * max(1, grid_size - 1)
                csv_rows.append({
                    "team": team,
                    "side": side,
                    "window_min_start": tstart // 60,
                    "window_min_end": tend // 60,
                    "cluster_rank": i + 1,
                    "count": cnt,
                    "pct": pct,
                    "center_px_x": cx,
                    "center_px_y": cy,
                    "approx_grid_x": gx,
                    "approx_grid_y": gy
                })

            import csv as _csv
            csv_path = out_dir / f"{safe_name(team)}_{side}_{tstart//60}-{tend//60}min_last{len(files)}"
            if args.player_slot is not None:
                csv_path = csv_path.with_name(csv_path.name + f"_slot{args.player_slot}.csv")
            else:
                csv_path = csv_path.with_suffix(".csv")
            if csv_rows:
                with csv_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = _csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
                    writer.writeheader()
                    for r in csv_rows:
                        writer.writerow(r)
                if args.verbose:
                    print(f"[write] summary CSV {csv_path}")

            # 保存注释图（带聚类圆与编号）
            img_out = out_dir / f"{safe_name(team)}_{side}_{tstart//60}-{tend//60}min_last{len(files)}"
            if args.player_slot is not None:
                img_out = img_out.with_name(img_out.name + f"_slot{args.player_slot}.png")
            else:
                img_out = img_out.with_suffix(".png")
            annotate_image_with_clusters(args.map_image, clusters, img_out)
            if args.verbose:
                print(f"[write] annotated image {img_out}")

    print("Done.")


if __name__ == "__main__":
    main()