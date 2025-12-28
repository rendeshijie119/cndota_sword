#!/usr/bin/env python3
"""
download_obs_logs.py

工具用途（通用版）：
- 扫描一个或多个 JSON 日志文件（或一个目录），提取玩家信息并导出为 CSV/打印到终端。
- 主要目标：修复/兼容玩家名字字段，优先使用 'personaname'，并回退到常见旧字段或拼写错误。
- 提供可选的文件重命名功能：将日志文件重命名为包含第一个玩家名称（已清洗）的形式（用于便于识别）。
- 该脚本为通用模板，适合放在 CI/ETL 或开发环境中。请根据你的真实数据源（API、存储路径、字段结构）做必要调整。

主要改动点（相对于旧版）：
- 增加了 get_player_name(...) 兼容函数，优先使用 personaname，回退到常见变体。
- 在解析日志时，尽量处理不同结构（players、player_list、participants 等）。
- 更强的错误与容错日志，避免因字段缺失抛异常而中断整个批处理。

用法示例：
    python download_obs_logs.py --input logs/ --output players.csv
    python download_obs_logs.py --input game-123.json --rename --dry-run

注意：
- 本脚本并不会实际从外部服务“下载”OBS日志（除非你扩展 network_fetch 函数）。
- 在集成到你的仓库前，请把 I/O、字段映射和任何写回（双写）逻辑与后端契约对齐。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("download_obs_logs")


def get_player_name(player: Any) -> str:
    """
    兼容地从 player dict 中获取名字。
    优先使用 'personaname'，然后回退到常见变体或拼写错误，最后返回 'unknown'.

    支持输入类型检查：如果 player 不是 dict，会将其转换成字符串或返回 "unknown".
    """
    if player is None:
        return "unknown"
    if isinstance(player, str):
        # 已经是字符串（有时数据就是列表名字），直接返回清洗过的字符串
        return _sanitize_name(player)

    if not isinstance(player, dict):
        # 非 dict 且非 str，尝试 str() 转换并清洗
        try:
            return _sanitize_name(str(player))
        except Exception:
            return "unknown"

    # 列出优先级，从最常用到最不常用
    candidates = [
        "personaname",
        "persona_name",
        "person_name",
        "player_name",
        "name",
        "nick",
        "nickname",
        "personal_name",  # 旧/误拼写
        "perssonname",    # 常见错拼
    ]

    # 直接尝试 keys，且允许嵌套（有时 player 可能包含 profile:{...}）
    for key in candidates:
        v = player.get(key)
        if v:
            return _sanitize_name(v)

    # 检查常见嵌套位置
    for parent in ("profile", "account", "user"):
        p = player.get(parent)
        if isinstance(p, dict):
            for key in candidates:
                v = p.get(key)
                if v:
                    return _sanitize_name(v)

    # 如果存在 display 或 title 字段，也尝试
    for alt in ("display_name", "display", "title"):
        v = player.get(alt)
        if v:
            return _sanitize_name(v)

    # 作为最后手段，尝试查找任何键中含 name 的值
    for k, v in player.items():
        if "name" in k.lower() and v:
            return _sanitize_name(v)

    # 仍然没有，尝试 steamid/steam_id/uid 当作标识
    for alt_id in ("steamid", "steam_id", "account_id", "uid", "id"):
        v = player.get(alt_id)
        if v:
            return _sanitize_name(str(v))

    return "unknown"


def _sanitize_name(raw: Any) -> str:
    """
    基本清洗：将 bytes/非 str 转为 str，去除换行、控制字符，收缩空格，截断过长名字。
    """
    if raw is None:
        return "unknown"
    try:
        s = str(raw)
    except Exception:
        return "unknown"

    # 去掉不可见控制字符
    s = re.sub(r"[\r\n\t]+", " ", s)
    # 收缩多个空格
    s = re.sub(r"\s{2,}", " ", s).strip()
    # 截断过长名称（可配置）
    max_len = 200
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    if s == "":
        return "unknown"
    return s


def find_players_root(obj: Any) -> Optional[Iterable[Any]]:
    """
    尝试在日志文件中找到包含玩家数组的字段。
    支持常见字段名：players, player, participants, participants_list, player_list 等。
    如果找不到，将返回 None。
    """
    if not isinstance(obj, dict):
        return None

    candidate_arrays = [
        "players",
        "player",
        "participants",
        "participants_list",
        "player_list",
        "players_list",
        "players_info",
        "playersData",
        "players_info",
    ]

    for key in candidate_arrays:
        arr = obj.get(key)
        if isinstance(arr, list):
            return arr

    # 有些结构可能是 obj["match"]["players"]
    for parent in ("match", "game", "data", "payload"):
        child = obj.get(parent)
        if isinstance(child, dict):
            for key in candidate_arrays:
                arr = child.get(key)
                if isinstance(arr, list):
                    return arr

    # 最后尝试扫描所有 values，寻找第一个 list 且包含 dict 元素（可能是玩家列表）
    for k, v in obj.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            # 简单启发式：若第一个元素含 name/steamid 等键，认为这是玩家列表
            first = v[0]
            if any(x in first for x in ("personaname", "name", "steamid", "id", "nickname")):
                logger.debug("Found players array heuristically under key '%s'", k)
                return v

    return None


def process_log_file(path: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    读取一个 JSON 日志文件，解析并返回 (filename, list_of_players)
    每个 player 为 dict，包含至少 'name' 字段以及可选的 'id'、'original'（原始字典）。
    """
    logger.debug("Processing file: %s", path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON in %s: %s", path, e)
        return (os.path.basename(path), [])
    except Exception as e:
        logger.error("Failed to read %s: %s", path, e)
        return (os.path.basename(path), [])

    players_root = find_players_root(data)
    if players_root is None:
        # 可能整个文件就是玩家数组
        if isinstance(data, list):
            players_root = data
        else:
            logger.warning("No players array found in %s", path)
            return (os.path.basename(path), [])

    out: List[Dict[str, str]] = []
    for p in players_root:
        name = get_player_name(p)
        # 尝试抓取 id
        pid = None
        if isinstance(p, dict):
            for k in ("steamid", "steam_id", "account_id", "id", "uid"):
                if p.get(k):
                    pid = str(p.get(k))
                    break
        out.append({"name": name, "id": pid or "", "original": json.dumps(p, ensure_ascii=False)})

    return (os.path.basename(path), out)


def walk_input_paths(paths: Iterable[str]) -> Iterable[str]:
    """
    给定一个或多个路径，展开成实际文件列表（递归目录，筛选 .json 文件）。
    """
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(".json"):
                        yield os.path.join(root, f)
        elif os.path.isfile(p):
            yield p
        else:
            logger.warning("Input path not found: %s", p)


def write_csv(output_file: str, rows: Iterable[Tuple[str, List[Dict[str, str]]]]) -> None:
    """
    将处理结果写入 CSV。输出列：source_file, player_name, player_id, original_json
    """
    with open(output_file, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_file", "player_name", "player_id", "original_json"])
        for source_file, players in rows:
            for p in players:
                writer.writerow([source_file, p.get("name", ""), p.get("id", ""), p.get("original", "")])
    logger.info("Wrote CSV: %s", output_file)


def safe_rename_file(src: str, new_basename: str, dry_run: bool = False) -> Optional[str]:
    """
    将 src 文件重命名为 new_basename（保持原目录），避免覆盖已有文件（会附加序号）。
    返回新的文件路径（或 None 如果失败或 dry_run）。
    """
    dirpath = os.path.dirname(src)
    ext = os.path.splitext(src)[1]
    safe_name = _sanitize_name_for_filename(new_basename)
    target = os.path.join(dirpath, f"{safe_name}{ext}")
    if os.path.exists(target):
        # 加序号
        i = 1
        while True:
            candidate = os.path.join(dirpath, f"{safe_name}-{i}{ext}")
            if not os.path.exists(candidate):
                target = candidate
                break
            i += 1
    if dry_run:
        logger.info("[dry-run] rename %s -> %s", src, target)
        return None
    try:
        os.rename(src, target)
        logger.info("Renamed %s -> %s", src, target)
        return target
    except Exception as e:
        logger.error("Failed to rename %s -> %s: %s", src, target, e)
        return None


def _sanitize_name_for_filename(name: str) -> str:
    """
    将名字转换为文件名安全格式：移除/替换非法字符，截断长度。
    """
    if not name:
        return "unknown"
    s = str(name)
    # 保留字母数字和少量符号，其他替换为下划线
    s = re.sub(r"[^\w\-_\. ]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    if len(s) > 80:
        s = s[:80]
    return s


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process OBS/log JSON files and extract player names (personaname compatible).")
    parser.add_argument("--input", "-i", nargs="+", required=True, help="Input file(s) or directory(ies) containing JSON logs.")
    parser.add_argument("--output", "-o", default="players.csv", help="Output CSV filename.")
    parser.add_argument("--rename", action="store_true", help="Rename source files to include first player's name.")
    parser.add_argument("--dry-run", action="store_true", help="When used with --rename, do not actually rename files.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    input_paths = list(walk_input_paths(args.input))
    if not input_paths:
        logger.error("No input files found from: %s", args.input)
        return 2

    results: List[Tuple[str, List[Dict[str, str]]]] = []
    for path in input_paths:
        filename, players = process_log_file(path)
        results.append((filename, players))
        if args.rename and players:
            # 使用第一个玩家的名称作为重命名基准（已清洗）
            first_player = players[0].get("name", "unknown")
            safe_rename_file(path, f"{first_player}", dry_run=args.dry_run)

    # 写入 CSV
    try:
        write_csv(args.output, results)
    except Exception as e:
        logger.error("Failed to write CSV %s: %s", args.output, e)
        return 3

    logger.info("Processed %d files.", len(input_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())