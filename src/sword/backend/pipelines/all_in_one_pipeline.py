#!/usr/bin/env python3
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict

# Unified paths from shared module (src/sword/data/*)
try:
    from src.sword.shared.paths import (
        SCRIPTS_DIR, DATA_DIR, MATCHES_DIR, WARDS_DIR, REPLAYS_DIR, PICKS_DIR,
        ensure_dirs, summary
    )
except Exception:
    repo_root_src = Path(__file__).resolve().parents[2]  # <repo>/src
    if str(repo_root_src) not in sys.path:
        sys.path.append(str(repo_root_src))
    from src.sword.shared.paths import (
        SCRIPTS_DIR, DATA_DIR, MATCHES_DIR, WARDS_DIR, REPLAYS_DIR, PICKS_DIR,
        ensure_dirs, summary
    )

ensure_dirs()
print("[PATHS]", json.dumps(summary(), ensure_ascii=False))

PROGRESS_PATH = DATA_DIR / "pipeline_progress.json"
RESULT_PATH = DATA_DIR / "pipeline_results.json"

def run_command(description, command):
    """
    Run a command using subprocess and log its progress.
    """
    print(f"[INFO] Starting: {description}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logs = []
        for line in process.stdout:
            line = line.strip()
            if line:
                print(line)
                logs.append(line)
        _, stderr = process.communicate()

        if process.returncode == 0:
            print(f"[SUCCESS] Finished: {description}")
            return True, logs
        else:
            print(f"[ERROR] Failed during: {description}")
            if stderr:
                print(stderr)
            return False, logs + ([stderr.strip()] if stderr else [])
    except Exception as e:
        print(f"[ERROR] Exception during {description}: {e}")
        return False, [f"Exception: {str(e)}"]

def rglob_match_files(root: Path, mid: str, patterns: List[str]) -> List[str]:
    """
    在 root 下递归查找包含 match_id 的文件，支持多个模式。
    """
    files = []
    for pat in patterns:
        files.extend([str(p) for p in root.rglob(pat) if mid in p.name])
    return files

def stage_presence_for_mid(mid: str) -> Dict[str, str]:
    """
    返回某个 match_id 各阶段的存在状态（existing/missing）。
    """
    status = {}
    # JSON
    json_files = rglob_match_files(MATCHES_DIR, mid, ["*.json"])
    status["JSON"] = "existing" if json_files else "missing"
    # Obs logs CSV（可能位于 teams 子目录）
    obs_files = rglob_match_files(WARDS_DIR, mid, ["*.csv"])
    status["Obs Logs"] = "existing" if obs_files else "missing"
    # Replay DEM
    dem_files = rglob_match_files(REPLAYS_DIR, mid, ["*.dem"])
    status["Replays"] = "existing" if dem_files else "missing"
    # Picks parquet/csv
    pick_files = rglob_match_files(PICKS_DIR, mid, ["*.parquet", "*.csv"])
    status["Hero Picks"] = "existing" if pick_files else "missing"
    return status

def verify_output_files(directory, extension):
    return [str(p) for p in Path(directory).glob(f"*.{extension}")]

def get_team_meta(match_file: str) -> Dict[str, str]:
    """
    读取 match JSON 并提取队伍名称（失败时返回空字段）。
    """
    try:
        with open(match_file, 'r', encoding='utf-8') as f:
            match_data = json.load(f)
        radiant = match_data.get("radiant_name") or ""
        dire = match_data.get("dire_name") or ""
        # 兼容嵌套
        for side in ("radiant", "dire"):
            v = match_data.get(f"{side}_team")
            if isinstance(v, dict):
                name = v.get("name") or v.get("team_name") or ""
                if side == "radiant": radiant = radiant or name
                else: dire = dire or name
        return {"radiant_team": radiant or "unknown", "dire_team": dire or "unknown"}
    except Exception:
        return {"radiant_team": "unknown", "dire_team": "unknown"}

def write_progress(table_data: List[Dict]):
    """
    写入阶段性进度文件，供后端读取并推送到前端。
    """
    try:
        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump({"table_data": table_data}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def download_matches(match_ids: List[str]):
    temp_file = DATA_DIR / "temp_match_ids.txt"
    temp_file.write_text("\n".join(match_ids), encoding="utf-8")

    command = [
        sys.executable, str(SCRIPTS_DIR / "download_matches_by_ids.py"),
        "--ids-file", str(temp_file),
        "--out-dir", str(MATCHES_DIR),
        "--verbose",
        "--skip-existing",
    ]
    return run_command("Downloading Match JSON files", command)

def extract_obs_logs():
    command = [
        sys.executable, str(SCRIPTS_DIR / "download_obs_logs_from_matches.py"),
        "--matches-dir", str(MATCHES_DIR),
        "--out-dir", str(WARDS_DIR),
        "--verbose",
        "--skip-existing",
        "--group-by-team",
    ]
    return run_command("Extracting Obs Logs", command)

def download_replays():
    command = [
        sys.executable, str(SCRIPTS_DIR / "download_replay_file_from_matches.py"),
        "--matches-dir", str(MATCHES_DIR),
        "--out-dir", str(REPLAYS_DIR),
        "--verbose",
        "--skip-existing",
        "--group-by-team",
    ]
    return run_command("Downloading Replay files", command)

def download_picks():
    command = [
        sys.executable, str(SCRIPTS_DIR / "download_picks_from_matches.py"),
        "--matches-dir", str(MATCHES_DIR),
        "--out-dir", str(PICKS_DIR),
        "--export-aggregate",
        "--verbose",
        "--skip-existing",
        "--group-by-team",
    ]
    return run_command("Extracting Hero Picks and Bans", command)

def run_pipeline(match_ids_str: str):
    """
    执行管道并在每个阶段更新 table_data 状态。
    """
    # 解析 ID 列表
    match_ids = [m.strip() for m in match_ids_str.split(",") if m.strip()]

    # 初始状态（existing / missing）
    table_data = []
    for mid in match_ids:
        presence = stage_presence_for_mid(mid)
        # 队伍名（若 JSON 已存在可读取）
        team_meta = {"radiant_team": "unknown", "dire_team": "unknown"}
        json_files = rglob_match_files(MATCHES_DIR, mid, ["*.json"])
        if json_files:
            team_meta = get_team_meta(json_files[0])
        table_data.append({
            "Match ID": mid,
            "Radiant Team": team_meta["radiant_team"],
            "Dire Team": team_meta["dire_team"],
            "JSON": presence["JSON"],
            "Obs Logs": presence["Obs Logs"],
            "Replays": presence["Replays"],
            "Hero Picks": presence["Hero Picks"],
        })
    write_progress(table_data)

    # 下载 JSON
    json_ok, _ = download_matches(match_ids)
    # 更新 JSON 阶段状态
    for row in table_data:
        row["JSON"] = "downloaded" if json_ok or row["JSON"] == "existing" else "failed"
        # 刷队伍名
        team_meta = get_team_meta(next(iter(rglob_match_files(MATCHES_DIR, row["Match ID"], ["*.json"])), ""))
        row["Radiant Team"] = team_meta["radiant_team"]
        row["Dire Team"] = team_meta["dire_team"]
    write_progress(table_data)

    # 提取 Obs logs
    obs_ok, _ = extract_obs_logs()
    for row in table_data:
        # 若已存在则保持 existing，否则根据 obs_ok 更新
        if row["Obs Logs"] != "existing":
            obs_files = rglob_match_files(WARDS_DIR, row["Match ID"], ["*.csv"])
            row["Obs Logs"] = "downloaded" if obs_files or obs_ok else "failed"
    write_progress(table_data)

    # 下载 replays
    rep_ok, _ = download_replays()
    for row in table_data:
        if row["Replays"] != "existing":
            dem_files = rglob_match_files(REPLAYS_DIR, row["Match ID"], ["*.dem"])
            row["Replays"] = "downloaded" if dem_files or rep_ok else "failed"
    write_progress(table_data)

    # 下载/提取 picks
    picks_ok, _ = download_picks()
    for row in table_data:
        if row["Hero Picks"] != "existing":
            pick_files = rglob_match_files(PICKS_DIR, row["Match ID"], ["*.parquet", "*.csv"])
            row["Hero Picks"] = "downloaded" if pick_files or picks_ok else "failed"
    write_progress(table_data)

    # 最终结果写入
    results = {
        "pipeline_results": {
            "matches": {"success": json_ok},
            "obs_logs": {"success": obs_ok},
            "replays": {"success": rep_ok},
            "picks": {"success": picks_ok},
        },
        "table_data": table_data,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Dota2 Data Pipeline and log structured results.")
    parser.add_argument("--match-ids", type=str, required=True, help="Comma-separated list of Match IDs.")
    args = parser.parse_args()

    res = run_pipeline(args.match_ids)
    print(json.dumps(res, indent=2, ensure_ascii=False))