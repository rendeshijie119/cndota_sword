#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 OpenDota 抓取并中文化输出 BP 数据：
- team_actions.json（逐手明细，中文字段/枚举/英雄名）
- team_hero_pick_stats.json（我方自选统计，中文）
- opponents_ban_vs_team.json（对手针对禁用统计，中文）
- manifest.json

运行示例：
  python opendota_team_bp_pipeline.py --team_id 9572001 --out_dir ./out --since 2024-01-01

环境变量（可选）：
  OPENDOTA_API_KEY  提高配额（没有也能用）
"""

import os, time, json, argparse, sys
import datetime as dt
from typing import Dict, Any, List

import requests
import pandas as pd

API_BASE = "https://api.opendota.com/api"
CONST_HEROES = "https://raw.githubusercontent.com/odota/dotaconstants/master/build/heroes.json"
CN_LOCALE_URL = "https://raw.githubusercontent.com/odota/dotaconstants/master/build/locales/zh.json"

# ====== 可选：自定义中文英雄名覆盖（键为 hero_id）======
# 建议只放少量你想要的队内叫法；官方中文名会自动加载作为兜底
HERO_CN_MAP: Dict[int, str] = {
    # 示例：
    # 17: "电狗",        # 覆盖风暴之灵
    # 120: "滚滚",       # 覆盖 Pangolier
    # 128: "老奶奶",     # 覆盖 Snapfire
}

def get(url, params=None, api_key=None, max_retries=3):
    headers = {"User-Agent": "bp-analyzer/1.0"}
    if api_key:
        if params is None: params = {}
        params["api_key"] = api_key
    for i in range(max_retries):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                raise RuntimeError(f"Non-JSON response from {url}")
        if r.status_code in (429, 502, 503, 504):
            time.sleep(1 + i)
            continue
        r.raise_for_status()
    r.raise_for_status()

def load_heroes() -> Dict[int, Dict[str, Any]]:
    data = requests.get(CONST_HEROES, timeout=30).json()
    out = {}
    for _, v in data.items():
        out[int(v["id"])] = v
    return out

def load_hero_cn_names_from_dotaconstants(timeout=30) -> Dict[int, str]:
    """
    从 dotaconstants 拉取完整中文英雄名映射：{hero_id: 中文名}
    """
    try:
        data = requests.get(CN_LOCALE_URL, timeout=timeout).json()
        heroes = data.get("heroes", {})
        out = {}
        for k, v in heroes.items():
            try:
                hid = int(k)
            except:
                continue
            name_cn = v.get("localized_name")
            if name_cn:
                out[hid] = name_cn
        return out
    except Exception:
        return {}

def fetch_team_matches(team_id: int, since: str = None, until: str = None, api_key=None) -> List[Dict[str, Any]]:
    url = f"{API_BASE}/teams/{team_id}/matches"
    params = {}
    if since:
        # teams/{id}/matches 接收 date=最近N天；这里把 since 转成天数传入
        t0 = int(dt.datetime.fromisoformat(since).timestamp())
        params["date"] = int((dt.datetime.utcnow() - dt.datetime.fromtimestamp(t0)).days)
    res = get(url, params=params, api_key=api_key)

    # 客户端精确过滤 since/until
    if since or until:
        out = []
        for m in res:
            st = m.get("start_time")
            if st is None:
                continue
            ok = True
            if since and st < int(dt.datetime.fromisoformat(since).timestamp()):
                ok = False
            if until and st > int(dt.datetime.fromisoformat(until).timestamp()):
                ok = False
            if ok:
                out.append(m)
        return out
    return res

def side_of(match: Dict[str, Any]) -> str:
    # teams/{id}/matches 的 'radiant' 是从该队视角：True=该队是天辉
    return "radiant" if match.get("radiant") else "dire"

def did_team_win(match: Dict[str, Any]) -> bool:
    rad_win = match.get("radiant_win")
    return bool(rad_win) if match.get("radiant") else not bool(rad_win)

def fetch_match_detail(match_id: int, api_key=None) -> Dict[str, Any]:
    url = f"{API_BASE}/matches/{match_id}"
    return get(url, api_key=api_key)

def normalize_action_rows(team_id: int,
                          match_summary: Dict[str, Any],
                          match_detail: Dict[str, Any],
                          heroes_map: Dict[int, Dict[str, Any]],
                          hero_name_map: Dict[int, str]) -> List[Dict[str, Any]]:
    out = []
    picks_bans = match_detail.get("picks_bans") or []

    # 对手名：我方为天辉则对手取 dire_team；否则 radiant_team
    opp = None
    opp_team = match_detail.get("dire_team") if match_summary.get("radiant") else match_detail.get("radiant_team")
    if isinstance(opp_team, dict):
        opp = opp_team.get("name")
    if not opp:
        opp = match_summary.get("opposing_team_name") or match_summary.get("opposing_team_id") or "UNKNOWN"

    for pb in picks_bans:
        hid = pb.get("hero_id")
        is_pick = bool(pb.get("is_pick"))
        t = pb.get("team")
        order = pb.get("order")
        side = "radiant" if t == 0 else "dire"

        # 中文名优先（官方中文 -> 你自定义覆盖），最后才回退英文
        hero_name = hero_name_map.get(hid)
        if not hero_name:
            hero_name = heroes_map.get(hid, {}).get("localized_name") or heroes_map.get(hid, {}).get("name") or str(hid)

        team_side = side_of(match_summary)
        team_str = "OurTeam" if side == team_side else "Opponent"
        action = "pick" if is_pick else "ban"

        row = {
            # 先用英文键，稍后统一翻译列名与枚举值
            "match_id": match_summary.get("match_id"),
            "start_time_utc": dt.datetime.utcfromtimestamp(match_summary.get("start_time", 0)).isoformat() + "Z",
            "leagueid": match_summary.get("leagueid"),
            "series_id": match_detail.get("series_id"),
            "game_mode": match_detail.get("game_mode"),
            "patch": match_detail.get("patch"),
            "duration": match_detail.get("duration"),
            "opponent": opp,
            "our_side": team_side,
            "result": "win" if did_team_win(match_summary) else "loss",
            "order": order if order is not None else -1,
            "team": team_str,
            "side": side,
            "action": action,
            "hero_id": hid,
            "hero": hero_name,
        }
        out.append(row)
    return out

def build_stats(actions_df: pd.DataFrame):
    # 该队自选与胜率（此时 hero 列已是中文名）
    picks = actions_df[(actions_df["team"] == "OurTeam") & (actions_df["action"] == "pick")]
    pick_stats = picks.groupby("hero", as_index=False).agg(
        picks=("hero", "count"),
        wins=("result", lambda s: (s == "win").sum()),
        losses=("result", lambda s: (s == "loss").sum()),
    )
    pick_stats["winrate"] = (pick_stats["wins"] / pick_stats["picks"].clip(lower=1)) * 100.0
    pick_stats = pick_stats.sort_values(["picks", "winrate"], ascending=[False, False])

    # 对手对你方的禁用统计（中文英雄名）
    opp_bans = actions_df[(actions_df["team"] == "Opponent") & (actions_df["action"] == "ban")]
    ban_stats = opp_bans.groupby("hero", as_index=False).agg(
        bans_vs_team=("hero", "count"),
        opponents=("opponent", pd.Series.nunique),
    ).sort_values(["bans_vs_team"], ascending=False)

    return pick_stats, ban_stats

# —— 中文本地化（列名与枚举）—— #
def _translate_values_actions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    map_side = {"radiant": "天辉", "dire": "夜魇"}
    map_team = {"OurTeam": "我方", "Opponent": "对手"}
    map_action = {"pick": "选择", "ban": "禁用"}
    map_result = {"win": "胜", "loss": "负"}

    if "our_side" in df.columns:
        df["our_side"] = df["our_side"].map(map_side).fillna(df["our_side"])
    if "side" in df.columns:
        df["side"] = df["side"].map(map_side).fillna(df["side"])
    if "team" in df.columns:
        df["team"] = df["team"].map(map_team).fillna(df["team"])
    if "action" in df.columns:
        df["action"] = df["action"].map(map_action).fillna(df["action"])
    if "result" in df.columns:
        df["result"] = df["result"].map(map_result).fillna(df["result"])
    return df

def _rename_columns_actions_zh(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "match_id": "比赛ID",
        "start_time_utc": "开始时间(UTC)",
        "leagueid": "联赛ID",
        "series_id": "系列赛ID",
        "game_mode": "游戏模式",
        "patch": "版本",
        "duration": "时长(秒)",
        "opponent": "对手",
        "our_side": "我方阵营",
        "result": "结果",
        "order": "顺位",
        "team": "出手方",
        "side": "该手阵营",
        "action": "动作",
        "hero_id": "英雄ID",
        "hero": "英雄",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

def _rename_columns_pickstats_zh(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "hero": "英雄",
        "picks": "选择次数",
        "wins": "胜",
        "losses": "负",
        "winrate": "胜率(%)",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

def _rename_columns_banstats_zh(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "hero": "英雄",
        "bans_vs_team": "被禁次数(对你)",
        "opponents": "不同对手数",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team_id", type=int, required=True)
    ap.add_argument("--since", type=str, default=None, help="ISO date, e.g. 2024-01-01")
    ap.add_argument("--until", type=str, default=None, help="ISO date, e.g. 2025-12-31")
    ap.add_argument("--out_dir", type=str, default="./out")
    ap.add_argument("--sleep", type=float, default=1.1, help="seconds between match detail calls (rate limiting)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    api_key = os.getenv("OPENDOTA_API_KEY", None)

    print("[1/5] 加载英文英雄常量 ...")
    heroes = load_heroes()

    print("[2/5] 拉取官方中文英雄名 ...")
    cn_official = load_hero_cn_names_from_dotaconstants()
    if cn_official:
        print(f"    已加载官方中文英雄名：{len(cn_official)} 个")
    else:
        print("    [warn] 加载官方中文英雄名失败，将仅使用自定义覆盖/英文回退")

    # 合并：官方中文为底 -> 你自定义覆盖优先
    hero_name_map = dict(cn_official)
    hero_name_map.update(HERO_CN_MAP)
    # 小自检（可注释）
    for test_id in (128, 114, 74, 73):  # Snapfire, Monkey King, Invoker, Alchemist
        if test_id in hero_name_map:
            print(f"    映射检查 hero_id={test_id} -> {hero_name_map[test_id]}")

    print("[3/5] 拉取队伍比赛列表 ...")
    matches = fetch_team_matches(args.team_id, since=args.since, until=args.until, api_key=api_key)
    if not matches:
        print("在指定时间范围内未找到比赛。", file=sys.stderr)
        sys.exit(1)

    msumm = []
    for m in matches:
        msumm.append({
            "match_id": m.get("match_id"),
            "start_time": m.get("start_time"),
            "radiant": m.get("radiant"),
            "radiant_win": m.get("radiant_win"),
            "leagueid": m.get("leagueid"),
            "opposing_team_id": m.get("opposing_team_id"),
            "opposing_team_name": m.get("opposing_team_name"),
        })

    print(f"[4/5] 获取 {len(msumm)} 场比赛的详情（含 ban/pick） ...")
    all_actions = []
    for i, m in enumerate(msumm, 1):
        mid = m["match_id"]
        try:
            detail = fetch_match_detail(mid, api_key=api_key)
            rows = normalize_action_rows(args.team_id, m, detail, heroes, hero_name_map)
            all_actions.extend(rows)
        except Exception as e:
            print(f"[warn] match {mid} failed: {e}", file=sys.stderr)
        time.sleep(args.sleep)

    if not all_actions:
        print("未获取到 ban/pick（可能回放未解析）。", file=sys.stderr)
        sys.exit(2)

    actions_df = pd.DataFrame(all_actions).sort_values(["start_time_utc", "match_id", "order"])

    # —— 仅导出中文 JSON —— #
    # 1) 动作明细（中文字段/枚举/英雄名已中文）
    actions_df_zh = _translate_values_actions(actions_df)
    actions_df_zh = _rename_columns_actions_zh(actions_df_zh)
    actions_json = os.path.join(args.out_dir, "team_actions.json")
    actions_df_zh.to_json(actions_json, orient="records", force_ascii=False, indent=2)

    print("[5/5] 计算统计并导出 ...")
    pick_stats, ban_stats = build_stats(actions_df)
    pick_stats_zh = _rename_columns_pickstats_zh(pick_stats)
    ban_stats_zh = _rename_columns_banstats_zh(ban_stats)

    pick_json = os.path.join(args.out_dir, "team_hero_pick_stats.json")
    ban_json = os.path.join(args.out_dir, "opponents_ban_vs_team.json")
    pick_stats_zh.to_json(pick_json, orient="records", force_ascii=False, indent=2)
    ban_stats_zh.to_json(ban_json, orient="records", force_ascii=False, indent=2)

    # 清单（中文）
    manifest = {
        "生成时间(UTC)": dt.datetime.utcnow().isoformat() + "Z",
        "队伍ID": args.team_id,
        "开始日期": args.since,
        "结束日期": args.until,
        "输出目录": os.path.abspath(args.out_dir),
        "文件": {
            "team_actions.json": actions_json,
            "team_hero_pick_stats.json": pick_json,
            "opponents_ban_vs_team.json": ban_json
        }
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n完成。中文 JSON 已生成于:", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()
