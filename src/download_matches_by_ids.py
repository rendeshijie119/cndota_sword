#!/usr/bin/env python3
"""
download_matches_by_ids.py

Download match JSONs for a list of match IDs, with optional grouping into
per-team folders.

Enhancement:
- Accept a directory of match id files (--ids-dir) in addition to a single
  --ids-file. When --ids-dir is provided the script will read all files
  matching --ids-pattern (default '*.txt') and gather ids from all of them.
  Duplicates are removed automatically.
- Supports recursive search in --ids-dir via --recursive flag.
- Behavior otherwise unchanged: supports --group-by-team, --skip-existing, per-match
  saving under <out_dir>/all and optional copies under <out_dir>/teams/<team>/.

Usage examples:
  python download_matches_by_ids.py --ids-file ids.txt --out-dir ./matches
  python download_matches_by_ids.py --ids-dir ./matchid_lists --out-dir ./matches
  python download_matches_by_ids.py --ids-dir ./matchid_lists --ids-pattern "*.matchids" --recursive --out-dir ./matches
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

DEFAULT_URL_TEMPLATE = "https://api.opendota.com/api/matches/{match_id}"


def safe_name(s: Optional[str], max_len: int = 120) -> str:
    if s is None:
        return "unknown"
    s2 = str(s).strip()
    s2 = "".join(ch if (ch.isalnum() or ch in " ._-") else "_" for ch in s2)
    s2 = s2.strip(" _-.")
    if not s2:
        return "unknown"
    if len(s2) > max_len:
        s2 = s2[: max_len - 3] + "..."
    return s2


def parse_ids_file(path: Path) -> List[int]:
    ids: List[int] = []
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            for token in ln.replace(",", " ").split():
                try:
                    ids.append(int(token))
                except ValueError:
                    print(f"Warning: skipping invalid match id token '{token}' in {path}")
    return ids


def parse_ids_from_dir(directory: Path, pattern: str = "*.txt", recursive: bool = False) -> List[int]:
    """
    Scan directory for files matching pattern and parse ids from each file.
    Returns combined list (may contain duplicates — caller should dedupe if needed).
    """
    ids: List[int] = []
    if not directory.exists() or not directory.is_dir():
        return ids
    if recursive:
        files = sorted(directory.rglob(pattern))
    else:
        files = sorted(directory.glob(pattern))
    for f in files:
        if not f.is_file():
            continue
        ids_from_file = parse_ids_file(f)
        if ids_from_file:
            print(f"Read {len(ids_from_file)} ids from {f}")
        ids.extend(ids_from_file)
    return ids


def get_team_names_from_match(match_json: dict) -> Tuple[str, str]:
    rad = match_json.get("radiant_name") or ""
    dire = match_json.get("dire_name") or ""
    if not rad or not dire:
        rt = match_json.get("radiant_team") or match_json.get("radiant_team_info") or {}
        dt = match_json.get("dire_team") or match_json.get("dire_team_info") or {}
        if isinstance(rt, dict):
            rad = rad or rt.get("name") or rt.get("team_name") or rt.get("tag") or ""
        if isinstance(dt, dict):
            dire = dire or dt.get("name") or dt.get("team_name") or dt.get("tag") or ""
    if not rad:
        rid = None
        if isinstance(match_json.get("radiant_team"), dict):
            rid = match_json.get("radiant_team").get("team_id")
        rid = rid or match_json.get("radiant_team_id")
        rad = rad or (f"team_{rid}" if rid else "unknown_radiant")
    if not dire:
        did = None
        if isinstance(match_json.get("dire_team"), dict):
            did = match_json.get("dire_team").get("team_id")
        did = did or match_json.get("dire_team_id")
        dire = dire or (f"team_{did}" if did else "unknown_dire")
    return str(rad), str(dire)


def download_match_json(
    session: requests.Session,
    match_id: int,
    url_template: str,
    timeout: int,
    max_retries: int,
    sleep_between_attempts: float,
    verbose: bool = False,
) -> Tuple[int, Optional[dict], Optional[str]]:
    url = url_template.format(match_id=match_id)
    attempt = 0
    backoff = 1.0
    last_status = None
    while attempt <= max_retries:
        try:
            if verbose:
                print(f"[{match_id}] GET {url} (attempt {attempt+1})")
            resp = session.get(url, timeout=timeout)
        except requests.RequestException as e:
            attempt += 1
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue

        status = resp.status_code
        last_status = status
        if status == 200:
            try:
                return match_id, resp.json(), None
            except Exception as e:
                return match_id, None, f"json parse error: {e}"
        if status == 429:
            ra = resp.headers.get("Retry-After")
            wait = None
            try:
                if ra is not None:
                    wait = float(ra)
            except Exception:
                wait = None
            if wait is None:
                wait = backoff
            if verbose:
                print(f"[{match_id}] HTTP 429, sleeping {wait:.1f}s (attempt {attempt+1})")
            time.sleep(wait)
            attempt += 1
            backoff = min(backoff * 2, 60.0)
            continue
        if 500 <= status < 600:
            if verbose:
                print(f"[{match_id}] server error {status}, retrying after {backoff:.1f}s")
            time.sleep(backoff)
            attempt += 1
            backoff = min(backoff * 2, 60.0)
            continue

        # other 4xx (client error) - don't retry
        try:
            text = resp.text[:400].replace("\n", " ")
        except Exception:
            text = ""
        return match_id, None, f"HTTP error {status}: {text}"

    return match_id, None, f"max_retries_exceeded (last status {last_status if last_status is not None else 'n/a'})"


def save_match_file(base_out: Path, match_id: int, match_json: dict, group_by_team: bool, verbose: bool = False) -> Tuple[str, List[str]]:
    all_dir = base_out / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    all_path = all_dir / f"match_{match_id}.json"
    with all_path.open("w", encoding="utf-8") as fh:
        json.dump(match_json, fh, indent=2, ensure_ascii=False)

    team_paths: List[str] = []
    if group_by_team:
        rad_name, dire_name = get_team_names_from_match(match_json)
        team_names = [rad_name, dire_name]
        # dedupe identical names
        seen = set()
        for name in team_names:
            if not name:
                continue
            safe = safe_name(name)
            if safe in seen:
                continue
            seen.add(safe)
            team_dir = base_out / "teams" / safe
            team_dir.mkdir(parents=True, exist_ok=True)
            team_path = team_dir / f"match_{match_id}.json"
            try:
                with team_path.open("w", encoding="utf-8") as fh:
                    json.dump(match_json, fh, indent=2, ensure_ascii=False)
                team_paths.append(str(team_path))
            except Exception as e:
                if verbose:
                    print(f"[{match_id}] failed to write team copy to {team_path}: {e}")

    if verbose:
        print(f"[{match_id}] saved to {all_path} (+{len(team_paths)} team copies)")
    return str(all_path), team_paths


def create_team_copies_from_existing(base_out: Path, match_id: int, group_by_team: bool, verbose: bool = False) -> Tuple[str, List[str]]:
    """
    Given that canonical file exists at base_out/all/match_<id>.json, create any missing
    team copies under base_out/teams/<team_name>/match_<id>.json. Returns (all_path, created_team_paths).
    """
    all_path = base_out / "all" / f"match_{match_id}.json"
    if not all_path.exists():
        raise FileNotFoundError(all_path)
    created = []
    if group_by_team:
        try:
            with all_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            if verbose:
                print(f"[{match_id}] failed to read existing canonical JSON: {e}")
            return str(all_path), created
        rad_name, dire_name = get_team_names_from_match(data)
        team_names = [rad_name, dire_name]
        seen = set()
        for name in team_names:
            if not name:
                continue
            safe = safe_name(name)
            if safe in seen:
                continue
            seen.add(safe)
            team_dir = base_out / "teams" / safe
            team_dir.mkdir(parents=True, exist_ok=True)
            team_path = team_dir / f"match_{match_id}.json"
            if not team_path.exists():
                try:
                    with team_path.open("w", encoding="utf-8") as fh:
                        json.dump(data, fh, indent=2, ensure_ascii=False)
                    created.append(str(team_path))
                except Exception as e:
                    if verbose:
                        print(f"[{match_id}] failed to write team copy to {team_path}: {e}")
    return str(all_path), created


def gather_match_ids(ids_file: Optional[Path], ids_dir: Optional[Path], ids_pattern: str, recursive: bool, verbose: bool) -> List[int]:
    """
    Gather match ids from either a single file or many files in a directory.
    Deduplicates and returns a sorted list.
    """
    ids_set: Set[int] = set()
    if ids_file:
        if not ids_file.exists():
            raise FileNotFoundError(f"IDs file not found: {ids_file}")
        ids_from_file = parse_ids_file(ids_file)
        if verbose:
            print(f"Read {len(ids_from_file)} ids from {ids_file}")
        ids_set.update(ids_from_file)
    if ids_dir:
        ids_from_dir = parse_ids_from_dir(ids_dir, pattern=ids_pattern, recursive=recursive)
        if verbose:
            print(f"Read total {len(ids_from_dir)} ids from files under {ids_dir} (pattern={ids_pattern})")
        ids_set.update(ids_from_dir)
    ids_list = sorted(ids_set)
    return ids_list


def main() -> int:
    p = argparse.ArgumentParser(description="Download match JSONs by IDs and optionally group by team.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids-file", type=Path, help="Text file with match ids (one per line, supports commas).")
    group.add_argument("--ids-dir", type=Path, help="Directory containing match id files (will read files matching --ids-pattern).")
    p.add_argument("--ids-pattern", type=str, default="*.txt", help="Glob pattern for files inside --ids-dir (default '*.txt').")
    p.add_argument("--recursive", action="store_true", help="When --ids-dir is used, search recursively (rglob).")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory to save matches.")
    p.add_argument("--url-template", type=str, default=DEFAULT_URL_TEMPLATE, help="URL template; use {match_id} placeholder.")
    p.add_argument("--workers", type=int, default=8, help="Number of parallel workers (lower if you hit rate limits).")
    p.add_argument("--timeout", type=int, default=30, help="Per-request timeout in seconds.")
    p.add_argument("--max-retries", type=int, default=6, help="Per-request retry attempts (for network/5xx/429).")
    p.add_argument("--sleep-between", type=float, default=0.05, help="Sleep (s) between scheduling requests (helps rate-limit).")
    p.add_argument("--group-by-team", action="store_true", help="Also save a copy into per-team folders under out-dir/teams/<team_name>.")
    p.add_argument("--skip-existing", action="store_true", help="Skip downloading if canonical file exists; optionally create missing team copies from it.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    try:
        match_ids = gather_match_ids(args.ids_file, args.ids_dir, args.ids_pattern, args.recursive, args.verbose)
    except Exception as e:
        print(f"Error gathering ids: {e}")
        return 2

    if not match_ids:
        print("No match ids found.")
        return 2

    print(f"Total distinct match ids to consider: {len(match_ids)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=max(10, args.workers))
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    results_success: Dict[int, str] = {}
    results_failed: Dict[int, str] = {}
    match_to_teams: Dict[str, List[str]] = {}

    downloaded_ids: List[int] = []
    skipped_ids: List[int] = []

    # decide which matches actually need downloading
    to_download: List[int] = []
    for mid in match_ids:
        canonical = args.out_dir / "all" / f"match_{mid}.json"
        if args.skip_existing and canonical.exists():
            # canonical exists; skip download and (if requested) create team copies if missing
            try:
                all_path, created_team = create_team_copies_from_existing(args.out_dir, mid, args.group_by_team, args.verbose)
                # read team names if possible for the mapping
                rad_name = dire_name = ""
                try:
                    with (args.out_dir / "all" / f"match_{mid}.json").open("r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    rad_name, dire_name = get_team_names_from_match(data)
                except Exception:
                    pass
                match_to_teams[str(mid)] = [rad_name, dire_name] if (rad_name or dire_name) else []
                results_success[mid] = str(canonical)
                skipped_ids.append(mid)

                # Print debug for skipped: show canonical and any team copies created (or existing)
                team_copy_count = len(created_team)
                if team_copy_count:
                    print(f"[{mid}] skipped (existing) -> {all_path} (+{team_copy_count} team copy(s) created)")
                    for pth in created_team:
                        print(f"    created: {pth}")
                else:
                    # If no team copies created we still want to ensure team folders exist (maybe they already do)
                    print(f"[{mid}] skipped (existing) -> {all_path} (no new team copies created)")
            except Exception as e:
                # reading existing failed; fall back to downloading
                if args.verbose:
                    print(f"[{mid}] failed to use existing canonical ({e}), will download")
                to_download.append(mid)
        else:
            to_download.append(mid)

    print(f"Downloading {len(to_download)} matches -> {args.out_dir} using template {args.url_template}")
    # download only those in to_download
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        future_map = {}
        for mid in to_download:
            time.sleep(args.sleep_between)
            fut = ex.submit(
                download_match_json,
                session,
                mid,
                args.url_template,
                args.timeout,
                args.max_retries,
                args.sleep_between,
                args.verbose,
            )
            future_map[fut] = mid

        for fut in as_completed(future_map):
            mid = future_map[fut]
            try:
                match_id, data, err = fut.result()
            except Exception as e:
                print(f"[{mid}] unexpected exception: {e}")
                results_failed[mid] = f"exception: {e}"
                continue

            if data is None:
                msg = err or "unknown error"
                print(f"[{mid}] {msg}; skipping.")
                results_failed[mid] = msg
                continue

            try:
                all_path, team_paths = save_match_file(args.out_dir, match_id, data, args.group_by_team, args.verbose)
            except Exception as e:
                print(f"[{match_id}] failed to save JSON: {e}")
                results_failed[match_id] = f"save error: {e}"
                continue

            rad_name, dire_name = get_team_names_from_match(data)
            match_to_teams[str(match_id)] = [rad_name, dire_name]
            results_success[match_id] = all_path
            downloaded_ids.append(match_id)

            # Debug output: print canonical path and any team copies written
            print(f"[{match_id}] downloaded and saved -> {all_path} (+{len(team_paths)} team copy(s))")
            if team_paths:
                for pth in team_paths:
                    print(f"    wrote: {pth}")

    # write mapping file
    map_path = args.out_dir / "match_to_teams.json"
    try:
        with map_path.open("w", encoding="utf-8") as fh:
            json.dump(match_to_teams, fh, indent=2, ensure_ascii=False)
        print(f"Wrote match->teams map to {map_path}")
    except Exception as e:
        print(f"Warning: failed to write match_to_teams.json: {e}")

    # summary counts and lists
    downloaded_count = len(downloaded_ids)
    skipped_count = len(skipped_ids)
    failed_count = len(results_failed)
    print(f"Summary: downloaded={downloaded_count}, skipped_existing={skipped_count}, failed={failed_count}")
    if downloaded_count:
        print("Downloaded match ids (showing up to 50):")
        for mid in downloaded_ids[:50]:
            print(f"  [{mid}]")
    if skipped_count:
        print("Skipped (existing) match ids (showing up to 50):")
        for mid in skipped_ids[:50]:
            print(f"  [{mid}]")
    if failed_count:
        print("Failures (showing up to 50):")
        for i, (mid, reason) in enumerate(sorted(results_failed.items())[:50]):
            print(f"  [{mid}] {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())