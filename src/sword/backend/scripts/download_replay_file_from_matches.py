#!/usr/bin/env python3
"""
download_replay_file_from_matches.py

Scan match JSON files (default ./data/matches), extract replay download URLs (or attempt to
construct them from cluster/replay_salt), download .dem/.dem.bz2 files into ./data/replay/all and
(optionally) copy them into team folders under ./data/replay/teams/<team_name>/.

Features added (per request):
- --group-by-team : if set, the downloaded replay will be copied into per-team folders.
- Filenames include match date (YYYYMMDD) and league name (sanitized) when available.
- Canonical replay is saved under out_dir/all/, team copies under out_dir/teams/<team_name>/.
- Console summary printed at the end (counts + lists).

Usage example:
  python3 download_replay_file_from_matches.py \
    --matches-dir data/matches --out-dir data/replay --workers 6 --skip-existing --group-by-team --verbose
"""
from __future__ import annotations
import argparse
import logging
import time
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Dict
import requests
import bz2
from urllib.parse import urlparse, unquote
from datetime import datetime

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_replay_file_from_matches")

DEFAULT_MATCHES_DIR = Path("data/matches")
DEFAULT_OUT_DIR = Path("data/replay")
DEFAULT_WORKERS = 4
DEFAULT_RETRIES = 3
DEFAULT_DELAY = 0.5  # seconds between downloads (politeness)
CHUNK_SIZE = 1024 * 1024  # 1MB


# ----------------- helpers -----------------
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


def find_match_json_files(matches_dir: Path) -> List[Path]:
    files: List[Path] = []
    if not matches_dir.exists():
        return files
    for p in matches_dir.rglob("*.json"):
        files.append(p)
    return sorted(files)


def infer_match_id(match_data: dict, filename: Optional[str] = None) -> Optional[int]:
    mid = match_data.get("match_id") or match_data.get("matchid") or match_data.get("matchId")
    try:
        if mid is not None:
            return int(mid)
    except Exception:
        pass
    if filename:
        fn = Path(filename).name
        m = __import__("re").search(r"(\d{6,})", fn)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def extract_replay_url_candidates(match_data: dict) -> List[str]:
    candidates: List[str] = []
    for key in ("replay_url", "replay_url_s3", "replay_url_https", "replay_url_steam", "replay_url_http"):
        v = match_data.get(key)
        if v and isinstance(v, str):
            candidates.append(v)
    # shallow scan for strings containing ".dem" or "replay"
    if isinstance(match_data, dict):
        for k, vv in match_data.items():
            if isinstance(vv, str) and (".dem" in vv or "replay" in vv):
                candidates.append(vv)
    # dedupe preserving order
    seen = set()
    out = []
    for c in candidates:
        if not c:
            continue
        s = str(c).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def construct_valve_replay_urls(match_id: Optional[int], match_data: dict) -> List[str]:
    candidates = []
    if match_id is None:
        return candidates
    salt = match_data.get("replay_salt") or match_data.get("replaySalt") or match_data.get("replay_salt64")
    cluster = match_data.get("cluster") or match_data.get("replay_cluster") or match_data.get("cluster_id")
    if salt is None:
        return candidates
    salt_s = str(salt)
    cluster_s = str(cluster) if cluster is not None else ""
    if cluster_s:
        for proto in ("https", "http"):
            candidates.append(f"{proto}://replay{cluster_s}.valve.net/570/{match_id}_{salt_s}.dem.bz2")
            candidates.append(f"{proto}://replay{cluster_s}.valve.net/570/{match_id}_{salt_s}.dem")
    for proto in ("https", "http"):
        candidates.append(f"{proto}://replay.valve.net/570/{match_id}_{salt_s}.dem.bz2")
        candidates.append(f"{proto}://replay.valve.net/570/{match_id}_{salt_s}.dem")
    # dedupe
    seen = set(); out = []
    for c in candidates:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def build_filename_base(match_id: Optional[int], match_data: dict, include_league: bool = True) -> Tuple[str, Dict[str, str]]:
    """
    Construct a canonical filename base (without extension) using match_id, date, league.
    Returns (basename, meta) where meta contains parsed fields (date_str, league, radiant_name, dire_name).
    """
    # date
    start_time = match_data.get("start_time") or match_data.get("starttime") or match_data.get("startTime")
    date_str = ""
    try:
        if start_time:
            # start_time may be seconds since epoch (int)
            t = int(start_time)
            date_str = datetime.utcfromtimestamp(t).strftime("%Y%m%d")
    except Exception:
        date_str = ""
    # league
    league_obj = match_data.get("league") or {}
    league_name = ""
    if isinstance(league_obj, dict):
        league_name = league_obj.get("name") or ""
    elif isinstance(league_obj, str):
        league_name = league_obj
    # safe values
    league_safe = safe_name(league_name).replace(" ", "_") if league_name else "unknown_league"
    date_part = date_str if date_str else "unknown_date"
    mid = str(match_id) if match_id is not None else "unknown_match"
    base = f"match_{mid}_{date_part}"
    if include_league:
        base = f"{base}_{league_safe}"
    # collect team meta
    team_meta = get_team_meta(match_data)
    return base, {"date": date_str, "league": league_name, "team_meta": team_meta}


def choose_canonical_path(all_dir: Path, basename: str) -> Path:
    all_dir.mkdir(parents=True, exist_ok=True)
    return all_dir / f"{basename}.dem"


def stream_download_to_file(url: str, dest: Path, retries: int = 3, timeout: int = 30) -> bool:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            logger.debug("Downloading %s (attempt %d)", url, attempt)
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(dest)
            logger.debug("Saved stream to %s", dest)
            return True
        except Exception as e:
            logger.warning("Attempt %d failed for %s: %s", attempt, url, e)
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            if attempt < retries:
                time.sleep(1.0 * attempt)
    logger.error("Failed to download %s after %d attempts", url, retries)
    return False


def download_and_decompress_bz2(url: str, out_dem: Path, retries: int = 3, timeout: int = 30) -> bool:
    """
    Download .bz2 to temporary file and decompress into out_dem (atomic replace).
    """
    tmp_bz2 = out_dem.with_suffix(out_dem.suffix + ".bz2.tmp")
    tmp_dem = out_dem.with_suffix(out_dem.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            logger.debug("GET (bz2) %s (attempt %d)", url, attempt)
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with tmp_bz2.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
            # decompress
            logger.debug("Decompressing %s -> %s", tmp_bz2, tmp_dem)
            with tmp_bz2.open("rb") as fin, tmp_dem.open("wb") as fout:
                decompressor = bz2.BZ2Decompressor()
                while True:
                    chunk = fin.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    try:
                        out_chunk = decompressor.decompress(chunk)
                    except Exception:
                        # fallback full-file decompress
                        fin.close()
                        tmp_dem.unlink(missing_ok=True)
                        with tmp_bz2.open("rb") as fin2, tmp_dem.open("wb") as fout2:
                            data = fin2.read()
                            fout2.write(bz2.decompress(data))
                        break
                    if out_chunk:
                        fout.write(out_chunk)
            tmp_bz2.unlink(missing_ok=True)
            tmp_dem.replace(out_dem)
            logger.info("Downloaded & decompressed -> %s", out_dem)
            return True
        except Exception as e:
            logger.warning("Attempt %d failed for %s: %s", attempt, url, e)
            try:
                tmp_bz2.unlink(missing_ok=True)
                tmp_dem.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt < retries:
                time.sleep(1.0 * attempt)
    logger.error("Giving up downloading/decompressing %s", url)
    return False


# ----------------- core per-file processing -----------------
def process_match_file_download(match_file: Path, out_dir: Path, skip_existing: bool, retries: int,
                                group_by_team: bool, verbose: bool = False, delay: float = 0.0) -> Tuple[Optional[int], str, bool, Optional[Path]]:
    """
    Returns (match_id, canonical_basename, did_download_bool, canonical_path_or_None)
    canonical_basename is the base used for naming (without extension)
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    try:
        with match_file.open("r", encoding="utf-8") as fh:
            match_data = json.load(fh)
    except Exception as e:
        logger.warning("Failed to load JSON %s: %s", match_file, e)
        return (None, "", False, None)

    match_id = infer_match_id(match_data, filename=str(match_file))
    # prepare basename (date + league)
    basename, meta = build_filename_base(match_id, match_data, include_league=True)
    all_dir = out_dir / "all"
    canonical_path = choose_canonical_path(all_dir, basename)

    # if skip and exists, still may need to create team copies if requested -> return path so caller can handle
    if skip_existing and canonical_path.exists():
        logger.info("[%s] skip existing canonical -> %s", match_id or match_file.name, canonical_path)
        # create team copies if needed (caller will inspect did_download False but path present)
        return (match_id, basename, False, canonical_path)

    # determine replay URL candidates
    candidates = extract_replay_url_candidates(match_data)
    constructed = construct_valve_replay_urls(match_id, match_data)
    for c in constructed:
        if c not in candidates:
            candidates.append(c)

    if not candidates:
        logger.info("[%s] no replay URL candidates found", match_id or match_file.name)
        return (match_id, basename, False, None)

    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded_ok = False
    for url in candidates:
        if url.startswith("//"):
            url = "https:" + url
        if url.startswith("replay") and not url.lower().startswith("http"):
            url = "https://" + url
        if not url.lower().startswith("http"):
            logger.debug("Skipping invalid candidate: %s", url)
            continue

        # Use the canonical_path as the primary download target (we will copy to team folders later)
        if url.lower().endswith(".bz2") or ".dem.bz2" in url.lower():
            ok = download_and_decompress_bz2(url, canonical_path, retries=retries)
        else:
            # stream to canonical path
            ok = stream_download_to_file(url, canonical_path, retries=retries)
            if ok:
                # detect BZ2 magic even if extension missing
                try:
                    with canonical_path.open("rb") as fh:
                        hdr = fh.read(3)
                    if hdr.startswith(b'BZh'):
                        # move and decompress local file
                        tmp_bz2 = canonical_path.with_suffix(canonical_path.suffix + ".bz2.tmp")
                        canonical_path.replace(tmp_bz2)
                        try:
                            with tmp_bz2.open("rb") as fin, canonical_path.open("wb") as fout:
                                data = fin.read()
                                fout.write(bz2.decompress(data))
                            tmp_bz2.unlink(missing_ok=True)
                            ok = True
                        except Exception as e:
                            logger.warning("Fallback local decompress failed: %s", e)
                            ok = False
                except Exception:
                    pass

        if ok:
            downloaded_ok = True
            logger.info("[%s] downloaded canonical -> %s", match_id or match_file.name, canonical_path)
            if delay and delay > 0:
                time.sleep(delay)
            break
        else:
            logger.debug("Candidate failed for match %s: %s", match_id or match_file.name, url)
            continue

    if not downloaded_ok:
        logger.info("[%s] none of the candidates succeeded", match_id or match_file.name)
        return (match_id, basename, False, None)

    # if we reach here, canonical_path exists and is the downloaded .dem
    # If group_by_team requested, create team folder copies with team-specific filenames
    team_meta = meta.get("team_meta", {})
    copied_paths = []
    if group_by_team and isinstance(team_meta, dict):
        teams_dir = out_dir / "teams"
        # create copy per team (radiant and dire)
        for side in ("radiant", "dire"):
            tname = team_meta.get(side, {}).get("team_name") or ""
            safe_team = safe_name(tname).replace(" ", "_") if tname else f"{side}"
            team_dir = teams_dir / safe_team
            try:
                team_dir.mkdir(parents=True, exist_ok=True)
                team_basename = f"{basename}_{safe_team}"
                team_path = team_dir / f"{team_basename}.dem"
                # copy canonical file (overwrite existing to keep latest)
                shutil.copy2(canonical_path, team_path)
                copied_paths.append(team_path)
                logger.info("[%s] copied to team folder -> %s", match_id or match_file.name, team_path)
            except Exception as e:
                logger.warning("Failed to copy to team folder %s: %s", team_dir, e)

    return (match_id, basename, True, canonical_path)


# ----------------- CLI and main -----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download replay files (.dem/.dem.bz2) from local match JSON files and group by team.")
    p.add_argument("--matches-dir", type=Path, default=DEFAULT_MATCHES_DIR, help="Folder containing match JSONs (default ./data/matches)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Folder to save replays (default ./data/replay)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel download workers")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries per candidate URL")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay (s) between downloads to be polite")
    p.add_argument("--skip-existing", action="store_true", help="Skip if canonical output file already exists")
    p.add_argument("--group-by-team", action="store_true", help="Also copy canonical replay into per-team folders under out-dir/teams/<team_name>.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging / debug")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    matches_dir: Path = args.matches_dir
    out_dir: Path = args.out_dir
    workers: int = max(1, int(args.workers))
    retries: int = max(1, int(args.retries))
    delay: float = float(args.delay)
    skip_existing: bool = args.skip_existing
    group_by_team: bool = args.group_by_team

    logger.info("Scanning matches dir: %s", matches_dir)
    files = find_match_json_files(matches_dir)
    if not files:
        logger.error("No match JSON files found under %s", matches_dir)
        return 2

    downloaded = 0
    skipped = 0
    failed = 0
    saved_paths: List[Path] = []

    downloaded_list: List[str] = []
    skipped_list: List[str] = []
    failed_list: List[str] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(process_match_file_download, f, out_dir, skip_existing, retries, group_by_team, args.verbose, delay): f
            for f in files
        }
        for fut in as_completed(future_map):
            match_file = future_map[fut]
            try:
                mid, basename, did_download, path = fut.result()
            except Exception as e:
                logger.warning("Worker failed for %s: %s", match_file, e)
                failed += 1
                failed_list.append(str(match_file))
                continue

            if did_download:
                downloaded += 1
                if path:
                    saved_paths.append(path)
                    downloaded_list.append(f"{mid or 'unknown'}\t{path}")
                logger.info("[%s] downloaded -> %s", mid or match_file.name, path or "<unknown>")
            else:
                # if path provided and exists, treated as skipped existing
                if path and path.exists():
                    skipped += 1
                    skipped_list.append(f"{mid or 'unknown'}\t{path}")
                    logger.info("[%s] skipped existing -> %s", mid or match_file.name, path)
                else:
                    failed += 1
                    failed_list.append(str(match_file))
                    logger.info("[%s] not downloaded", mid or match_file.name)

    # final summary logging
    logger.info("Done. downloaded=%d, skipped_existing=%d, failed=%d, saved=%d", downloaded, skipped, failed, len(saved_paths))

    # console output: summary + lists
    summary = {
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": failed,
        "saved": len(saved_paths),
        "downloaded_list_count": len(downloaded_list),
        "skipped_list_count": len(skipped_list),
        "failed_list_count": len(failed_list),
    }
    print("\n===== Replay download summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("")

    if downloaded_list:
        print("Downloaded matches (match_id \t canonical path):")
        for line in downloaded_list:
            print(line)
        print("")

    if skipped_list:
        print("Skipped (existing) matches (match_id \t canonical path):")
        for line in skipped_list:
            print(line)
        print("")

    if failed_list:
        print("Failed matches (match JSON path):")
        for line in failed_list:
            print(line)
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())