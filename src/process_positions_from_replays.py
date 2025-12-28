#!/usr/bin/env python3
"""
process_positions_from_replays.py

Scan for .dem files (single file, directory, or directory tree) and run the
clarity-parser fat-jar to produce a .csv for each .dem.

Behavior:
 - By default, finds all .dem files under --input-dir (recursive) and writes CSVs
   next to each .dem with the same basename and .csv extension.
 - You can instead specify individual files with --files.
 - Optionally mirror results into a separate --out-dir while preserving relative paths.
 - Uses a ThreadPoolExecutor to run multiple Java processes concurrently (bounded by --workers).
 - Prints a summary at the end and exits with non-zero when critical failures occur
   unless --continue-on-error is set.

Requirements:
 - Java runtime available as `java`
 - clarity-parser fat-jar built and available (default: target/clarity-parser-0.1.0-with-deps.jar)

Examples:
  # Process all replays under data/replay
  python process_positions_from_replays.py --input-dir data/replay --workers 2 --process

  # Process a single replay file (explicit)
  python process_positions_from_replays.py --files data/replay/all/match_12345_20220101_unknown_league.dem

  # Process and mirror outputs into out/csv_mirror directory
  python process_positions_from_replays.py --input-dir data/replay --out-dir out/csv_mirror --process --workers 3

  # Use custom jar, limit to first 600 seconds, skip existing CSVs
  python process_positions_from_replays.py --input-dir data/replay --jar target/clarity-parser-0.1.0-with-deps.jar --max-seconds 600 --skip-existing
"""
from __future__ import annotations
import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

LOG = logging.getLogger("process_positions_from_replays")
DEFAULT_JAR = Path("target/clarity-parser-0.1.0-with-deps.jar")


def find_dem_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {input_dir}")
    pattern = "**/*.dem" if recursive else "*.dem"
    return sorted([p for p in input_dir.glob(pattern) if p.is_file()])


def build_csv_path(dem: Path, out_dir: Optional[Path]) -> Path:
    csv_name = dem.with_suffix(".csv").name
    if out_dir is None:
        return dem.with_suffix(".csv")
    # preserve relative structure
    rel = dem.resolve().relative_to(Path.cwd().resolve()) if dem.is_relative_to(Path.cwd()) else dem.relative_to(dem.anchor)
    # if input is absolute and not under cwd, just mirror using dem.name under out_dir
    try:
        rel = dem.relative_to(Path.cwd())
    except Exception:
        rel = Path(dem.name)
    out_path = out_dir / rel.parent / csv_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def find_jar(jar_arg: Optional[str]) -> Path:
    if jar_arg:
        p = Path(jar_arg)
        if not p.exists():
            raise FileNotFoundError(f"Specified jar not found: {p}")
        return p
    if DEFAULT_JAR.exists():
        return DEFAULT_JAR
    raise FileNotFoundError(f"Jar not found at default {DEFAULT_JAR}. Build it or pass --jar")


def run_extractor_for_file(jar: Path, dem: Path, csv: Path, max_seconds: Optional[int], java_opts: str, timeout: int) -> Tuple[Path, bool, int, str]:
    """
    Run: java {java_opts} -jar {jar} {dem} {csv} [max_seconds]
    Returns (dem_path, success_bool, returncode, stderr_or_stdout_excerpt)
    """
    cmd = ["java"]
    if java_opts:
        cmd += java_opts.split()
    cmd += ["-jar", str(jar), str(dem), str(csv)]
    if max_seconds is not None:
        cmd.append(str(int(max_seconds)))
    LOG.debug("Running extractor: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        msg = f"TIMEOUT after {timeout}s"
        LOG.error("%s: %s", dem, msg)
        return dem, False, 124, msg
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        LOG.error("Extractor failed for %s (rc=%d). stderr: %s", dem, proc.returncode, err[:1000])
        return dem, False, proc.returncode, err[:1000] or out[:1000]
    LOG.info("Extractor OK -> %s", csv)
    return dem, True, proc.returncode, out[:1000] or err[:1000]


def process_files(
    dem_files: List[Path],
    jar: Path,
    out_dir: Optional[Path],
    max_seconds: Optional[int],
    java_opts: str,
    timeout: int,
    workers: int,
    skip_existing: bool,
    continue_on_error: bool,
) -> int:
    if not dem_files:
        LOG.warning("No .dem files to process")
        return 0

    LOG.info("Processing %d .dem files with %d workers", len(dem_files), workers)
    failures = 0
    successes = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {}
        for dem in dem_files:
            csv = build_csv_path(dem, out_dir)
            if skip_existing and csv.exists():
                LOG.info("Skipping existing CSV: %s", csv)
                continue
            future = ex.submit(run_extractor_for_file, jar, dem, csv, max_seconds, java_opts, timeout)
            future_map[future] = (dem, csv)

        for fut in as_completed(future_map):
            dem, csv = future_map[fut]
            try:
                _, ok, rc, msg = fut.result()
            except Exception as e:
                LOG.exception("Worker exception for %s", dem)
                ok = False
                rc = 1
                msg = str(e)
            if ok:
                successes += 1
            else:
                failures += 1
                LOG.error("Failed: %s (rc=%s) msg=%s", dem, rc, msg)
                if not continue_on_error:
                    LOG.error("Aborting due to failure and --continue-on-error not set")
                    return 2

    LOG.info("Done. successes=%d failures=%d", successes, failures)
    return 0 if failures == 0 else 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process .dem files into position CSVs using clarity-parser jar.")
    p.add_argument("--input-dir", type=Path, default=Path("data/replay"), help="Directory to search for .dem files (default data/replay)")
    p.add_argument("--files", nargs="+", type=Path, help="Specific .dem files to process (overrides --input-dir)")
    p.add_argument("--out-dir", type=Path, default=None, help="If set, mirror outputs under this directory (preserves relative paths). Default writes CSV next to .dem")
    p.add_argument("--jar", type=str, default=None, help=f"Path to jar (default {DEFAULT_JAR})")
    p.add_argument("--max-seconds", type=int, default=None, help="Optional max seconds to pass to jar")
    p.add_argument("--java-opts", type=str, default="-Xmx3g", help="Extra java options (default '-Xmx3g')")
    p.add_argument("--timeout", type=int, default=3600, help="Timeout (s) for each java invocation (default 3600)")
    p.add_argument("--workers", type=int, default=2, help="Number of concurrent java processes")
    p.add_argument("--skip-existing", action="store_true", help="Skip if CSV already exists")
    p.add_argument("--continue-on-error", action="store_true", help="Continue processing other files if a file fails")
    p.add_argument("--recursive/--no-recursive", dest="recursive", default=True, help="Search recursively under input-dir (default True)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def main(argv: List[str] | None = None) -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        jar = find_jar(args.jar)
    except FileNotFoundError as e:
        LOG.error(e)
        return 2

    if args.files:
        dem_files = [p for p in args.files if p.exists() and p.suffix.lower() == ".dem"]
        if not dem_files:
            LOG.error("No valid .dem files specified")
            return 2
    else:
        dem_files = find_dem_files(args.input_dir, recursive=args.recursive)

    return process_files(
        dem_files=dem_files,
        jar=jar,
        out_dir=args.out_dir,
        max_seconds=args.max_seconds,
        java_opts=args.java_opts,
        timeout=args.timeout,
        workers=max(1, int(args.workers)),
        skip_existing=args.skip_existing,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())