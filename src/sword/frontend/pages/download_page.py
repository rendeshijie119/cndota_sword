import sys
from pathlib import Path
import os
import glob

import streamlit as st
import pandas as pd

# 解析仓库根
def resolve_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if parent.name == "src":
            return parent.parent
    return p.parents[4]

REPO_ROOT = resolve_repo_root()

# 让顶层包可导入
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 导入你的管道
run_pipeline = None
import_errors = []
try:
    from src.sword.backend.pipelines.all_in_one_pipeline import run_pipeline as _rp
    run_pipeline = _rp
except Exception as e1:
    import_errors.append(("src.sword.backend.pipelines.all_in_one_pipeline", e1))
    try:
        from src.sword.backend.all_in_one_pipeline import run_pipeline as _rp2
        run_pipeline = _rp2
    except Exception as e2:
        import_errors.append(("src.sword.backend.all_in_one_pipeline", e2))

# S3 上传工具
from src.sword.backend.scripts.pipeline_s3_upload import upload_paths, safe_name

DATA_ROOT = REPO_ROOT / "src" / "sword" / "data"
DEFAULT_WARDS_DIR = DATA_ROOT / "obs_logs"
DEFAULT_MATCHES_DIR = DATA_ROOT / "matches"
DEFAULT_REPLAYS_DIR = DATA_ROOT / "replays"
DEFAULT_PICKS_DIR  = DATA_ROOT / "picks"

def discover_artifacts(team_name: str, match_ids: list[str]) -> list[Path]:
    """
    根据 match_ids 和 team 名，在默认数据目录中查找相关文件。
    """
    team_safe = safe_name(team_name)
    found = []

    # matches JSON
    for mid in match_ids:
        found.extend([Path(p) for p in glob.glob(str(DEFAULT_MATCHES_DIR / f"*{mid}*.json"))])

    # obs_logs CSV under teams/<team_safe>
    wards_team_dir = DEFAULT_WARDS_DIR / "teams" / team_safe
    if wards_team_dir.exists():
        for mid in match_ids:
            found.extend([Path(p) for p in glob.glob(str(wards_team_dir / f"*{mid}*_*wards.csv"))])
            found.extend([Path(p) for p in glob.glob(str(wards_team_dir / f"*{mid}*.csv"))])

    # replays (可选)
    if DEFAULT_REPLAYS_DIR.exists():
        for mid in match_ids:
            found.extend([Path(p) for p in glob.glob(str(DEFAULT_REPLAYS_DIR / f"*{mid}*"))])

    # picks （parquet 或 json）
    if DEFAULT_PICKS_DIR.exists():
        for mid in match_ids:
            found.extend([Path(p) for p in glob.glob(str(DEFAULT_PICKS_DIR / f"*{mid}*.parquet"))])
            found.extend([Path(p) for p in glob.glob(str(DEFAULT_PICKS_DIR / f"*{mid}*.json"))])

    # 去重
    uniq, seen = [], set()
    for p in found:
        if str(p) not in seen:
            uniq.append(p); seen.add(str(p))
    return uniq

def main():
    st.title("Dota 2 Data Processing Pipeline")
    st.caption(f"Repo root: {REPO_ROOT}")

    if run_pipeline is None:
        st.error("Cannot import run_pipeline. Please ensure one of these files exists:")
        st.text(str(REPO_ROOT / "src/sword/backend/pipelines/all_in_one_pipeline.py"))
        st.text(str(REPO_ROOT / "src/sword/backend/all_in_one_pipeline.py"))
        with st.expander("Import errors"):
            for mod, err in import_errors:
                st.write(f"{mod}: {err}")
        return

    team_name = st.text_input("Team Name", value="Team Falcons")
    match_ids_raw = st.text_area("Enter Match IDs (comma-separated):", help="Example: 8615531269,9876542050")
    upload_to_s3 = st.checkbox("Upload outputs to S3 after pipeline", value=True)

    if st.button("Start Pipeline"):
        mids = [m.strip() for m in match_ids_raw.split(",") if m.strip()]
        if not mids:
            st.error("Please enter valid Match IDs!")
            return

        with st.spinner("Running pipeline..."):
            results = run_pipeline(match_ids_raw)

        st.success("Pipeline finished.")
        st.expander("Raw results JSON").json(results)

        # 自动发现并上传
        if upload_to_s3:
            paths = discover_artifacts(team_name, mids)
            if not paths:
                st.warning("No local artifacts found to upload. Check data directories under src/sword/data.")
            else:
                st.info(f"Found {len(paths)} files. Uploading to S3...")
                ups = upload_paths(paths, team_name)
                df = pd.DataFrame(ups)
                st.subheader("S3 upload results")
                st.dataframe(df, width='stretch')
                # 展示成功上传的 S3 链接
                ok_df = df[df["ok"] == True]
                if not ok_df.empty:
                    st.write("Uploaded objects:")
                    for r in ok_df.itertuples():
                        st.write(f"- {getattr(r, 's3_url', '')}")
                else:
                    st.warning("No files uploaded. Check errors above.")

if __name__ == "__main__":
    main()