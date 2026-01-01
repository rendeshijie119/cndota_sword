import sys
from pathlib import Path

import streamlit as st
import pandas as pd

def resolve_repo_root() -> Path:
    """
    从当前文件路径向上找到名为 'src' 的目录，然后取其父级作为仓库根。
    例如：<repo>/src/sword/frontend/pages/download_page.py -> 返回 <repo>
    """
    p = Path(__file__).resolve()
    for parent in p.parents:
        if parent.name == "src":
            return parent.parent
    return p.parents[4]  # 兜底

REPO_ROOT = resolve_repo_root()

# 确保顶层目录 <repo> 在 sys.path（这样可以 import src.sword...）
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 依次尝试两种可能位置的管道函数（注意：第一个是 pipelines/all_in_one_pipeline.py 单数）
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

def main():
    st.title("Dota 2 Data Processing Pipeline")
    # st.caption(f"Repo root: {REPO_ROOT}")
    # st.caption("Import target: src.sword.backend.pipelines.all_in_one_pipeline (preferred) or src.sword.backend.all_in_one_pipeline")

    if run_pipeline is None:
        st.error("Cannot import run_pipeline. Please ensure one of these files exists:")
        st.text(str(REPO_ROOT / "src/sword/backend/pipelines/all_in_one_pipeline.py"))
        st.text(str(REPO_ROOT / "src/sword/backend/all_in_one_pipeline.py"))
        with st.expander("Import errors"):
            for mod, err in import_errors:
                st.write(f"{mod}: {err}")
        return

    match_ids = st.text_area("Enter Match IDs (comma-separated):", help="Example: 8615531269,9876542050")
    if st.button("Start Pipeline"):
        if not match_ids or not match_ids.strip():
            st.error("Please enter valid Match IDs!")
            return

        with st.spinner("Running pipeline..."):
            # 直接调用管道函数（同步执行）
            results = run_pipeline(match_ids)

        # 展示表格
        table_data = results.get("table_data", [])
        if table_data:
            df = pd.DataFrame(table_data)
            cols = ["Match ID", "Radiant Team", "Dire Team", "JSON", "Obs Logs", "Replays", "Hero Picks"]
            df = df[[c for c in cols if c in df.columns]]
            st.subheader("Pipeline Results")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No table data returned.")

        # 可选：原始结果 JSON
        st.expander("Raw results JSON").json(results)