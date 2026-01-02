import sys, pathlib, streamlit as st

THIS = pathlib.Path(__file__).resolve()
REPO_ROOT = THIS.parents[3]
SRC_DIR = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.sword.frontend.pages import ward_plot_page, download_page

st.set_page_config(page_title="CN Dota Sword", layout="wide")

def main():
    st.sidebar.title("Pages")
    page = st.sidebar.radio("Select", ["Ward Plot", "Download"])
    if page == "Ward Plot":
        ward_plot_page.main()
    else:
        download_page.main()

if __name__ == "__main__":
    main()