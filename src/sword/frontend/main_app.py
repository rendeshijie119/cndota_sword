import streamlit as st
from pages import ward_plot_page, download_page

def main():
    st.set_page_config(page_title="Sword", layout="wide")
    tabs = st.tabs(["Ward Analysis", "Data Pipeline"])
    with tabs[0]:
        ward_plot_page.main()
    with tabs[1]:
        download_page.main()

if __name__ == "__main__":
    main()