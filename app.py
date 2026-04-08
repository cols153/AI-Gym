import streamlit as st
from tabs.tab_home import render_tab_home
from tabs.tab_analyze import render_tab_analyze
from tabs.tab_live import render_tab_live

st.set_page_config(
    page_title="AI Gym",
    page_icon="🏋️",
    layout="wide"
)

left, main, right = st.columns([1, 5, 1])

with main:
    # Header
    st.title("Push-Up Posture Demo")

    # Tabs layout
    tab1, tab2, tab3 = st.tabs(["Home", "Analyze", "Live"])

    with tab1:
        render_tab_home()
        
    with tab2:
        render_tab_analyze()

    with tab3:
        render_tab_live()