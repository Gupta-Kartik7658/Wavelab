import streamlit as st
import pathlib

st.set_page_config(page_title="Probability",initial_sidebar_state="collapsed",page_icon="Kartik/logo (1).png")

def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)

st.markdown("""
<div class = "button-container">
    <a href="PrincipalComponentAnalysis" target="_self" class = "button-container-probability">Principal Component Analysis</a>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class = "button-container">
    <a href="CentralLimitTheorem" target="_self" class = "button-container-probability">Central Limit Theorem</a>
</div>
""", unsafe_allow_html=True)
