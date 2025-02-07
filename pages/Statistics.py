import streamlit as st
import pathlib

st.set_page_config(page_title="Probability",initial_sidebar_state="collapsed",page_icon="logo (1).png",layout="wide")

def load_css(file):
    try:
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found.")

csspath = pathlib.Path("style.css")
load_css(csspath)

st.markdown(f"""
        <div class="title-container">
            <h2>Estimation Techniques</h2>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class = "button-container">
    <a href="CentralLimitTheorem" target="_self" class = "button-container-probability">Central Limit Theorem</a>
    <a href="WeakLawOfLargeNumbers" target="_self" class = "button-container-probability">Weak Law of Large Numbers</a>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
        <div class="title-container-new">
            <h2>Dimension Reduction Techniques</h2>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class = "button-container">
    <a href="PrincipalComponentAnalysis" target="_self" class = "button-container-probability">Principal Component Analysis</a>
</div>
""", unsafe_allow_html=True)


