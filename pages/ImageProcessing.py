import streamlit as st
import pathlib

st.set_page_config(page_title="WaveLab",initial_sidebar_state="collapsed",page_icon="logo (1).png",layout="wide")

def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("style.css")
load_css(csspath)

st.markdown(f"""
        <div class="title-container">
            <h2>Image Processing Techniques</h2>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div class = "button-container">
    <a href="ImageDistribution" target="_self" class = "button-container-probability">Image Distribution</a>
    <a href="YCbCrSpace" target="_self" class = "button-container-probability">Colour Space Conversion</a>
    
</div>
""",unsafe_allow_html=True)
