import streamlit as st
import pathlib

st.set_page_config(page_title="WaveLab",initial_sidebar_state="collapsed",page_icon="Kartik/logo (1).png")

def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)

st.markdown(f"""
        <div class="title-container">
            <h2>Probability Distributions</h2>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class = "button-container">
    <a href="GaussianDistribution" target="_self" class = "button-container-probability">Gaussian Distribution</a>
    <a href="LaplacianDistribution" target="_self" class = "button-container-probability">Laplacian Distribution</a>
    <a href="ExponentialDistribution" target="_self" class = "button-container-probability">Exponential Distribution</a>
    <a href="BivariateGaussianDistribution" target="_self" class = "button-container-probability">Bivariate Standard Normal</a>
    
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
        <div class="title-container-new">
            <h2>Inequality Theorems</h2>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div class = "button-container">
    <a href="ChebyshevInequality" target="_self" class = "button-container-probability">Chebyshev's Inequality</a>
</div>
""",unsafe_allow_html=True)
