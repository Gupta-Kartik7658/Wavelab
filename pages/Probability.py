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
            <h2>Univariate Distributions</h2>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class = "button-container">
    <a href="GaussianDistribution" target="_self" class = "button-container-probability">Gaussian Distribution</a>
    <a href="LaplacianDistribution" target="_self" class = "button-container-probability">Laplacian Distribution</a>
    <a href="ExponentialDistribution" target="_self" class = "button-container-probability">Exponential Distribution</a>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
        <div class="title-container-new">
            <h2>Bivariate Distributions</h2>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div class = "button-container">
    <a href="BivariateGaussianDistribution" target="_self" class = "button-container-probability">Bivariate Standard Normal</a>
    <a href="BivariateLaplacianDistribution" target="_self" class = "button-container-probability">Independent Bivariate Laplacian</a>
    <a href="BivariateExponentialDistribution" target="_self" class = "button-container-probability">Independent Bivariate Exponential</a>
</div>
""",unsafe_allow_html=True)

st.markdown(f"""
        <div class="title-container-new">
            <h2>Inequality Theorems and Convergence Experiment</h2>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div class = "button-container">
    <a href="ChebyshevInequality" target="_self" class = "button-container-probability">Chebyshev's Inequality</a>
    <a href="MarkovInequality" target="_self" class = "button-container-probability">Markov's Inequality</a>
    <a href="ExpectationConvergence" target="_self" class = "button-container-probability">Expectation Value Convergence</a>
</div>
""",unsafe_allow_html=True)
