import streamlit as st
import numpy as np
import plotly.express as px
import pathlib

st.set_page_config(page_title="Minimum Mean Square Error Estimation", initial_sidebar_state="collapsed", page_icon="logo (1).png", layout="wide")

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
            <h2>Minimum Mean Squared Error Estimation</h2>
        </div>
        """, unsafe_allow_html=True)
X = np.zeros(1)
col = st.columns(2)
with col[0]:
    distribution = st.selectbox(
        "Select the Noise Distribution",
        ("Gaussian (Normal)", "Exponential", "Uniform", "Laplacian", "Poisson")
    )
    # Define default x-axis limits
    if distribution == "Gaussian (Normal)":
        mean = st.slider("Enter the mean (μ)", min_value=0.0, max_value=5.0, step=0.5, value=0.0)
        std = st.slider("Standard Deviation (σ)", min_value=1.0, max_value=5.0, step=0.5, value=1.0)
        X = np.random.normal(mean,std,20000)
    elif distribution == "Exponential":
        rate = st.slider("Rate Parameter (λ)", min_value=0.1, max_value=5.0, step=0.1, value=0.5)
        X = np.random.exponential(rate,20000)
    elif distribution == "Uniform":
        a = st.slider("Lower Bound (a)", min_value=0.0, max_value=5.0, step=0.5, value=0.0)
        b = st.slider("Upper Bound (b)", min_value=0.0, max_value=5.0, step=0.5, value=1.0)
        if a >= b:
            st.error("Upper bound must be greater than lower bound.")
        X = np.random.uniform(a,b,20000)
    elif distribution == "Laplacian":
        mean = st.slider("Enter the mean (μ)", min_value=0.0, max_value=5.0, step=0.5, value=0.0)
        scale = st.slider("Scale Parameter (b)", min_value=0.5, max_value=5.0, step=0.5, value=0.5)
        X = np.random.laplace(mean,scale,20000)
    elif distribution == "Poisson":
        rate = st.slider("Rate Parameter (λ)", min_value=0.5, max_value=5.0, step=0.5, value=1.0)
        X = np.random.poisson(rate,20000)

# with col[1]:
    