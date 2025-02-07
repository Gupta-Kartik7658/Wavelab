import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pathlib

st.set_page_config(page_title="Central Limit Theorem", page_icon="logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("style.css")
load_css(csspath)

st.markdown(f"""
        <div class="title-container">
            <h2>Central Limit Theorem</h2>
        </div>
        """, unsafe_allow_html=True)

col = st.columns(2)
X = np.zeros(200000)
def gaussianDistibution(mu, sigma, number_of_distributions):
    global X
    for i in range(number_of_distributions):
        X+=np.random.normal(mu,sigma,200000)
        X-=np.mean(X)
    

def laplacianDistribution(mu, scale, number_of_distributions):
    # X = np.zeros(200000)
    global X
    for i in range(number_of_distributions):
        X+=np.random.laplace(mu,scale,200000)
        X-=np.mean(X)
    

def uniformDistribution(a,b,number_of_distributions):
    # X = np.zeros(200000)
    global X
    for i in range(number_of_distributions):
        X+=np.random.uniform(a,b,200000)
        X-=np.mean(X)
    

def exponentialDistribution(location,rate,number_of_distributions):
    # X = np.zeros(1)
    global X
    for i in range(number_of_distributions):
        X+=np.random.exponential(rate,200000) + location
        X-=np.mean(X)
    

with col[0]:
    gaussian = st.number_input("Number of Gaussian Distributions",0,100)
    mean_gaussian = st.slider("Enter the mean for Gaussian Distribution (μ)", min_value=0.0, max_value=5.0, step=0.5, value=0.0)
    sigma_gaussian = st.slider("Standard Deviation for Gaussian Distribution(σ)", min_value=1.0, max_value=5.0, step=0.5, value=1.0)
    gaussianDistibution(mean_gaussian,sigma_gaussian,gaussian)
    
    laplacian = st.number_input("Number of Laplacian Distributions",0,100)
    mean_laplacian = st.slider("Enter the mean for Laplacian Distribution(μ)", min_value=0.0, max_value=5.0, step=0.5, value=0.0)
    scale_laplacian = st.slider("Scale Parameter for Laplacian Distribution(b)", min_value=0.5, max_value=5.0, step=0.5, value=0.5)
    laplacianDistribution(mean_laplacian,scale_laplacian,laplacian)
    
with col[1]:  
    exponential = st.number_input("Number of Exponential Distributions",0,100)
    rate_exponential = st.slider("Rate Parameter for Exponential Distribution(λ)", min_value=0.1, max_value=5.0, step=0.1, value=0.5)
    location_exponential = st.slider("Location Parameter for Exponential Distribution(γ)", min_value=0.1, max_value=5.0, step=0.1, value=0.5)
    exponentialDistribution(location_exponential,rate_exponential,exponential)
    
    uniform = st.number_input("Number of Uniform Distributions",0,100)
    a_uniform = st.slider("Lower Bound for Uniform Distribution(a)", min_value=0.0, max_value=5.0, step=0.5, value=0.0)
    b_uniform = st.slider("Upper Bound for Uniform Distribution(b)", min_value=a_uniform, max_value=5.0, step=0.5, value=1.0)
    uniformDistribution(a_uniform,b_uniform,uniform)
    
fig1 = px.histogram(X, nbins=100, title="Probability Distributive Function", histnorm="probability density")
fig1.update_layout(
    xaxis_title="Value",
    yaxis_title="Probability",
    bargap=0.1,
    showlegend=False,           
)
st.plotly_chart(fig1)

st.markdown(''' 
    <br>

    ### Central Limit Theorem
    
    The Central Limit Theorem (CLT) is a key result in probability theory and statistics. It explains how the distribution of the sum (or average) of a large number of independent and identically distributed (IID) random variables approaches a normal distribution, regardless of the original distribution.
    The theorem states that:
    1. If $X_1, X_2, \\dots, X_n$ are **independent and identically distributed (IID)** random variables with a finite mean $\\mu$ and finite variance $\\sigma^2$,
    2. Then, as $n$ increases, the distribution of the standardized sum approaches a **standard normal distribution**:

    $$
    Z_n = \\frac{\\sum_{i=1}^n X_i - n\\mu}{\\sqrt{n}\\sigma} \\to \\mathcal{N}(0, 1) \, \\text{as} \, n \\to \\infty.
    $$

    ### Applications
    The Central Limit Theorem is fundamental in:
    - Constructing confidence intervals.
    - Hypothesis testing.
    - Approximating distributions of sums and averages.
    - Data analysis and predictive modeling.

    This theorem explains why the normal distribution is so widely used in statistics and serves as the basis for many statistical methods.
''', unsafe_allow_html=True)

