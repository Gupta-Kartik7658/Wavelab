import streamlit as st
import numpy as np
import plotly.express as px
import pathlib
import pandas as pd

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

col = st.columns(2)
N = st.slider("Number of Samples (N)", min_value=10000, max_value=100000, step=10000,value=10000)
X = np.zeros(1)
with col[1]:
    distribution = st.selectbox(
        "Select the Noise Distribution",
        ("Gaussian (Normal)", "Exponential", "Uniform", "Laplacian", "Poisson")
    )
    if distribution == "Gaussian (Normal)":
        mean = st.slider("Enter the mean for Noise(μ)", min_value=0.0, max_value=5.0, step=0.5, value=0.0)
        std = st.slider("Standard Deviation for Noise(σ)", min_value=1.0, max_value=5.0, step=0.5, value=1.0)
        Noise = np.random.normal(mean,std,N)
    elif distribution == "Exponential":
        rate = st.slider("Rate Parameter for Noise(λ)", min_value=0.1, max_value=5.0, step=0.1, value=0.5)
        Noise = np.random.exponential(rate,N)
    elif distribution == "Uniform":
        a = st.slider("Lower Bound for Noise(a)", min_value=0.0, max_value=5.0, step=0.5, value=0.0)
        b = st.slider("Upper Bound for Noise(b)", min_value=a, max_value=a+5.0, step=0.5, value=1.0)
        Noise = np.random.uniform(a,b,N)
    elif distribution == "Laplacian":
        mean = st.slider("Enter the mean for Noise(μ)", min_value=0.0, max_value=5.0, step=0.5, value=0.0)
        scale = st.slider("Scale Parameter for Noise(b)", min_value=0.5, max_value=5.0, step=0.5, value=0.5)
        Noise = np.random.laplace(mean,scale,N)
    elif distribution == "Poisson":
        rate = st.slider("Rate Parameter for Noise(λ)", min_value=0.5, max_value=5.0, step=0.5, value=1.0)
        Noise = np.random.poisson(rate,N)

    # Define default x-axis limits

with col[0]:
    distribution = st.selectbox(
        "Select the X Distribution",
        ("Gaussian (Normal)", "Exponential", "Uniform", "Laplacian", "Poisson")
    )
    if distribution == "Gaussian (Normal)":
        mean = st.slider("Enter the mean for X(μ)", min_value=0.0, max_value=5.0, step=0.5, value=0.0)
        std = st.slider("Standard Deviation for X(σ)", min_value=1.0, max_value=5.0, step=0.5, value=1.0)
        X = np.random.normal(mean,std,N)
    elif distribution == "Exponential":
        rate = st.slider("Rate Parameter for X(λ)", min_value=0.1, max_value=5.0, step=0.1, value=0.5)
        X = np.random.exponential(rate,N)
    elif distribution == "Uniform":
        a = st.slider("Lower Bound for X(a)", min_value=0.0, max_value=5.0, step=0.5, value=0.0)
        b = st.slider("Upper Bound for X(b)", min_value=a, max_value=a+5.0, step=0.5, value=1.0)
        X = np.random.uniform(a,b,N)
    elif distribution == "Laplacian":
        mean = st.slider("Enter the mean for X(μ)", min_value=0.0, max_value=5.0, step=0.5, value=0.0)
        scale = st.slider("Scale Parameter for X(b)", min_value=0.5, max_value=5.0, step=0.5, value=0.5)
        X = np.random.laplace(mean,scale,N)
    elif distribution == "Poisson":
        rate = st.slider("Rate Parameter for X(λ)", min_value=0.5, max_value=5.0, step=0.5, value=1.0)
        X = np.random.poisson(rate,N)
m = st.slider("Slope (m) of the Line: Y = mX + c",min_value=0.0,max_value=10.0,step=0.5,value=1.0)
c = st.slider("Intercept (c) of the Line: Y = mX + c",min_value=0.0,max_value=10.0,step=0.5,value=1.0)


Y0 = m*X + c
Y = m*X + c + Noise 
a_o = np.cov(X,Y)[0][1]/np.var(X)
b_o = np.mean(Y) - (a_o * np.mean(X))
Y_cap = a_o*X + b_o
col = st.columns(3)
col[0].metric(label="Optimal Slope ($a_o$)", value=str(round(a_o,3)))
col[1].metric(label="Optimal Intercept ($b_o$)", value=str(round(b_o,3)))
col[2].metric(label="Mean Square Error $E((\\hat{Y}-Y)^2)$", value=str(round(np.mean(np.dot(Y-Y0,Y-Y0)),2)))

data = pd.DataFrame({
    "x": list(X) + list(X),
    "y": list(Y) + list(Y_cap),
    "category": ["Y"] * len(X) + ["Y_hat"] * len(X),
    "size": [15] * (2*len(X))
})

# Plot both in the same figure
fig = px.scatter(data, x="x", y="y", color="category", size="size" ,title="Combined Scatter Plot")
st.plotly_chart(fig)
col = st.columns(2)

with col[0]:
    fig1 = px.histogram(X, title="PDF of X", histnorm="probability density")
    fig1.update_layout(
        xaxis_title="Value",
        yaxis_title="Probability",
        bargap=0.1,
        showlegend=False,           
    )
    st.plotly_chart(fig1)  
    fig2 = px.histogram(Y0, title="PDF of Y = mX + c", histnorm="probability density")
    fig2.update_layout(
        xaxis_title="Value",
        yaxis_title="Probability",
        bargap=0.1,
        showlegend=False,           
    )
    st.plotly_chart(fig2)  
with col[1]:
    fig1 = px.histogram(Noise, title="PDF of Noise", histnorm="probability density")
    fig1.update_layout(
        xaxis_title="Value",
        yaxis_title="Probability",
        bargap=0.1,
        showlegend=False,           
    )
    st.plotly_chart(fig1)  
    fig2 = px.histogram(Y, title="PDF of Y = mX + c + Noise", histnorm="probability density")
    fig2.update_layout(
        xaxis_title="Value",
        yaxis_title="Probability",
        bargap=0.1,
        showlegend=False,           
    )
    st.plotly_chart(fig2)  
    
    import streamlit as st

st.markdown('''
            <style>
                *{
                    font-size:20px
                }
            </style>
        <br>
        Minimum Mean Square Error (MMSE) estimation for line fitting is a fundamental concept in statistics and machine learning. It provides an optimal way to fit a linear model to data by minimizing the mean squared error between predicted and observed values.

        ### Minimum Mean Square Error Estimation
        The goal of MMSE estimation is to find the optimal line:
        $$
        y = mx + c
        $$
        where:
        - $m$ is the **slope** of the line.
        - $c$ is the **intercept** of the line.

        The **MMSE estimation** minimizes the following error:
        $$
        \\text{MSE} = \\frac{1}{n} \\sum_{i=1}^n (y_i - (mx_i + c))^2
        $$

        ### Optimal Slope and Intercept
        The optimal values of $m$ and $c$ can be expressed in terms of **expectations, variances, and covariances**:

        1. **Optimal Slope $a_o$**:
        $$
        a_o = \\frac{\\text{Cov}(x, y)}{\\text{Var}(x)}
        $$

        2. **Optimal Intercept $b_o$**:
        $$
        b_o = \\mathbb{E}[y] - m \\mathbb{E}[x]
        $$

        Where:
        - $ \\text{Cov}(x, y) = \\mathbb{E}[xy] - \\mathbb{E}[x]\\mathbb{E}[y] $ is the **covariance** between $x$ and $y$.
        - $ \\text{Var}(x) = \\mathbb{E}[x^2] - (\\mathbb{E}[x])^2 $ is the **variance** of $x$.
        - $ \\mathbb{E}[x] $ and $ \\mathbb{E}[y] $ are the **expectations (means)** of $x$ and $y$.

        ### MMSE Fitted Line
        The fitted line using MMSE is given by:
        $$
        \\hat{y} = a_o x + b_o
        $$

        ### Applications
        MMSE estimation for line fitting is widely used in:
        - Regression analysis to model relationships between variables.
        - Machine learning for supervised learning tasks.
        - Signal processing and time-series analysis.
        - Data science to analyze trends and patterns in data.

        This method is crucial for understanding and modeling linear relationships in data and forms the basis for many advanced statistical and machine learning techniques.
''', unsafe_allow_html=True)



