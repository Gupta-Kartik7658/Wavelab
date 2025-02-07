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

Y_cap = np.mean(Y) + (np.cov(X,Y)[0][1]/np.var(X))*(X - np.mean(X))
# st.write(Y_cap.shape)

data = pd.DataFrame({
    "x": list(X) + list(X),
    "y": list(Y) + list(Y_cap),
    "category": ["Y"] * len(X) + ["Y_cap"] * len(X),
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
    



  
  