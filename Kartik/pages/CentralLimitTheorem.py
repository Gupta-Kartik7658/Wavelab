import streamlit as st
import numpy as np
import plotly.express as px
import pathlib

st.set_page_config(page_title="CentralLimitTheorem", initial_sidebar_state="collapsed", page_icon="Kartik/logo (1).png", layout="wide")

def load_css(file):
    try:
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found.")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)

st.markdown(f"""
        <div class="title-container">
            <h2>Central Limit Theorem</h2>
        </div>
        """, unsafe_allow_html=True)

col = st.columns(2)
X = np.zeros(1)
def centralLimitTheorem(Distribution, parameters, N=10, meanCentered=1):
    X = np.zeros(100000)
    for i in range(N):
        if Distribution == "Gaussian (Normal)":
            y = np.random.normal(parameters[0], parameters[1], 100000)
        elif Distribution == "Exponential":
            y = np.random.exponential(1 / parameters[0], 100000)
        elif Distribution == "Uniform":
            if parameters[0] >= parameters[1]:
                st.error("Upper bound must be greater than lower bound.")
                return np.zeros(100000)
            y = np.random.uniform(parameters[0], parameters[1], 100000)
        elif Distribution == "Laplacian":
            y = np.random.laplace(parameters[0], parameters[1], 100000)
        elif Distribution == "Poisson":
            y = np.random.poisson(parameters[0], 100000)
        else:
            st.error("Unsupported distribution selected.")
            return np.zeros(100000)

        X += y - (np.mean(y) * meanCentered)
    return X

with col[0]:
    distribution = st.selectbox(
        "Select a Distribution",
        ("Gaussian (Normal)", "Exponential", "Uniform", "Laplacian", "Poisson")
    )
    parameter = []

    # Define default x-axis limits
    x_min, x_max = -10, 10

    if distribution == "Gaussian (Normal)":
        parameter.append(st.slider("Enter the mean (μ)", min_value=0.0, max_value=5.0, step=0.5, value=0.0))
        parameter.append(st.slider("Standard Deviation (σ)", min_value=1.0, max_value=5.0, step=0.5, value=1.0))
        x_min = parameter[0] - 5 * parameter[1]
        x_max = parameter[0] + 5 * parameter[1]
    elif distribution == "Exponential":
        parameter.append(st.slider("Rate Parameter (λ)", min_value=0.1, max_value=5.0, step=0.1, value=0.5))
        x_min = 0
        x_max = 8 / parameter[0]  # Approximate range for exponential distribution
    elif distribution == "Uniform":
        parameter.append(st.slider("Lower Bound (a)", min_value=0.0, max_value=5.0, step=0.5, value=0.0))
        parameter.append(st.slider("Upper Bound (b)", min_value=0.0, max_value=5.0, step=0.5, value=1.0))
        if parameter[0] >= parameter[1]:
            st.error("Upper bound must be greater than lower bound.")
        x_min, x_max = parameter[0], parameter[1]
    elif distribution == "Laplacian":
        parameter.append(st.slider("Enter the mean (μ)", min_value=0.0, max_value=5.0, step=0.5, value=0.0))
        parameter.append(st.slider("Scale Parameter (b)", min_value=0.5, max_value=5.0, step=0.5, value=0.5))
        x_min = parameter[0] - 4 * parameter[1]
        x_max = parameter[0] + 4 * parameter[1]
    elif distribution == "Poisson":
        parameter.append(st.slider("Rate Parameter (λ)", min_value=0.5, max_value=5.0, step=0.5, value=1.0))
        x_min = 0
        x_max = parameter[0] * 4  # Approximate range for Poisson distribution

    number_of_distributions = st.slider("Number of Distributions to add (N)", min_value=1, max_value=100, step=1, value=5)
    meanCentred = int(st.checkbox("Mean Centred",value=True))
    
with col[1]:
    X = centralLimitTheorem(distribution, parameter, number_of_distributions, meanCentred)
    
    if(meanCentred):
        X = X/number_of_distributions
        fig = px.histogram(X, nbins=50, title="Histogram of Central Limit Theorem", histnorm="probability density")
        fig.update_layout(
            xaxis_title="Value",
            yaxis_title="Probability",
            xaxis_range=[x_min, x_max],  # Fix the x-axis range
            bargap=0.1,
            showlegend=False  # Remove legend
        )
    else:
        fig = px.histogram(X, nbins=50, title="Histogram of Central Limit Theorem", histnorm="probability density")
        fig.update_layout(
            xaxis_title="Value",
            yaxis_title="Probability",
            bargap=0.1,
            showlegend=False  # Remove legend
        )
    st.plotly_chart(fig)

st.markdown(''' 
            <br>
    The Central Limit Theorem (CLT) is a fundamental result in probability theory and statistics. It explains how the sum (or average) of a large number of independent and identically distributed (IID) random variables behaves.

    ### Central Limit Theorem
    The theorem states that:
    1. If the random variables $X_1, X_2, \\dots, X_n$ are **independent and identically distributed (IID)** with a finite mean $\\mu$ and finite variance $\\sigma^2$,
    2. Then, as the number of random variables $n$ increases, the distribution of their normalized sum approaches the **standard normal distribution**.

    Mathematically, this can be written as:

    $$
    Z = \\frac{\\sum_{i=1}^n X_i - n\\mu}{\\sqrt{n\\sigma^2}} \\to N(0, 1),
    $$
    
    where:
    - $\\sum_{i=1}^n X_i$: The sum of $n$ random variables.
    - $\\mu$: The mean of the random variables.
    - $\\sigma^2$: The variance of the random variables.
    - $N(0, 1)$: The standard normal distribution with mean $0$ and variance $1$.

    This theorem explains why many distributions observed in nature and statistics resemble the **normal distribution**.

    ### Weak Law of Large Numbers (WLLN)
    The **Weak Law of Large Numbers** is closely related to the CLT and provides a foundation for understanding averages.

    The WLLN states that for a sequence of IID random variables $X_1, X_2, \\dots, X_n$ with mean $\\mu$:

    $$
    \\bar{X}_n = \\frac{1}{n} \\sum_{i=1}^n X_i \\to \\mu \, \\text{as} \, n \\to \\infty,
    $$

    where:
    - $\\bar{X}_n$: The sample mean of $n$ random variables.

    In simple terms, the WLLN states that the sample mean of a large number of IID random variables will converge to the true mean $\\mu$ as the sample size $n$ increases.

    ### Applications
    Both the CLT and WLLN are widely used in:
    - Hypothesis testing and confidence intervals.
    - Estimating population parameters from sample data.
    - Understanding the behavior of averages and sums in large datasets.

    These results form the foundation of much of modern statistics and data science.
''', unsafe_allow_html=True)
