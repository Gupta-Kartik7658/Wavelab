import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.stats import gaussian_kde

st.set_page_config(page_title="Chebyshev's Inequality", initial_sidebar_state="collapsed", page_icon="logo (1).png", layout="wide")

def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("style.css")
load_css(csspath)


st.markdown(f"""
        <div class="title-container">
            <h2>Chebyshev's Inequality</h2>
        </div>
        """, unsafe_allow_html=True)
# Left Column: Explanation and Inputs
col1, col2 = st.columns([1, 2])

with col1:
    # Title and explanation
    
    st.markdown('''
    ### Chebyshev's Statement
    Chebyshev's Inequality states that for any random variable $$X$$ with a finite mean $$\\mu$$ and variance $$\\sigma^2$$, 
    the probability that $$X$$ lies within $$k$$ standard deviations of the mean is at most $$\\frac{1}{k^2}$$, where $$k > 1$$. 
    Mathematically:

    $$ 
    P(|X - \\mu| \\geq k \\sigma) \\leq \\frac{1}{k^2} 
    $$''')


    # Dropdown for selecting the distribution
    distribution = st.selectbox(
        "Select a Distribution",
        ("Gaussian (Normal)", "Exponential", "Uniform", "Laplacian", "Poisson")
    )

    # Input parameters based on the selected distribution
    if distribution == "Gaussian (Normal)":
        mean = st.slider(label="Enter the mean (μ)", min_value = 0.0, max_value = 5.0, step = 0.5,value = 0.0)
        std_dev = st.slider(label="Standard Deviation(σ)", min_value = 1.0, max_value = 5.0, step = 0.5,value = 1.0)
        X = np.random.normal(mean, std_dev, 100000)
    elif distribution == "Exponential":
        rate = st.slider(label="Rate Parameter (λ)", min_value = 0.1, max_value = 5.0, step = 0.5,value = 0.1)
        X = np.random.exponential(1 / rate, 100000)
    elif distribution == "Uniform":
        lower = st.slider(label="Lower Bound (a)", min_value = 0.0, max_value = 5.0, step = 0.5,value = 0.0)
        upper = st.slider(label="Upper Bound (b)", min_value = 0.0, max_value = 5.0, step = 0.5,value = 0.0)
        if lower >= upper:
            st.error("Upper bound must be greater than lower bound.")
        else:
            X = np.random.uniform(lower, upper, 100000)
    elif distribution == "Laplacian":
        mean = st.slider(label="Enter the mean (μ)", min_value = 0.0, max_value = 5.0, step = 0.5,value = 0.0)
        scale = st.slider(label="Enter the scale parameter (b)", min_value = 0.5, max_value = 5.0, step = 0.5,value = 0.5)
        X = np.random.laplace(mean, scale, 100000)
    elif distribution == "Poisson":
        lam = st.slider(label="Rate Parameter (λ)", min_value = 0.5, max_value = 5.0, step = 0.5,value = 0.5)
        X = np.random.poisson(lam, 100000)

    # Input for k (distance in terms of standard deviation)
    k = st.slider(label="k (distance in terms of σ)", min_value = 1.0, max_value = 5.0, step = 0.1,value = 1.0)

    # Calculate mean and standard deviation
    mean = np.mean(X)
    std_dev = np.std(X)

    # Calculate Chebyshev's bounds
    lower_bound = mean - k * std_dev
    upper_bound = mean + k * std_dev

# Right Column: Plot
with col2:
    # Filter data
    inside_bounds = (X >= lower_bound) & (X <= upper_bound)
    outside_bounds = ~inside_bounds

    # Generate KDE for smoother curves
    kde = gaussian_kde(X)
    x_vals = np.linspace(np.min(X) - 1, np.max(X) + 1, 1000)
    density = kde(x_vals)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot full distribution
    ax.plot(x_vals, density, color="black")

    # Shade the regions
    ax.fill_between(
        x_vals,
        density,
        where=(x_vals >= lower_bound) & (x_vals <= upper_bound),
        color="blue",
        alpha=0.5,
        label = "Enclosed Probability",
    )
    ax.fill_between(
        x_vals,
        density,
        where=(x_vals < lower_bound) | (x_vals > upper_bound),
        color="red",
        alpha=0.5,
        label = "Leftover Probability",
    )

    # Annotate lines
    ax.axvline(mean, color="white", linestyle="--")
    ax.axvline(lower_bound, color="white", linestyle="--")
    ax.axvline(upper_bound, color="white", linestyle="--")

    # Labels and title
    ax.set_title(f"Chebyshev's Theorem for {distribution} Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.legend()

    # Display the plot
    st.pyplot(fig)

# Display Chebyshev's probability details
with col1:
    st.markdown(f''' **Chebyshev's bounds (Red region): $$P(|X-μ| ≥ kσ^2)$$ ≤ {round(1 / (k ** 2),2)}**''')
