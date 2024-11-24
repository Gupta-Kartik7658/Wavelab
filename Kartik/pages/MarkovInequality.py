import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.stats import gaussian_kde

st.set_page_config(page_title="Markov's Inequality", initial_sidebar_state="collapsed", page_icon="Kartik/logo (1).png", layout="wide")

def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)
st.markdown(f"""
        <div class="title-container">s
            <h2>Markov's Inequality</h2>
        </div>
        """, unsafe_allow_html=True)
# Left Column: Explanation and Inputs
st.markdown('''<br>''',unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    # Title and explanation
    
    st.markdown('''
    ### Markov's Statement

Markov's Inequality provides an upper bound on the probability that a non-negative random variable $$ X $$ is greater than or equal to a positive constant $$\\alpha$$. Specifically, for any random variable $$ X $$ and $$ \\alpha > 0 $$:

$$
P(X \\geq a) \\leq \\frac{\\mathbb{E}[X]}{a}
$$

Where:
- $$ \mathbb{E}[X] $$ is the expected value (mean) of $$ X $$.
- $$ \\alpha $$ is a positive constant.

This inequality is useful in probability theory for bounding the tails of a distribution.
''')


    # Dropdown for selecting the distribution
    distribution = st.selectbox(
        "Select a Distribution",
        ( "Exponential", "Uniform", "Poisson")
    )
    X = np.random.uniform(0, 1, 100000)
    # Input parameters based on the selected distribution
    if distribution == "Exponential":
        rate = st.slider(label="Rate Parameter (λ)", min_value = 0.1, max_value = 5.0, step = 0.5,value = 0.1)
        X = np.random.exponential(1 / rate, 100000)
    elif distribution == "Uniform":
        lower = st.slider(label="Lower Bound (a)", min_value = 0.0, max_value = 5.0, step = 0.5,value = 0.0)
        upper = st.slider(label="Upper Bound (b)", min_value = 0.0, max_value = 5.0, step = 0.5,value = 1.0)
        if lower >= upper:
            st.error("Upper bound must be greater than lower bound.")
        else:
            X = np.random.uniform(lower, upper, 100000)
    elif distribution == "Poisson":
        lam = st.slider(label="Rate Parameter (λ)", min_value = 0.5, max_value = 5.0, step = 0.5,value = 0.5)
        X = np.random.poisson(lam, 100000)

    # Input for k (distance in terms of standard deviation)
    alpha = st.slider(label="α", min_value = 1.0, max_value = 5.0, step = 0.1,value = 1.0)

    # Calculate mean and standard deviation`   `
    mean = np.mean(X)

    # Calculate Markov's bounds
    lower_bound = mean/alpha

# Right Column: Plot
with col2:
    # Filter data
    inside_bounds = (X >= lower_bound)

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
        where=(x_vals >= alpha),
        color="blue",
        alpha=0.5,
        label = "Markov's Bound",
    )
    ax.fill_between(
        x_vals,
        density,
        where=(x_vals < alpha),
        color="red",
        alpha=0.5,
        label = "Total Density",
    )
    ax.legend()
    # Annotate lines
    ax.axvline(alpha, color="black", linestyle="--")

    # Labels and title
    ax.set_title(f"Markov's Theorem for {distribution} Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.grid()

    # Display the plot
    st.pyplot(fig)

# Display Markov's probability details
with col1:
    st.markdown(f''' **Markov's bounds (Red region): $$P(X ≥ α)$$ ≤ {round(lower_bound,2)}**''')
