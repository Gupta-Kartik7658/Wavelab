import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Set up the layout with two columns
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 2], gap="large")

# Left Column: Explanation and Inputs
with col1:
    # Title and explanation
    st.title("Chebyshev's Theorem Visualization")
    st.markdown("""
    ### What is Chebyshev's Inequality?

    Chebyshev's Inequality states that for any random variable \(X\) with a finite mean \( \mu \) and variance \( \sigma^2 \), 
    the probability that \(X\) lies within \(k\) standard deviations of the mean is at least \(1 - \frac{1}{k^2}\), where \(k > 1\). 
    Mathematically:

    \[
    P(|X - \mu| \geq k \sigma) \leq \frac{1}{k^2}
    \]

    ### Key Points:
    - **For \(k = 2\):** At least 75% of the data lies within 2 standard deviations of the mean.
    - **For \(k = 3\):** At least 89% of the data lies within 3 standard deviations of the mean.
    - **Universal Applicability:** Works for any distribution, even non-Gaussian ones.
    """)

    # Dropdown for selecting the distribution
    distribution = st.selectbox(
        "Select a Distribution",
        ("Gaussian (Normal)", "Exponential", "Uniform", "Laplacian", "Poisson")
    )

    # Input parameters based on the selected distribution
    if distribution == "Gaussian (Normal)":
        mean = st.number_input("Enter the mean (μ)", value=0.0)
        std_dev = st.number_input("Enter the standard deviation (σ)", value=1.0, min_value=0.01)
        X = np.random.normal(mean, std_dev, 100000)
    elif distribution == "Exponential":
        rate = st.number_input("Enter the rate parameter (λ)", value=1.0, min_value=0.01)
        X = np.random.exponential(1 / rate, 100000)
    elif distribution == "Uniform":
        lower = st.number_input("Enter the lower bound (a)", value=0.0)
        upper = st.number_input("Enter the upper bound (b)", value=1.0)
        if lower >= upper:
            st.error("Upper bound must be greater than lower bound.")
        else:
            X = np.random.uniform(lower, upper, 100000)
    elif distribution == "Laplacian":
        mean = st.number_input("Enter the mean (μ)", value=0.0)
        scale = st.number_input("Enter the scale parameter (b)", value=1.0, min_value=0.01)
        X = np.random.laplace(mean, scale, 100000)
    elif distribution == "Poisson":
        lam = st.number_input("Enter the rate parameter (λ)", value=5.0, min_value=0.01)
        X = np.random.poisson(lam, 100000)

    # Input for k (distance in terms of standard deviation)
    k = st.number_input("Enter the value of k (distance in terms of σ)", value=1.0, min_value=0.1)

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
    )
    ax.fill_between(
        x_vals,
        density,
        where=(x_vals < lower_bound) | (x_vals > upper_bound),
        color="red",
        alpha=0.5,
    )

    # Annotate lines
    ax.axvline(mean, color="black", linestyle="--", label="Mean (μ)")
    ax.axvline(lower_bound, color="gray", linestyle="--", label="μ - kσ")
    ax.axvline(upper_bound, color="gray", linestyle="--", label="μ + kσ")

    # Labels and title
    ax.set_title(f"Chebyshev's Theorem for {distribution} Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.legend()

    # Display the plot
    st.pyplot(fig)

# Display Chebyshev's probability details
with col1:
    chebyshev_lower_bound = 1 - 1 / (k ** 2)
    st.markdown(f"""
    ### Probability Insights:
    - **Within bounds (Red region):** \(P(μ - kσ ≤ X ≤ μ + kσ) ≥ {chebyshev_lower_bound:.2f}\)
    - **Outside bounds (Blue region):** \(P(X < μ - kσ \text{{ or }} X > μ + kσ) ≤ {1 / (k ** 2):.2f}\)
    """)
