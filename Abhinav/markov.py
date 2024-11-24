import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import multivariate_normal

# Set up the layout with two columns
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 2], gap="large")

# Left Column: Explanation and Inputs
with col1:
    # Title and explanation
    st.title("3D Joint Distribution and Chebyshev's Inequality")
    st.markdown("""
    ### What is Chebyshev's Inequality?

    Chebyshev's Inequality states that for any random variable \( X \) with mean \( \mu \) and variance \( \sigma^2 \),
    the probability that \( X \) deviates from its mean by more than \( k \) standard deviations is bounded by:

    \[
    P(|X - \mu| \geq k \cdot \sigma) \leq \frac{1}{k^2}
    \]

    In other words, no more than \( \frac{1}{k^2} \) of the distribution's values can lie more than \( k \) standard deviations away from the mean.

    ### What is a Joint Distribution?
    A joint distribution describes the probability of two (or more) random variables occurring together. In this plot, 
    the joint distribution of \( X \) and \( Y \) will show the likelihood of both variables occurring simultaneously.
    """)

    # Input fields for joint distribution
    mean_x = st.number_input("Enter the mean of X (μx)", value=0.0)
    std_dev_x = st.number_input("Enter the standard deviation of X (σx)", value=1.0, min_value=0.01)
    mean_y = st.number_input("Enter the mean of Y (μy)", value=0.0)
    std_dev_y = st.number_input("Enter the standard deviation of Y (σy)", value=1.0, min_value=0.01)
    
    correlation = st.number_input("Enter the correlation between X and Y", value=0.0, min_value=-1.0, max_value=1.0)
    
    # Generate the joint distribution
    cov_matrix = np.array([[std_dev_x**2, correlation*std_dev_x*std_dev_y],
                           [correlation*std_dev_x*std_dev_y, std_dev_y**2]])
    
    # Number of points to generate
    num_points = 10000
    mean = [mean_x, mean_y]
    X, Y = np.random.multivariate_normal(mean, cov_matrix, num_points).T
    
    # Thresholds for Chebyshev's Inequality
    k_x = st.number_input("Enter k for Chebyshev's inequality (X)", value=1.0, min_value=0.1)
    k_y = st.number_input("Enter k for Chebyshev's inequality (Y)", value=1.0, min_value=0.1)

# Right Column: 3D Plot using Plotly
with col2:
    # Calculate the expected values for Chebyshev's Inequality
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    std_dev_x = np.std(X)
    std_dev_y = np.std(Y)
    
    # Apply Chebyshev's inequality
    chebyshev_bound_x = 1 / k_x**2
    chebyshev_bound_y = 1 / k_y**2
    
    # Create 3D plot for joint distribution using Plotly
    x = np.linspace(np.min(X) - 1, np.max(X) + 1, 100)
    y = np.linspace(np.min(Y) - 1, np.max(Y) + 1, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # Calculate the joint PDF for the bivariate normal distribution
    pos = np.dstack((X_grid, Y_grid))
    rv = multivariate_normal(mean, cov_matrix)
    Z = rv.pdf(pos)
    
    # Create the Plotly surface plot
    fig = go.Figure(data=[go.Surface(
        z=Z,
        x=X_grid,
        y=Y_grid,
        colorscale='Viridis',
        opacity=0.7
    )])
    
    # Highlight regions outside the Chebyshev bound (X and Y)
    Z_chebyshev_x = np.where(np.abs(X_grid - mean_x) >= k_x * std_dev_x, Z, 0)
    Z_chebyshev_y = np.where(np.abs(Y_grid - mean_y) >= k_y * std_dev_y, Z, 0)
    
    fig.add_trace(go.Surface(
        z=Z_chebyshev_x,
        x=X_grid,
        y=Y_grid,
        colorscale='Reds',
        opacity=0.5,
        showscale=False,
        name="X Outside Bound"
    ))

    fig.add_trace(go.Surface(
        z=Z_chebyshev_y,
        x=X_grid,
        y=Y_grid,
        colorscale='Blues',
        opacity=0.5,
        showscale=False,
        name="Y Outside Bound"
    ))

    # Labels and title
    fig.update_layout(
        title="Joint Distribution of X and Y with Chebyshev's Inequality",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Probability Density"
        ),
        autosize=True
    )
    
    # Display the plot
    st.plotly_chart(fig)

# Display Chebyshev's inequality details
with col1:
    st.markdown(f"""
    ### Probability Insights for Chebyshev's Inequality:
    - **Chebyshev's upper bound for X (|X - μx| ≥ kσx):** 
    \( P(|X - μx| \geq kσx) \leq {chebyshev_bound_x:.2f} \)
    - **Chebyshev's upper bound for Y (|Y - μy| ≥ kσy):** 
    \( P(|Y - μy| \geq kσy) \leq {chebyshev_bound_y:.2f} \)
    
    In this joint distribution:
    - **The red region** represents where \( |X - μx| \geq kσx \).
    - **The blue region** represents where \( |Y - μy| \geq kσy \).
    """)
