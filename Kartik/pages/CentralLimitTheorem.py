import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import pathlib

st.set_page_config(page_title="CentralLimitTheorem", initial_sidebar_state="collapsed", page_icon="Kartik/logo (1).png", layout="wide")

def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)

# Function to generate a uniform distribution and return centered data
def uniform_distribution(size):
    return np.random.uniform(0, 1, size)

# Streamlit App Layout
st.title("Adding Uniform Distributions with Plotly")


# Display Explanation
st.markdown("""
### What is the Concept Behind Adding Uniform Distributions?
In this demonstration, we are adding multiple uniform distributions together. Each distribution is generated randomly, and then its mean is subtracted to center it. The cumulative sum of these centered distributions is plotted as a smooth curve.

- **Central Limit Theorem (CLT)**: As more uniform distributions are added, the resulting distribution will tend to approximate a normal (Gaussian) distribution due to the **Central Limit Theorem**.
""")

# Input for the number of uniform distributions to add using slider
n = st.slider("Select Number of Uniform Distributions to Add", min_value=1, max_value=50, value=5)

# Initialize cumulative sum
U = np.zeros(500000)  # Initialize array for summed distributions

# Add distributions and calculate the cumulative sum
for i in range(1, n + 1):
    x = uniform_distribution(500000)  # Generate uniform distribution
    x_c = x - np.mean(x)  # Center the data by subtracting the mean
    U += x_c  # Add to cumulative sum

# Calculate the final standard deviation of the cumulative sum
std_dev = np.std(U / n)

# Perform Kernel Density Estimation (KDE) for smoother plot
kde = gaussian_kde(U / n, bw_method='silverman')  # Silverman’s method for bandwidth selection
x_values = np.linspace(-5, 5, 1000)  # X values for KDE plot
kde_values = kde(x_values)  # Calculate KDE values for the range of x

# Create the Plotly figure
fig = go.Figure()

# Plot the smoothed KDE curve
fig.add_trace(go.Scatter(
    x=x_values, 
    y=kde_values, 
    mode='lines',
    name=f"After {n} Distributions (Std Dev: {std_dev:.4f})",
    line=dict(color='blue', width=2)
))

# Update layout for Plotly figure
fig.update_layout(
    title=f"Cumulative Sum of {n} Uniform Distributions (Smoothed)",
    xaxis_title="Value",
    yaxis_title="Density",
    template='plotly',
    showlegend=True,
    height=600,
    width=800
)

# Display the Plotly figure
st.plotly_chart(fig)

# Display the standard deviation of the cumulative sum
st.subheader(f"Standard Deviation of Cumulative Sum (After {n} Distributions)")
st.write(f"{std_dev:.4f}")
