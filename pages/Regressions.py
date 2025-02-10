import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Streamlit app title
st.title("Interactive Linear and Polynomial Regression")

# Explanation
st.markdown("""
## Understanding Regression
Regression analysis is a fundamental statistical technique used to model relationships between variables. It helps in predicting values based on observed data.

### Types of Regression:
- **Linear Regression**: Fits data with a straight line.
- **Polynomial Regression**: Fits data with a polynomial curve for better flexibility and capturing more complex relationships.

### How to Use This Tool:
1. Adjust the "Number of Data Points" slider to control the dataset size.
2. Modify the "Polynomial Degree" slider to observe how different degrees affect curve fitting.
3. Analyze the plotted polynomial fit and its relationship to the original data.

This interactive visualization helps in understanding how regression models approximate real-world data!
""")

# Sidebar for user input
datapoints = st.slider("Number of data points", 10, 500, 100)
poly_degree = st.slider("Polynomial Degree", 1, 20, 2)

# Initialize x and y only once
if "x" not in st.session_state or "y" not in st.session_state or st.session_state.datapoints != datapoints:
    st.session_state.x = np.linspace(-10, 10, datapoints)
    coefficients = np.random.randn(21) * (2000 / (10 ** 20))  # Generate coefficients once for max degree
    st.session_state.y = np.polyval(coefficients[:poly_degree + 1], st.session_state.x) + (np.random.randn(datapoints) * 500)
    st.session_state.datapoints = datapoints  # Store datapoints to detect change

# Use stored x and y values
x = st.session_state.x
y = st.session_state.y

# Fit polynomial regression
p = np.polyfit(x, y, poly_degree)

# Generate new x values for smooth curve
x_new = np.linspace(min(x), max(x), 100)
y_fit = np.polyval(p, x_new)

# Create Plotly figure
fig = go.Figure()

# Scatter plot for data points
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='red'), name='Data Points'))

# Line plot for polynomial fit
fig.add_trace(go.Scatter(x=x_new, y=y_fit, mode='lines', line=dict(color='blue', width=2), name=f'Polynomial Fit (Degree {poly_degree})'))

# Update layout
fig.update_layout(
    title='Polynomial Curve Fitting',
    xaxis_title='X-axis',
    yaxis_title='Y-axis',
    yaxis=dict(range=[-2000, 2000]),  # Restrict y-axis
    showlegend=True,
    template='plotly_white'
)

# Show plot in Streamlit
st.plotly_chart(fig)
