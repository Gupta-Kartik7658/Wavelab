import streamlit as st
import numpy as np
import plotly.graph_objects as go 
from scipy.integrate import dblquad
import pathlib


st.set_page_config(page_title="Bivariate Exponential Distribution", initial_sidebar_state="collapsed", page_icon="logo (1).png", layout="wide")

def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("style.css")
load_css(csspath)


def bivariate_exponential_distribution(col1, col2, lamx = 1,lamy = 1, xlim=5, ylim=5, grid_points=50):
    x_vals = np.linspace(0, xlim, grid_points)
    y_vals = np.linspace(0, ylim, grid_points)
    x, y = np.meshgrid(x_vals, y_vals)

    # Calculate PDF values
    X = x
    Y = y
    # PDF calculation
    fxy = lamx * lamy * np.exp(-(lamx * X) - (lamy * Y))

    # Calculate CDF values
    def exponential_cdf(x, rate):
        return np.where(
            x < 0,
            0,  # CDF is 0 for x < 0
            1 - np.exp(-rate * x)  # CDF for x >= 0
        )

    # Compute joint CDF
    cdf = np.zeros((grid_points, grid_points))
    for i in range(grid_points):
        for j in range(grid_points):
            # Joint CDF as the product of marginal CDFs
            cdf[i, j] = exponential_cdf(x_vals[j], 1 / lamx) * exponential_cdf(y_vals[i], 1 / lamy)


    # Plot the PDF using Plotly
    pdf_fig = go.Figure(
        data=[
            go.Surface(
                z=fxy,
                x=x_vals,
                y=y_vals,
                colorscale="Viridis",
                showscale=False
            )
        ]
    )
    pdf_fig.update_layout(
        # font=dict(color="black"),  # Set text color to black
        title="PDF of Bivariate exponential Distribution",
        width = 800,
        height = 800, 
            
        # plot_bgcolor="white",  # Set the inner plot background color to white
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='f(x, y)',
            xaxis=dict(range=[0, xlim]),
            yaxis=dict(range=[0, ylim]),
        )
    )

    # Plot the CDF using Plotly
    cdf_fig = go.Figure(
        data=[
            go.Surface(
                z=cdf,
                x=x_vals,
                y=y_vals,
                colorscale="Viridis",
                showscale=False
            )
        ]
    )
    cdf_fig.update_layout(
        title="CDF of Bivariate exponential Distribution",
        width = 800,
        height = 800,
        # font=dict(color="black"),  # Set text color to black
        # plot_bgcolor="white",  # Set the inner plot background color to white
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='F(x, y)',
            xaxis=dict(range=[0, xlim]),
            yaxis=dict(range=[0, ylim]),
        )
    )

    # Display the plots in Streamlit
    with col1:
        st.plotly_chart(pdf_fig, use_container_width=True)
    with col2:
        st.plotly_chart(cdf_fig, use_container_width=True)

# Example usage in Streamlit:
# bivariate_exponential_distribution(col1, col2)


# Example usage in Streamlit:

st.markdown(f"""
        <div class="title-container">
            <h2>Standard Bivariate Exponential Distribution</h2>
        </div>
        <br>
        """, unsafe_allow_html=True)


col1, col2 = st.columns(2)
with col1:
    st.markdown('''
    The **independent bivariate exponential distribution** describes the joint distribution of two random variables, $X$ and $Y$, 
    that are independently exponentially distributed. This distribution is particularly useful in reliability analysis, queuing theory, 
    and survival analysis.

    ### Probability Density Function (PDF)
    The **PDF** for the independent bivariate exponential distribution is given by:

    $$
    f(x, y) = 
    \\begin{cases} 
    \\lambda_x \\lambda_y \\exp(-\\lambda_x x - \\lambda_y y), & x \\geq 0, y \\geq 0 \\
    0, & \\text{otherwise}
    \\end{cases}
    $$

    where:
    - $x \\geq 0$ and $y \\geq 0$ are the values of the random variables $X$ and $Y$,
    - $\\lambda_x$ and $\\lambda_y$ are the rate parameters (inverse of the mean) for $X$ and $Y$.


    ### Key Characteristics
    - **Independence**: The joint distribution assumes $X$ and $Y$ are independent, so their joint PDF and CDF are the products of their marginal PDFs and CDFs, respectively.
    - **Shape**: The exponential distribution is memoryless, meaning the probability of an event occurring does not depend on how much time has already passed.
    - **Applications**: Common in modeling time until failure, waiting times, and reliability systems.

    ''', unsafe_allow_html=True)


with col2:
    lamx = st.slider('Rate Parameter in x ($\lambda_x$)', 0.1, 10.0, 1.0) 
    st.markdown(f"""<br>""", unsafe_allow_html=True)
    lamy = st.slider('Rate Parameter in y ($\lambda_y$)', 0.1, 10.0, 1.0) 
   

st.markdown(f"""<br>""", unsafe_allow_html=True)
col1, col2 = st.columns(2)
bivariate_exponential_distribution(col1, col2, lamx,lamy)
