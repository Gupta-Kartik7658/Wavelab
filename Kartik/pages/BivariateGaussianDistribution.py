import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go 
from scipy.integrate import dblquad
import pathlib


st.set_page_config(page_title="Bivariate Gaussian Distribution", initial_sidebar_state="collapsed", page_icon="Kartik/logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)


import numpy as np
import plotly.graph_objects as go
from scipy.integrate import dblquad
import streamlit as st

def bivariate_normal_distribution(col1, col2, mx=3, my=5, stdx=0.25, stdy=0.5, xlim=5, ylim=5, grid_points=50):
    x_vals = np.linspace(-xlim, xlim, grid_points)
    y_vals = np.linspace(-ylim, ylim, grid_points)
    x, y = np.meshgrid(x_vals, y_vals)

    # Calculate PDF values
    X = (x - mx) / stdx
    Y = (y - my) / stdy
    fxy = (1 / (2 * np.pi * stdx * stdy)) * np.exp(-0.5 * (X**2 + Y**2))

    # Calculate CDF values
    def integrand(x, y):
        X = (x - mx) / stdx
        Y = (y - my) / stdy
        return (1 / (2 * np.pi * stdx * stdy)) * np.exp(-0.5 * (X**2 + Y**2))

    cdf = np.zeros_like(fxy)
    for i in range(grid_points):
        for j in range(grid_points):
            # Integrate from -infinity to x, -infinity to y
            cdf[i, j], _ = dblquad(
                integrand,
                -xlim, x_vals[j],  # x limits from -xlim to x_val[j]
                -ylim, y_vals[i]   # y limits from -ylim to y_val[i]
            )

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
        title="PDF of Bivariate Normal Distribution",
        width = 800,
        height = 800, 
            
        # plot_bgcolor="white",  # Set the inner plot background color to white
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='f(x, y)',
            # xaxis=dict(range=[-xlim, xlim]),
            # yaxis=dict(range=[-ylim, ylim]),
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
        title="CDF of Bivariate Normal Distribution",
        width = 800,
        height = 800,
        # font=dict(color="black"),  # Set text color to black
        # plot_bgcolor="white",  # Set the inner plot background color to white
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='F(x, y)',
            # xaxis=dict(range=[-xlim, xlim]),
            # yaxis=dict(range=[-ylim, ylim]),
        )
    )

    # Display the plots in Streamlit
    with col1:
        st.plotly_chart(pdf_fig, use_container_width=True)
    with col2:
        st.plotly_chart(cdf_fig, use_container_width=True)

# Example usage in Streamlit:
# bivariate_normal_distribution(col1, col2)


# Example usage in Streamlit:

st.markdown(f"""
        <div class="title-container">
            <h2>Standard Bivariate Gaussian Distribution</h2>
        </div>
        """, unsafe_allow_html=True)


col1, col2 = st.columns(2)
with col1:
    st.markdown(''' 
    The **bivariate standard normal distribution** is a two-dimensional extension of the standard normal distribution. It describes the joint probability of two random variables, $X$ and $Y$, that are both normally distributed with a mean of $0$ and a standard deviation of $1$. The distribution is symmetric and bell-shaped in two dimensions and is centered at the origin $(0, 0)$. 

    The probability density function (PDF) for the bivariate standard normal distribution is given by:

    $$ 
    f(x, y) = \\frac{1}{2 \\pi} e^{-\\frac{1}{2}(x^2 + y^2)}
    $$

    In this formula:
    - $x$: the value for the first random variable.
    - $y$: the value for the second random variable.
    - The exponent $-\\frac{1}{2}(x^2 + y^2)$ ensures that the distribution is bell-shaped and centered at the origin.

    The **cumulative distribution function (CDF)** of the bivariate normal distribution represents the probability that the two random variables, $X$ and $Y$, simultaneously fall within a specified region. It can be expressed as an integral of the PDF:

    $$ 
    F(x, y) = \\int_{-\\infty}^{x} \\int_{-\\infty}^{y} f(u, v) \, du \, dv
    $$

    This CDF gives the probability that the random variables $X$ and $Y$ take values less than or equal to $x$ and $y$, respectively. 

    In the case of the **standard bivariate normal distribution**, both $X$ and $Y$ have a mean of $0$ and a standard deviation of $1$, and they are uncorrelated, meaning that their covariance is zero.

    This distribution is widely used in multivariate statistics, machine learning, and data analysis, especially in modeling correlated random variables.
    ''', unsafe_allow_html=True)

with col2:
    mux = st.slider('Mean in x ($\mu_x$)', -10.0, 10.0, 0.0)  
    st.markdown(f"""<br>""", unsafe_allow_html=True)
    muy = st.slider('Mean in y ($\mu_y$)', -10.0, 10.0, 0.0)  
    st.markdown(f"""<br>""", unsafe_allow_html=True)
    stdx = st.slider('Standard Deviation ($\sigma_x$)', 0.1, 10.0, 1.0) 
    st.markdown(f"""<br>""", unsafe_allow_html=True)
    stdy = st.slider('Standard Deviation ($\sigma_y$)', 0.1, 10.0, 1.0) 
    # xlim = st.slider('X-Axis Limit', 5.0, 40.0, step=2.5) 
    # ylim = st.slider('Y-Axis Limit', 5.0, 40.0, step=2.5) 
    # grid_points = st.slider('Grid Points', 5, 50, 30, step=5) 

st.markdown(f"""<br>""", unsafe_allow_html=True)
col1, col2 = st.columns(2)
bivariate_normal_distribution(col1, col2, mux, muy, stdx, stdy)
