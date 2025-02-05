import streamlit as st
import numpy as np
import plotly.graph_objects as go 
from scipy.integrate import dblquad
import pathlib


st.set_page_config(page_title="Bivariate Laplacian Distribution", initial_sidebar_state="collapsed", page_icon="logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("style.css")
load_css(csspath)


def bivariate_laplacian_distribution(col1, col2, mx=3, my=5, bx=0.25, by=0.5, xlim=5, ylim=5, grid_points=50):
    x_vals = np.linspace(-xlim, xlim, grid_points)
    y_vals = np.linspace(-ylim, ylim, grid_points)
    x, y = np.meshgrid(x_vals, y_vals)

    # Calculate PDF values
    X = x
    Y = y
    # PDF calculation
    fxy = (1 / (4 * bx * by)) * np.exp(-np.abs(X - mux) / bx - np.abs(Y - muy) / by)

    # Calculate CDF values
    def laplacian_cdf(x, mu, b):
        return np.where(
            x < mu,
            0.5 * np.exp((x - mu) / b),  # CDF for x < mu
            1 - 0.5 * np.exp(-(x - mu) / b)  # CDF for x >= mu
        )

    # Compute joint CDF
    cdf = np.zeros((grid_points, grid_points))
    for i in range(grid_points):
        for j in range(grid_points):
            # Joint CDF as the product of marginal CDFs
            cdf[i, j] = laplacian_cdf(x_vals[j], mux, bx) * laplacian_cdf(y_vals[i], muy, by)


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
        title="PDF of Bivariate laplacian Distribution",
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
        title="CDF of Bivariate laplacian Distribution",
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
# bivariate_laplacian_distribution(col1, col2)


# Example usage in Streamlit:

st.markdown(f"""
        <div class="title-container">
            <h2>Standard Bivariate Laplacian Distribution</h2>
        </div>
        """, unsafe_allow_html=True)


col1, col2 = st.columns(2)
with col1:
    st.markdown('''
    The **independent bivariate Laplacian distribution** describes the joint distribution of two random variables, $X$ and $Y$, 
    that are independently Laplacian distributed. This distribution is useful for modeling data where the variables have heavy tails 
    and are uncorrelated.

    The **probability density function (PDF)** for the independent bivariate Laplacian distribution is given by:

    $$
    f(x, y) = \\frac{1}{4 b_x b_y} \\exp\\left(-\\frac{|x - \\mu_x|}{b_x} - \\frac{|y - \\mu_y|}{b_y}\\right),
    $$

    where:
    - $\\mu_x$ and $\\mu_y$ are the means of the variables $X$ and $Y$,
    - $b_x$ and $b_y$ are the scale parameters of $X$ and $Y$ (determining their spread),
    - $|x - \\mu_x|$ and $|y - \\mu_y|$ represent the absolute deviations of $x$ and $y$ from their respective means.

    ### Key Characteristics:
    - **Independence**: The joint PDF is the product of the marginal PDFs, as $X$ and $Y$ are independent.
    - **Shape**: The distribution exhibits heavy tails, characteristic of the Laplacian distribution, in each dimension.
    - **Applications**: This distribution is used in signal processing, robust statistics, and modeling noise with heavy tails.

    This independent bivariate Laplacian model is particularly useful when the two dimensions are uncorrelated but still exhibit the 
    sharp peak and heavy tails characteristic of the Laplacian distribution.
''', unsafe_allow_html=True)


with col2:
    mux = st.slider('Mean in x ($\mu_x$)', -10.0, 10.0, 0.0)  
    st.markdown(f"""<br>""", unsafe_allow_html=True)
    muy = st.slider('Mean in y ($\mu_y$)', -10.0, 10.0, 0.0)  
    st.markdown(f"""<br>""", unsafe_allow_html=True)
    bx = st.slider('Scale Parameter ($b_x$)', 0.1, 10.0, 1.0) 
    st.markdown(f"""<br>""", unsafe_allow_html=True)
    by = st.slider('Scale Parameter ($b_y$)', 0.1, 10.0, 1.0) 
    # xlim = st.slider('X-Axis Limit', 5.0, 40.0, step=2.5) 
    # ylim = st.slider('Y-Axis Limit', 5.0, 40.0, step=2.5) 
    # grid_points = st.slider('Grid Points', 5, 50, 30, step=5) 

st.markdown(f"""<br>""", unsafe_allow_html=True)
col1, col2 = st.columns(2)
bivariate_laplacian_distribution(col1, col2, mux, muy, bx, by)
