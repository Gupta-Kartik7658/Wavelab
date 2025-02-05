import streamlit as st
import numpy as np
import plotly.express as px
import pathlib

st.set_page_config(page_title="Laplacian Distribution", initial_sidebar_state="collapsed", page_icon="Kartik/logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)

def laplacian_distribution(col1,col2,mu=0, b=1, N=20000,xlim=5):
    Z = np.random.laplace(mu,b,N)
    
    fig1 = px.histogram(Z, nbins=100, title="Probability Distributive Function", histnorm="probability density")
    fig1.update_layout(
        xaxis_title="Value",
        yaxis_title="Probability",
        bargap=0.1,
        showlegend=False,
        xaxis_range=[-xlim, xlim],           
    )
    
    fig2 = px.histogram(Z, nbins=100, title="Cumulative Distribution Function", histnorm="probability density",cumulative=True)
    fig2.update_layout(
        xaxis_title="Value",
        yaxis_title="Cumulative probability",
        bargap=0.1,
        showlegend=False,
        xaxis_range=[-xlim, xlim],           
    ) 
    
    with col1:
        st.plotly_chart(fig1)
    with col2:
        st.plotly_chart(fig2)

st.markdown(f"""
        <div class="title-container">
            <h2>Laplacian Distribution</h2>
        </div>
        """, unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('''
    The Laplacian curve, also known as the double-exponential distribution, is a continuous probability distribution that is symmetric around its mean ($\\mu$) and characterized by its sharp peak at the mean and heavy tails. The curve's spread is determined by the scale parameter ($b$), which controls the width of the distribution. Unlike the normal distribution, the Laplacian distribution has sharper peaks and heavier tails.

    The probability density function (PDF) for a Laplacian distribution is:

    $$ 
    f(x) = \\frac{1}{2b} e^{-\\frac{|x - \\mu|}{b}}
    $$
     The cumulative distribution function (CDF), which represents the probability that a random variable $X$ will be less than or equal to a given value $x$, is:

    $$ 
    F(x) = 
    \\begin{cases} 
        0.5 e^{\\frac{x - \\mu}{b}} & \\text{for } x < \\mu \\\\
        1 - 0.5 e^{-\\frac{x - \\mu}{b}} & \\text{for } x \\geq \\mu 
    \\end{cases}
    $$

    In this formula:
    - $\\mu$: the mean, representing the central value.
    - $b$: the scale parameter, indicating the spread of the distribution.

    This function is often used in statistics, machine learning, and signal processing, especially in situations where extreme deviations or outliers are significant.
''', unsafe_allow_html=True)


with col2:
    mu = st.slider('Mean ($\mu$)', -10.0, 10.0, 0.0)  
    sigma = st.slider('The Scale Parameter ($b$)', 0.1, 10.0, 1.0)  
    N = st.slider('Number of Samples ($N$)', 1000, 100000, 20000, step=1000)  
    xlim = st.slider('X-Axis Limit', 5.0, 60.0, step=2.5) 

st.markdown(f"""<br><br>""",unsafe_allow_html=True)
col1,col2 = st.columns(2)
laplacian_distribution(col1,col2,mu, sigma, N,xlim)
