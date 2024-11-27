import streamlit as st
import numpy as np
import plotly.express as px
import pathlib



st.set_page_config(page_title="Gaussian Distribution", initial_sidebar_state="collapsed", page_icon="Kartik/logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)

def gaussian_distribution(col1,col2,mu=0, sigma=1, N=2000000,xlim=5):
   
    U1 = np.random.rand(N)
    U2 = np.random.rand(N)

    
    Z1 = np.sqrt(-2 * np.log(U1))
    Z2 = np.cos(2 * np.pi * U2)
    
    
    Z = mu + sigma * (Z1 * Z2)
    
    fig1 = px.histogram(Z, nbins=100, title="Histogram of Central Limit Theorem", histnorm="probability density")
    fig1.update_layout(
        xaxis_title="Value",
        yaxis_title="Probability",
        bargap=0.1,
        showlegend=False,
        xaxis_range=[-xlim, xlim],           
    )
    
    fig1 = px.histogram(Z, nbins=100, title="Histogram of Central Limit Theorem", histnorm="cumulative density")
    fig1.update_layout(
        xaxis_title="Value",
        yaxis_title="Cumulative Distributive",
        bargap=0.1,
        showlegend=False,
        xaxis_range=[-xlim, xlim],           
    ) 

   
    # ax1.hist(Z, bins=100, histtype="stepfilled", density="True", range=[-xlim, xlim])
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('$f_x(x)$')
    # ax1.set_title('Standard Gaussian Distribution PDF')
    # ax1.grid(True)  # Add grid to the PDF plot
    # ax1.set_facecolor('#f7f7f7')  # Set background color
    
    
    ax2.hist(Z, bins=100, cumulative=True, density=True, alpha=0.6, color='g', range=[-xlim, xlim])
    ax2.set_xlabel('X')
    ax2.set_ylabel('$F_x(x)$')
    ax2.set_title('Standard Gaussian Distribution CDF')
    ax2.grid(True)  
    ax2.set_facecolor('#f7f7f7')  

   
    with col1:
        st.plotly_chart(fig1)
    with col2:
        st.pyplot(fig2, use_container_width=False)
    return Z


st.markdown(f"""
        <div class="title-container">
            <h2>Gaussian Distribution (Bell-Curve)</h2>
        </div>
        """, unsafe_allow_html=True)


col1, col2 = st.columns(2)


with col1:
    st.markdown('''
    The Gaussian curve, also known as the normal distribution curve, is a continuous probability distribution that is symmetric around its mean ($\\mu$) and characterized by its bell shape. The curve's spread is determined by the standard deviation ($\\sigma$), where about 68% of the data falls within one standard deviation of the mean, 95% within two, and 99.7% within three. The probability density function (PDF) for a Gaussian distribution is:

    $$ 
    f(x) = \\frac{1}{\\sqrt{2 \\pi \\sigma^2}} e^{ -\\frac{(x - \\mu)^2}{2 \\sigma^2} }
    $$
    The cumulative distribution function (CDF), which represents the probability that a random variable $X$ will be less than or equal to a given value $x$, is:

    $$ 
    F(x) = \\frac{1}{2} \\left[1 + \\text{erf} \\left( \\frac{x - \\mu}{\\sqrt{2 \\sigma^2}} \\right) \\right]
    $$

    In this formula:
    - $\\mu$: the mean, representing the central value.
    - $\\sigma$: the standard deviation, indicating the spread of the distribution.

    This function is often used in statistics and machine learning for data analysis and modeling.
    ''', unsafe_allow_html=True)


with col2:
    mu = st.slider('Mean ($\mu$)', -10.0, 10.0, 0.0)  
    sigma = st.slider('Standard Deviation ($\sigma$)', 0.1, 10.0, 1.0)  
    N = st.slider('Number of Samples ($N$)', 100000, 10000000, 2000000, step=10000)  
    xlim = st.slider('X-Axis Limit', 5.0, 40.0, step=2.5) 


st.markdown(f"""<br><br>""",unsafe_allow_html=True)
col1,col2 = st.columns(2)
Z = gaussian_distribution(col1,col2,mu, sigma, N,xlim)

