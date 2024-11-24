import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pathlib




st.set_page_config(page_title="Laplacian Distribution", initial_sidebar_state="collapsed", page_icon="Kartik/logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)


def laplacian_distribution(col1,col2,mu=0, b=1, N=2000000,xlim=5):
   
    U1 = np.random.rand(N)

    
    Z = mu - b * np.sign(U1 - 0.5) * np.log(1 - 2 * np.abs(U1 - 0.5))

   
    fig1, ax1 = plt.subplots(figsize=(8, 3))  
    fig2, ax2 = plt.subplots(figsize=(8 ,3))

   
    ax1.hist(Z, bins=100, histtype="stepfilled", density="True", range=[-xlim, xlim])
    ax1.set_xlabel('X')
    ax1.set_ylabel('$f_x(x)$')
    ax1.set_title('Standard Laplacian Distribution PDF')
    ax1.grid(True)  # Add grid to the PDF plot
    ax1.set_facecolor('#f7f7f7')  # Set background color
    
    
    ax2.hist(Z, bins=100, cumulative=True, density=True, alpha=0.6, color='g', range=[-xlim, xlim])
    ax2.set_xlabel('X')
    ax2.set_ylabel('$F_x(x)$')
    ax2.set_title('Standard Laplacian Distribution CDF')
    ax2.grid(True)  
    ax2.set_facecolor('#f7f7f7')  

   
    with col1:
        st.pyplot(fig1, use_container_width=False)
    with col2:
        st.pyplot(fig2, use_container_width=False)



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
    N = st.slider('Number of Samples ($N$)', 100000, 10000000, 2000000, step=10000)  
    xlim = st.slider('X-Axis Limit', 5.0, 60.0, step=2.5) 


st.markdown(f"""<br><br>""",unsafe_allow_html=True)
col1,col2 = st.columns(2)
laplacian_distribution(col1,col2,mu, sigma, N,xlim)
