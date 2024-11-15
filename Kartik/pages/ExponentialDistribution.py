import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
import time
import pathlib
from PIL import Image
import base64


st.set_page_config(page_title="Exponential Distribution", initial_sidebar_state="collapsed", page_icon="Kartik/logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)


def Exponential_distribution(col1,col2, lambda_v=1, N=2000000,xlim=5):
   
    U = np.random.rand(N)

    
    Z =  -np.log(U)/lambda_v

   
    fig1, ax1 = plt.subplots(figsize=(8, 3))  
    fig2, ax2 = plt.subplots(figsize=(8 ,3))

   
    ax1.hist(Z, bins=100, histtype="stepfilled", density="True", range=[0, xlim])
    ax1.set_xlabel('X')
    ax1.set_ylabel('$f_x(x)$')
    ax1.set_title('Standard Exponential Distribution PDF')
    ax1.grid(True)  # Add grid to the PDF plot
    ax1.set_facecolor('#f7f7f7')  # Set background color
    
    
    ax2.hist(Z, bins=100, cumulative=True, density=True, alpha=0.6, color='g', range=[0, xlim])
    ax2.set_xlabel('X')
    ax2.set_ylabel('$F_x(x)$')
    ax2.set_title('Standard Exponential Distribution CDF')
    ax2.grid(True)  
    ax2.set_facecolor('#f7f7f7')  

   
    with col1:
        st.pyplot(fig1, use_container_width=False)
    with col2:
        st.pyplot(fig2, use_container_width=False)



st.markdown(f"""
        <div class="title-container">
            <h2>Exponential Distribution</h2>
        </div>
        """, unsafe_allow_html=True)


col1, col2 = st.columns(2)


with col1:
    st.markdown('''
    The Exponential distribution is a continuous probability distribution often used to model the time between events in a Poisson process, where events occur continuously and independently at a constant average rate. The Exponential distribution is characterized by a constant rate parameter ($\\lambda$), which determines the spread of the distribution. Unlike the Gaussian distribution, the Exponential distribution is asymmetric, with a peak near zero and a long tail to the right.

    The probability density function (PDF) for an Exponential distribution is:

    $$ 
    f(x) = \\lambda e^{-\\lambda x}, \\quad x \\geq 0
    $$
    The cumulative distribution function (CDF), which gives the probability that a random variable $X$ will be less than or equal to a given value $x$, is:

    $$ 
    F(x) = 1 - e^{-\\lambda x}, \\quad x \\geq 0
    $$

    In this formula:
    - $\\lambda$: the rate parameter, representing the frequency of occurrences.
    - $x$: the variable representing time or distance until the next event.

    The mean of the Exponential distribution is given by $1/\\lambda$. This distribution is commonly used in statistics, reliability engineering, and queuing theory to model waiting times and lifetimes of products.
''', unsafe_allow_html=True)



with col2:
    # mu = st.slider('Mean ($\mu$)', -10.0, 10.0, 0.0)  
    lambda_v = st.slider('Lambda Value ($\lambda$)', 0.1, 10.0, 1.0)  
    N = st.slider('Number of Samples ($N$)', 100000, 10000000, 2000000, step=10000)  
    xlim = st.slider('X-Axis Limit', 5.0, 40.0, step=2.5) 


st.markdown(f"""<br><br>""",unsafe_allow_html=True)
col1,col2 = st.columns(2)
Exponential_distribution(col1,col2, lambda_v, N,xlim)
