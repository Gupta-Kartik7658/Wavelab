import streamlit as st
import numpy as np
import plotly.express as px
import pathlib


st.set_page_config(page_title="Exponential Distribution", initial_sidebar_state="collapsed", page_icon="logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("style.css")
load_css(csspath)


def Exponential_distribution(col1,col2, lambda_v=1, N=20000,xlim=5):
   
    Z =  np.random.exponential(lambda_v,N)

   
    fig1 = px.histogram(Z, nbins=100, title="Probability Distributive Function", histnorm="probability density")
    fig1.update_layout(
        xaxis_title="Value",
        yaxis_title="Probability",
        bargap=0.1,
        showlegend=False,
        xaxis_range=[0, xlim],           
    )
    
    fig2 = px.histogram(Z, nbins=100, title="Cumulative Distributive Function", histnorm="probability density",cumulative=True)
    fig2.update_layout(
        xaxis_title="Value",
        yaxis_title="Cumulative probability",
        bargap=0.1,
        showlegend=False,
        xaxis_range=[0, xlim],           
    ) 
    
    with col1:
        st.plotly_chart(fig1)
    with col2:
        st.plotly_chart(fig2)



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
    N = st.slider('Number of Samples ($N$)', 1000, 100000, 20000, step=1000)  
    xlim = st.slider('X-Axis Limit', 5.0, 40.0, step=2.5) 


st.markdown(f"""<br><br>""",unsafe_allow_html=True)
col1,col2 = st.columns(2)
Exponential_distribution(col1,col2, lambda_v, N,xlim)
