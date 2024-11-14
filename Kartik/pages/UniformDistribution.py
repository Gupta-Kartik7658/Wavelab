import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime


st.set_page_config(page_title="Uniform Distribution", initial_sidebar_state="collapsed", page_icon="Kartik/logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)

def randomRange(seed=101,a=7**5,c=19,m=2147483647,p=0,q=1):
    l = [seed]
    u = [l[-1]/m]
    r = [p + (q-p)*u[-1]]
    
    for i in range(1,200000):
        l.append((a*l[-1]+c)%m)
        u.append(float(l[-1]/m))
        r.append(p + float((q-p)*u[-1]))
    
    return r

def uniform_distribution(col1,col2,a = 0,b = 1,xlim = 1):
    current_time = datetime.now()
    seed = int((current_time.timestamp() % 1) * 100000000) % 10000
    
    fig1, ax1 = plt.subplots(figsize=(3, 2))  
    fig2, ax2 = plt.subplots(figsize=(3 ,2))

    Z = randomRange(seed=seed,p=a,q=b)
    ax1.hist(Z, bins=100, histtype="stepfilled", density="True", range=[min(-xlim,0), xlim])
    ax1.set_xlabel('X')
    ax1.set_ylabel('$f_x(x)$')
    ax1.set_title('Standard Uniform Distribution PDF')
    ax1.grid(True)  # Add grid to the PDF plot
    ax1.set_facecolor('#f7f7f7')  # Set background color
    
    
    ax2.hist(Z, bins=100, cumulative=True, density=True, alpha=0.6, color='g', range=[min(-xlim,0), xlim])
    ax2.set_xlabel('X')
    ax2.set_ylabel('$F_x(x)$')
    ax2.set_title('Standard Uniform Distribution CDF')
    ax2.grid(True)  
    ax2.set_facecolor('#f7f7f7')  

   
    with col1:
        st.pyplot(fig1, use_container_width=False)
    with col2:
        st.pyplot(fig2, use_container_width=False)

st.markdown(f"""
        <div class="title-container">
            <h2>Uniform Distribution (Bell-Curve)</h2>
        </div>
        """, unsafe_allow_html=True)


col1, col2 = st.columns(2)


with col1:
    st.markdown('''
    The *Uniform distribution* is a type of continuous probability distribution in which all outcomes are equally likely within a defined interval [a, b]. Unlike the Uniform curve, the uniform distribution has a constant probability density, represented by a flat, rectangular shape across its range. The probability density function (PDF) for a uniform distribution is:

    $$ 
    f(x) = \\frac{1}{b - a} \\quad \\text{for } a \\leq x \\leq b
    $$

    Here:
    - $a$: the minimum value of the interval, indicating the lower bound.
    - $b$: the maximum value of the interval, indicating the upper bound.

    The cumulative distribution function (CDF), which represents the probability that a random variable $X$ will be less than or equal to a given value $x$, is:

    $$ 
    F(x) = 
    \\begin{cases} 
      0 & \\text{for } x < a \\\\
      \\frac{x - a}{b - a} & \\text{for } a \\leq x \\leq b \\\\
      1 & \\text{for } x > b 
    \\end{cases}
    $$

    In this distribution:
    - All values within the interval [a, b] are equally likely.
    - The mean is $\\mu = \\frac{a + b}{2}$, representing the midpoint.
    - The variance is $\\sigma^2 = \\frac{(b - a)^2}{12}$, indicating the spread of values around the mean.

    The uniform distribution is often used in simulations, random sampling, and situations where each outcome in a range is equally probable.
    ''', unsafe_allow_html=True)


with col2:
    a = st.slider('Start ($\a$)', -10.0, 10.0, 0.0)  
    b = st.slider('End ($\b$)', 0.1, 10.0, 1.0)   
    xlim = st.slider('X-Axis Limit', 5.0, 40.0, step=2.5) 


col = st.columns(2)
uniform_distribution(col[0],col[1],a,b,xlim)