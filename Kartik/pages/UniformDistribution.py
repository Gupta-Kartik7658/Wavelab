import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime


st.set_page_config(page_title="Exponential Distribution", initial_sidebar_state="collapsed", page_icon="Kartik/logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)

def randomRange(seed=101,a=7**5,c=19,m=2147483647,p=0,q=1):
    l = [seed]
    u = [l[-1]/m]
    r = [p + (q-p)*u[-1]]
    st.write(r)
    for i in range(1,200000):
        l.append((a*l[-1]+c)%m)
        u.append(float(l[-1]/m))
        r.append(p + float((q-p)*u[-1]))
    st.write(r)
    return r

def uniform_distribution(col1,col2,a = 0,b = 1,xlim = 1):
    current_time = datetime.now()
    seed = int((current_time.timestamp() % 1) * 100000000) % 10000
    
    fig1, ax1 = plt.subplots(figsize=(6, 3))  
    fig2, ax2 = plt.subplots(figsize=(8 ,3))

    Z = randomRange(seed=seed,p=a,q=b)
    ax1.hist(Z, bins=100, histtype="stepfilled", density="True", range=[-xlim, xlim])
    ax1.set_xlabel('X')
    ax1.set_ylabel('$f_x(x)$')
    ax1.set_title('Standard Gaussian Distribution PDF')
    ax1.grid(True)  # Add grid to the PDF plot
    ax1.set_facecolor('#f7f7f7')  # Set background color
    
    
    ax2.hist(Z, bins=100, cumulative=True, density=True, alpha=0.6, color='g', range=[-xlim, xlim])
    ax2.set_xlabel('X')
    ax2.set_ylabel('$F_x(x)$')
    ax2.set_title('Standard Gaussian Distribution CDF')
    ax2.grid(True)  
    ax2.set_facecolor('#f7f7f7')  

   
    with col1:
        st.pyplot(fig1, use_container_width=False)
    with col2:
        st.pyplot(fig2, use_container_width=False)
    return Z

col = st.columns(2)
uniform_distribution(col[0],col[1])