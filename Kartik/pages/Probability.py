import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
import time
import pathlib
from PIL import Image
import base64


st.set_page_config(page_title="Probability",initial_sidebar_state="collapsed",page_icon="Kartik/logo (1).png")

def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)



# Define the Gaussian Distribution function
def gaussian_distribution(mu=0, sigma=1, N=2000000):
    # Generate uniform random variables
    U1 = np.random.rand(N)
    U2 = np.random.rand(N)

    # Box-Muller Transform to generate Gaussian distributed values
    Z1 = np.sqrt(-2 * np.log(U1))
    Z2 = np.cos(2 * np.pi * U2)
    
    # Apply the Gaussian transformation
    Z = mu + sigma * (Z1 * Z2)

    # Create the PDF and CDF plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the PDF (Probability Density Function)
    ax1.hist(Z, bins=100, histtype="stepfilled",density="True",range=[-40,40])
    ax1.set_xlabel('X')
    ax1.set_ylabel('$f_x(x)$')
    ax1.set_title('Standard Gaussian Distribution PDF')
    ax1.grid(True)  # Add grid to the PDF plot
    ax1.set_facecolor('#f7f7f7')  # Set background color
    
    

    # Plot the CDF (Cumulative Distribution Function)
    ax2.hist(Z, bins=50, cumulative=True, density=True, alpha=0.6, color='g',range=[-40,40])
    ax2.set_xlabel('X')
    ax2.set_ylabel('$F_x(x)$')
    ax2.set_title('Standard Gaussian Distribution CDF')
    ax2.grid(True)  
    ax2.set_facecolor('#f7f7f7')  

    
    st.pyplot(fig)


st.title("Gaussian Distribution Generator")


mu = st.slider('Mean (mu)', -10.0, 10.0, 0.0)  
sigma = st.slider('Standard Deviation (sigma)', 0.1, 10.0, 1.0)  
N = st.slider('Number of Samples (N)', 100000, 10000000, 2000000, step=10000)  

# Automatically generate the Gaussian distribution as the user adjusts sliders
gaussian_distribution(mu, sigma, N)
