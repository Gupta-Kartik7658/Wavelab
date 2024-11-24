import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import imageio.v3 as im

# Set page configuration
st.set_page_config(page_title="Image Distribution", initial_sidebar_state="collapsed", 
                   page_icon="Kartik/logo (1).png", layout="wide")

# Load CSS for styling
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("Kartik/style.css")
load_css(css_path)

# Title container
st.markdown("""
<div class="title-container">
    <h2>Image Distribution</h2>
</div>
""", unsafe_allow_html=True)

# Center-align the submit button using CSS for Streamlit's button

# Function to display image distributions
def image_distribution(img):
    mat = im.imread(img)
    
    # Extract color channels
    r = mat[:, :, 0].ravel()
    g = mat[:, :, 1].ravel()
    b = mat[:, :, 2].ravel()
    
    # Convert to grayscale
    grayscale = np.dot(mat[..., :3], [0.2989, 0.5870, 0.1140]).ravel()

    # Plotting histograms
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.hist(r, bins=100, histtype="stepfilled", density=True, alpha=0.6, color='r', range=[0, 255])
    ax1.set_title("Red Channel Distribution")
    ax1.grid(True)
   
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(g, bins=100, histtype="stepfilled", density=True, alpha=0.6, color='g', range=[0, 255])
    ax2.set_title("Green Channel Distribution")
    ax2.grid(True)
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.hist(b, bins=100, histtype="stepfilled", density=True, alpha=0.6, color='b', range=[0, 255])
    ax3.set_title("Blue Channel Distribution")
    ax3.grid(True)

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.hist(grayscale, bins=100, histtype="stepfilled", density=True, alpha=0.6, color='black', range=[0, 255])
    ax4.set_title("Grayscale Distribution")
    ax4.grid(True)
    
    # Display histograms in columns
    col1, col2 = st.columns(2) 
    col3, col4 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)
    with col3:
        st.pyplot(fig3)
    with col4:
        st.pyplot(fig4)

# Form for file upload and button submission
with st.form(key="upload_form"):
    img_uploader = st.file_uploader(label="Upload Image", type=["jpeg", "jpg", "png", "webp"])
    
    # Center-aligned submit button inside form
    l = st.columns(11)
    with l[5]:
        submit_button = st.form_submit_button(label="Submit Image", help="Click to analyze image")

# Process the image after form submission
if submit_button and img_uploader is not None:
    # Display progress bar
    mybar = st.progress(0, "Analyzing the Image")  
    for i in range(101):
        time.sleep(0.01)
        mybar.progress(i)
    time.sleep(1)
    mybar.empty()
    
    # Display the image distributions
    image_distribution(img_uploader)
