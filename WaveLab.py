import streamlit as st
import pathlib
from PIL import Image
import base64
import os



st.set_page_config(page_title="WaveLab",initial_sidebar_state="collapsed",page_icon="logo (1).png",layout="wide")

def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("style.css")
load_css(csspath)


# Convert the image to Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/png;base64,{b64_string}"


logo_base64 = get_base64_image("logo (1).png") 


st.markdown(
    f"""
    <div class="title-container">
        <img src="{logo_base64}" class="logo-pop" alt="WaveLAB Logo">
        <br>
        <br>
        <p>A Personalised Interactive Tool-kit exclusively for IIIT Vadodara Students</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Display buttons horizontally with proper centering
st.markdown("""
<div class = "button-container">
    <a href="Probability" target="_self" class = "button-container-probability">Probability</a>
    <a href="Statistics" target="_self" class = "button-container-statistics">Statistics</a>
    <a href="ImageProcessing" target="_self" class = "button-container-physics">Image Processing</a>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div class = "button-container">
    <a href="Probability" target="_self" class = "button-container-probability">Numerical Techniques</a>
    <a href="Statistics" target="_self" class = "button-container-statistics">Linear Algebra</a>
    <a href="Physics" target="_self" class = "button-container-physics">Physics</a>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="title-container">
        <br>
        <br>
        <br>
        <br>
        <br>
        <br>
        <p>Developed By : </p>
        <p>Kartik Gupta (202351056)</p>
        <p>Abhinav Chhajed (202351001)</p>
    </div>
    """,
    unsafe_allow_html=True
)