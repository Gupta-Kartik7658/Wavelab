import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
import time
import pathlib
from PIL import Image
import base64

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
        <p>A Personalised and Customizable Tool-kit exclusively for IIIT Vadodara Students</p>
    </div>
    """,
    unsafe_allow_html=True
)


# Display buttons horizontally with proper centering
st.markdown("""
<div class = "button-container">
    <a href="#" class = "button-container-probability">Probability</a>
    <a href="#" class = "button-container-statistics">Statistics</a>
    <a href="#" class = "button-container-physics">Physics</a>
</div>
""", unsafe_allow_html=True)
