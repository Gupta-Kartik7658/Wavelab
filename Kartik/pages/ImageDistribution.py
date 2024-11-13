import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pathlib
from PIL import Image
import base64
import imageio.v3 as im

st.set_page_config(page_title="Exponential Distribution", initial_sidebar_state="collapsed", page_icon="Kartik/logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)

st.markdown(f"""<div class = "title-container">
            <h2>Image Distribution</h2></div>
            """)


def imageDistribution(image):
    mat = im.imread(image)