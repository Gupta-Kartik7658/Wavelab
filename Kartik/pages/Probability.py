import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
import time
import pathlib
from PIL import Image
import base64
from pathlib import Path

st.set_page_config(page_title="WaveLab",initial_sidebar_state="collapsed",page_icon="Kartik/logo (1).png")

def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("Kartik/style.css")
load_css(csspath)

st.markdown("""
<div class = "button-container">
    <a href="GaussianDistribution" target="_self" class = "button-container-probability">Gaussian Distribution</a>
</div>""",unsafe_allow_html=True)