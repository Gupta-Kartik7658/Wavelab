import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
import time
import pathlib
from PIL import Image
import base64

st.set_page_config(initial_sidebar_state="collapsed")

def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")
        
csspath = pathlib.Path("Kartik\style.css")
load_css(csspath)