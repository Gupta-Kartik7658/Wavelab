import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pathlib

st.set_page_config(page_title="Bivariate Gaussian Distribution", page_icon="logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("style2.css")
load_css(csspath)

