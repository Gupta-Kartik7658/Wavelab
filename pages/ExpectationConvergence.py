import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pathlib
from scipy.stats import binom

st.set_page_config(page_title="Coin Toss Simulation - WLLN", initial_sidebar_state="collapsed", page_icon="logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("style.css")
load_css(csspath)


def simulate_coin_tosses(n, p):
    tosses = np.random.rand(n) < p  # 1 if head, 0 if tail
    cumulative_heads = np.cumsum(tosses)
    running_mean = cumulative_heads / np.arange(1, n + 1)
    return tosses, running_mean


def binomial_distribution_plot(n, p):
    x = np.arange(0, n + 1)
    probs = binom.pmf(x, n, p)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=probs, name="Binomial PMF"))
    fig.add_vline(x=n * p, line_dash="dash", line_color="red", annotation_text=f"Expectation np={n*p:.1f}")
    fig.update_layout(
        title=f"Binomial Distribution of Heads in {n} Tosses",
        xaxis_title="Number of Heads",
        yaxis_title="Probability",
        showlegend=False
    )
    return fig


def WLLN_simulation(col1, col2, n=1000, p=0.5):
    # Single run for convergence demonstration
    _, running_mean = simulate_coin_tosses(n, p)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=running_mean, mode="lines", name="Running Mean", line=dict(width=3)))
    fig1.add_hline(y=p, line_dash="dash", line_color="red", annotation_text=f"Expectation p={p}")
    fig1.update_layout(
        title="Convergence of Sample Mean to Expectation",
        xaxis_title="Number of Tosses",
        yaxis_title="Proportion of Heads",
        yaxis=dict(range=[0, 1]),
        showlegend=False,
    )

    fig2 = binomial_distribution_plot(n, p)

    with col1:
        st.plotly_chart(fig1)
    with col2:
        st.plotly_chart(fig2)


st.markdown(f"""
        <div class="title-container">
            <h2>Coin Toss Simulation</h2>
        </div>
        """, unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('''
    Imagine tossing a coin multiple times. Each toss can either result in **Head (1)** or **Tail (0)**.

    If the coin has a bias $p$ (probability of heads), then the **expected value** of a single toss is simply:

    $$ E[X] = 1 \cdot p + 0 \cdot (1-p) = p $$

    This means that on average, we expect a fraction $p$ of the tosses to result in heads. However, in small samples, randomness can cause the observed proportion to be far from $p$.

    As the number of tosses increases, these fluctuations tend to smooth out, and the running average of heads stabilizes closer to the true expectation $p$.

    Another way to view this is through the **distribution of outcomes**: if we repeat the experiment many times, the number of heads in $n$ tosses follows a **Binomial distribution** with parameters $(n, p)$. This distribution becomes more sharply centered around its mean $np$ as $n$ grows, showing that large samples are more likely to be close to the expected value.

    This app demonstrates both perspectives:
    - The left plot shows how the running proportion of heads approaches $p$ as we increase the number of tosses.
    - The right plot shows the Binomial distribution for $n$ tosses, highlighting how the most likely outcomes cluster around the expected number of heads.
''', unsafe_allow_html=True)

with col2:
    p = st.slider('Bias of Coin (p)', 0.0, 1.0, 0.5, step=0.05)
    n = st.slider('Number of Tosses (n)', 10, 10000, 10, step=500)

    tosses, running_mean = simulate_coin_tosses(n, p)
    empirical_mean = np.mean(tosses)

    st.metric("Empirical Mean (Observed)", f"{empirical_mean:.3f}", delta=f"vs Expectation {p:.2f}")

st.markdown(f"""<br><br>""", unsafe_allow_html=True)
col1, col2 = st.columns(2)
WLLN_simulation(col1, col2, n, p)
