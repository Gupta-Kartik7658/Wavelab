import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pathlib

st.set_page_config(page_title="Bivariate Gaussian Distribution",page_icon="logo (1).png", layout="wide")


def load_css(file):
    with open(file) as f:
        st.html(f"<style>{f.read()}</style>")

csspath = pathlib.Path("style2.css")
load_css(csspath)

# PCA Function with Exception Handling
def pca(X, k=None):
    try:
        # Validate k value
        if k is None or k < 1:
            raise ValueError("Number of principal components (k) must be at least 1.")
        if k > X.shape[1]:
            raise ValueError(f"Number of principal components (k) cannot exceed the number of features ({X.shape[1]}).")

        mu = np.mean(X, axis=0)
        X_c = X - mu  # Centering the data
        
        # Covariance matrix
        Cx = np.cov(X_c.T)
        col1,col2 = st.columns(2)
        with col1:
            st.write("### Original Covariance Matrix:")
            st.write(pd.DataFrame(Cx))
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(Cx)
        
        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top k eigenvectors
        W = eigenvectors[:, :k]
        X_p = np.dot(X_c, W)
        
        # New covariance matrix
        with col2:
            st.write(f"### New Covariance Matrix for k = {k}:")
            st.write(pd.DataFrame(np.cov(X_p.T)))
        
        return X_p, eigenvalues[:k]
    
    except ValueError as e:
        st.error(f"Error: {e}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None

# Streamlit App Layout
st.markdown(f"""
        <div class="title-container">
            <h2>Principal Component Analysis</h2>
        </div>
        """, unsafe_allow_html=True)

# Expanded Explanation Section
st.markdown("""
### Understanding Principal Component Analysis (PCA)

**Principal Component Analysis (PCA)** is a powerful dimensionality reduction technique widely used in data analysis and machine learning. It works by transforming the data into a new set of variables, the **principal components**, which are linear combinations of the original features. These components are chosen to maximize the variance and retain as much information as possible.

**Key Concepts**:
1. **Data Centering**: Before applying PCA, the data is centered by subtracting the mean of each feature.
2. **Covariance Matrix**: Measures how changes in one feature are associated with changes in another.
3. **Eigenvalues and Eigenvectors**: Derived from the covariance matrix, they determine the principal components' directions and magnitudes.
4. **Dimensionality Reduction**: By selecting the top \( k \) principal components, we project the data into a lower-dimensional space while preserving the most significant variance.

""")

# Data Upload or Generation
st.sidebar.header("Data Input")
data_option = st.sidebar.radio("Choose Data Source:", ["Generate Random Data", "Upload CSV"])

if data_option == "Generate Random Data":
    n_samples = st.sidebar.slider("Number of Samples:", 100, 10000, 1000)
    n_features = st.sidebar.slider("Number of Features:", 2, 10, 3)
    data = np.random.randn(n_samples, n_features)  # Normally distributed data
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file:", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file).values
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

# Number of Principal Components
k = st.sidebar.slider("Number of Principal Components (k):", 2, n_features)

# Apply PCA
X_pca, top_eigenvalues = pca(data, k)
if X_pca is not None:
    # Visualization of PCA Result using Plotly
    st.subheader("PCA Result Visualization")

    if k == 2:
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], labels={'x': 'PC1', 'y': 'PC2'},
                         title='2D Projection of Data onto First Two Principal Components')
    elif k == 3:
        fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                            labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                            title='3D Projection of Data onto First Three Principal Components')
        # Increase size of the 3D plot
        fig.update_layout(
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            width=1200,  # Set width of the plot
            height=800,  # Set height of the plot
            title='3D Projection of Data onto First Three Principal Components'
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1, k+1), y=top_eigenvalues,
                                 mode='lines+markers', line=dict(dash='dash'),
                                 marker=dict(size=10, color='blue')))
        fig.update_layout(title='Screen Plot of Eigenvalues',
                          xaxis_title='Principal Components',
                          yaxis_title='Eigenvalues',
                          width=1000,  # Set width of the plot
                          height=600)  # Set height of the plot

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Adjust the parameters and try again.")
