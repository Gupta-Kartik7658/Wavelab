import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import imageio.v3 as im
import time
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Colour Space Conversion", initial_sidebar_state="collapsed", 
                   page_icon="Kartik/logo (1).png", layout="wide")

# Load CSS for styling
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("Kartik/style.css")
load_css(css_path)

# Title container
st.markdown('''
<div class="title-container">
    <h2>Colour Space Conversion</h2>
</div>
''', unsafe_allow_html=True)

# Center-align the submit button using CSS for Streamlit's button

# Function to display image distributions
def color_space_conversion(img):
    image = im.imread(img)
    
    # Extract color channels
    # Splitting into respective RGB color component matrices
    R = image[:, :, 0].astype(np.float64)
    G = image[:, :, 1].astype(np.float64)
    B = image[:, :, 2].astype(np.float64)

    # Transformation Matrix (ITU-R BT.601 for JPEG conversion)
    TM = np.array([
        [0.256788235294118, 0.504129411764706, 0.0979058823529412],
        [-0.148223529411765, -0.290992156862745, 0.43921568627451],
        [0.43921568627451, -0.367788235294118, -0.0714274509803921]
    ])

    # Converting to respective Luma and Chroma components
    Y1 = TM[0, 0] * R + TM[0, 1] * G + TM[0, 2] * B + 16
    Cb1 = TM[1, 0] * R + TM[1, 1] * G + TM[1, 2] * B + 128
    Cr1 = TM[2, 0] * R + TM[2, 1] * G + TM[2, 2] * B + 128
    
    ITM = np.linalg.inv(TM)

    # Initialize z (null chroma) and Ysub (constant luminance)
    z = 127 * np.ones((image.shape[0], image.shape[1]))
    Ysub = 127 * np.ones((image.shape[0], image.shape[1]))

    # Chrominance Blue (Cb)
    Cbr = ITM[0, 0] * (Ysub - 16) + ITM[0, 1] * (Cb1 - 128) + ITM[0, 2] * (z - 128)
    Cbg = ITM[1, 0] * (Ysub - 16) + ITM[1, 1] * (Cb1 - 128) + ITM[1, 2] * (z - 128)
    Cbb = ITM[2, 0] * (Ysub - 16) + ITM[2, 1] * (Cb1 - 128) + ITM[2, 2] * (z - 128)

    # Chrominance Red (Cr)
    Crr = ITM[0, 0] * (Ysub - 16) + ITM[0, 1] * (z - 128) + ITM[0, 2] * (Cr1 - 128)
    Crg = ITM[1, 0] * (Ysub - 16) + ITM[1, 1] * (z - 128) + ITM[1, 2] * (Cr1 - 128)
    Crb = ITM[2, 0] * (Ysub - 16) + ITM[2, 1] * (z - 128) + ITM[2, 2] * (Cr1 - 128)

    # Concatenate to image matrix
    just_Cb2 = np.stack((Cbr, Cbg, Cbb), axis=-1).astype(np.uint8)
    just_Cr2 = np.stack((Crr, Crg, Crb), axis=-1).astype(np.uint8)

    
    # Display original image
    colm = st.columns(3)
    with colm[1]:
        st.subheader("Original Image")
        st.image(Image.fromarray(image))
    st.markdown("<br>",unsafe_allow_html=True)
    col = st.columns(3)
    # Display Luminance (Y)
    with col[0]:
        st.subheader("Luminance (Y)")
        st.image(Image.fromarray(Y1.astype(np.uint8)))

    # Display Chrominance Blue (Cb)
    with col[1]:
        st.subheader("Chrominance Blue (Cb)")
        st.image(Image.fromarray(just_Cb2))

    # Display Chrominance Red (Cr)
    with col[2]:
        st.subheader("Chrominance Red (Cr)")
        st.image(Image.fromarray(just_Cr2))
    
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
    color_space_conversion(img_uploader)

st.markdown("<br><br>",unsafe_allow_html=True)

st.markdown('''
# YCbCr Color Space Conversion

The YCbCr color space is a commonly used color model in video and image processing, particularly in compression formats such as JPEG and MPEG. It separates image intensity (luminance, Y) from color information (chrominance, Cb and Cr). This separation enables efficient compression by reducing the resolution of the chrominance components without significantly affecting perceived image quality.

### Transformation Matrix (RGB to YCbCr)

The conversion from RGB to YCbCr in the ITU-R BT.601 standard is defined by the following matrix operation:
$$
\\begin{bmatrix}
Y \\\\ 
Cb \\\\ 
Cr
\\end{bmatrix}
=
\\begin{bmatrix}
0.2568 & 0.5041 & 0.0979 \\\\ 
-0.1482 & -0.2910 & 0.4392 \\\\ 
0.4392 & -0.3678 & -0.0714
\\end{bmatrix}
\\cdot
\\begin{bmatrix}
R \\\\ 
G \\\\ 
B
\\end{bmatrix}
+
\\begin{bmatrix}
16 \\\\ 
128 \\\\ 
128
\\end{bmatrix}
$$

- **Y (Luminance):** Represents brightness and is derived as a weighted sum of the red, green, and blue components.
- **Cb (Chrominance Blue):** Represents the blue-difference chroma information.
- **Cr (Chrominance Red):** Represents the red-difference chroma information.

### Inverse Transformation Matrix (YCbCr to RGB)

To convert from YCbCr back to RGB, the inverse transformation matrix is applied:

$$
\\begin{bmatrix}
R \\\\ 
G \\\\ 
B
\\end{bmatrix}
=
\\begin{bmatrix}
1 & 0 & 1.402 \\\\ 
1 & -0.3441 & -0.7141 \\\\ 
1 & 1.772 & 0
\\end{bmatrix}
\\cdot
\\begin{bmatrix}
Y - 16 \\\\ 
Cb - 128 \\\\ 
Cr - 128
\\end{bmatrix}
$$

### Applications of YCbCr
- Used in compression techniques like JPEG, where chrominance is subsampled to reduce storage while preserving visual quality.
- Employed in television broadcasting standards such as PAL, NTSC, and SECAM.
- Facilitates efficient processing in computer vision tasks by isolating luminance from chrominance.

Understanding YCbCr is essential in various multimedia and vision applications due to its ability to optimize image storage and processing while maintaining visual fidelity.
''',unsafe_allow_html=True)
