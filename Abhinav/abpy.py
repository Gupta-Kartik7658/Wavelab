import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit setup
st.title("Potential Distribution Visualization")

# Parameters
V_nought = 5
a = 10
b = 10

# Function to calculate potential for the first figure
def calculate_potential_figure1():
    V = np.zeros((51, 51))
    for x in range(51):
        for y in range(51):
            for n in range(1, 101, 2):
                V[x, y] += (1 / n) * np.exp(-(n * np.pi * (x - 1)) / a) * np.sin((n * np.pi * (y - 1)) / a)
            V[x, y] = 4 * V_nought / np.pi * V[x, y]
    return V

# Function to calculate potential for the second figure
def calculate_potential_figure2():
    V = np.zeros((21, 21))
    for x in range(21):
        for y in range(21):
            for n in range(1, 101, 2):
                V[x, y] += (1 / n) * np.cosh((n * np.pi * (x - 10)) / a) * np.sin((n * np.pi * y) / a) * (1 / np.cosh(n * np.pi * (b / a)))
            V[x, y] = 4 * V_nought / np.pi * V[x, y]
    return V

# Plotting first figure with mesh plot
V1 = calculate_potential_figure1()
x1, y1 = np.meshgrid(np.arange(0, 51), np.arange(0, 51))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_wireframe(x1, y1, V1, color='blue')
ax1.set_xlim([0, 10])  # Corrected x-axis
ax1.set_ylim([0, 10])  # Corrected y-axis
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_zlabel('Potential')

st.pyplot(fig1)

# Plotting second figure
V2 = calculate_potential_figure2()
x2, y2 = np.meshgrid(np.linspace(0, 20, 21), np.linspace(0, 20, 21))

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(x2, y2, V2, cmap='winter')
ax2.set_xlim([0, 20])
ax2.set_ylim([0, 20])
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.set_zlabel('Potential')

st.pyplot(fig2)
