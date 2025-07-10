import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def create_plot():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Sine Wave")

    st.pyplot(fig)
