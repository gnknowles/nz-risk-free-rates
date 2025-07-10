import streamlit as st
from utils.custom_plot import create_plot

st.title("📄 Page 1 - Data Overview")

st.write("This page could show some data statistics or preprocessing steps.")

st.subheader("Sample Plot")
create_plot()