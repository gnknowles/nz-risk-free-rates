import streamlit as st
from components.custom_plot import create_plot

st.set_page_config(page_title="My Streamlit App", layout="wide")

st.title("🏠 Home Page")
st.markdown("Welcome to the **Streamlit App** template.")

st.subheader("Sample Plot")
create_plot()
