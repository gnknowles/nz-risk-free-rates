
## Landing/welcome page for the app

import streamlit as st

welcome_css = 'colour: white; font-size: 28px; text-align: centre; border-radius: 1rem; padding: 1rem 1rem 1rem 1rem; font-weighht: bold; background-colour: #D04A02;'

# Welcome text - using a st.html to inject custom CSS
st.html(f"<p style='{welcome_css}'> New Zealand Risk-free Discount Rate Tool </p>")

st.container(heigh=10, border=False)

st.subheader("Introduction")
st.markdown("""
            Add various text here...
            """)

st.container(heigh=10, border=False)

st.subheader("User Guide")
st.markdown("""
            Add various text here...
            """)

st.container(heigh=10, border=False)

if st.button("Go to tool", type = "primary"):
    st.switch_page("app-pages/page1.py")