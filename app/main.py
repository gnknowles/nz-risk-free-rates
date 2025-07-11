import streamlit as st
import os
import sys

def load_local_environment():
    """
    Load local environment variables from .env file
    """
    from dotenv import load_dotenv

    if os.path.exists(".env"):
        load_dotenv(".env", override=True)
    
    else:
        raise Exception("WARNING | No local .env file found - ensure you are running from app/ and have a .env file in app/")


# Set up pages
home_page = st.Page(
    page="app-pages/welcome.py",
    title="Welcome",
    icon="🏠",
    default=True
)

data_page = st.Page(
    page="app-pages/data_select.py",
    title="Select Data",
    icon="💡",
    url_path="data_select"
)

curves_page = st.Page(
    page="app-pages/create_curves.py",
    title="Create Yield Curves",
    icon="📅",
    url_path="create_curves"
)


pages = [home_page, data_page, curves_page]

pages_nav = {
    "Welcome": [
        home_page
    ],
    "Tool Pages": [
        data_page, curves_page
    ]
}

layouts = ["centered", "centered", "centered", "centered"]

# Setup Navigation
pg = st.navigation(pages=pages_nav, position="sidebar", expanded=True)


load_local_environment()

pg.run()
