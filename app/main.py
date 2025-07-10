import streamlit as st

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

tool_page = st.Page(
    page="app-pages/page1.py",
    title="Tool",
    icon="📅",
    url_path="riskfree_run_tool"
)

pages = [home_page, data_page, tool_page]

pages_nav = {
    "Welcome": [
        home_page
    ],
    "Tool Pages": [
        data_page, tool_page
    ]
}

layouts = ["centered", "centered", "centered"]

# Setup Navigation
pg = st.navigation(pages=pages_nav, position="sidebar", expanded=True)

pg.run()
