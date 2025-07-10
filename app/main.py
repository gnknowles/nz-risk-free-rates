import streamlit as st

# Set up pages
home_page = st.Page(
    page="app-pages/welcome.py",
    title="Welcome",
    icon="🏠",
    default=True
)

tool_page = st.Page(
    page="app-pages/page1.py",
    title="Tool",
    icon="💡",
    url_path="riskfree_run_tool"
)

pages = [home_page, tool_page]

pages_nav = {
    "Welcome": [
        home_page
    ],
    "Tool Pages": [
        tool_page
    ]
}

layouts = ["centered", "centered"]

# Setup Navigation
pg = st.navigation(pages=pages_nav, position="sidebar", expanded=True)

pg.run()
