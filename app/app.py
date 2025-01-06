import streamlit as st

st.set_page_config(layout="wide",
    page_title="LLM app",
    page_icon="👋",
)

st.write("# Welcome to the LLM application! 👋")

st.sidebar.success("Hello")

st.markdown(
    """
    A local implementation of large language models for genetics.
"""
)
