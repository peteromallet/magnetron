from magnetron import display_rows_with_editing
import streamlit as st

st.set_page_config(layout="wide") 

display_rows_with_editing(status='approved', allow_action=False)