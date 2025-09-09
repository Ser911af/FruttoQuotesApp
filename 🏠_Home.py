import streamlit as st
from auth import get_authenticator, require_login, logout_button, current_role

st.set_page_config(page_title="AppFruttoQuotations", layout="wide")

authenticator = get_authenticator()
name, auth_status, username = require_login(authenticator, location="main")

if auth_status:
    logout_button(authenticator, location="sidebar")
    st.success(f"Bienvenido, {name} 👋")
    st.write("Selecciona una página en la barra lateral.")
