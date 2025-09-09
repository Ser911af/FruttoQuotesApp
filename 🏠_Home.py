import streamlit as st
from auth import get_authenticator, require_login, logout_button, current_role

st.set_page_config(page_title="AppFruttoQuotations", layout="wide")

# --- Auth ---
authenticator = get_authenticator()
name, auth_status, username = require_login(authenticator, location="main")

if not auth_status:
    st.stop()

logout_button(authenticator, location="sidebar")
role = current_role(username)

# --- UI ---
st.title("AppFruttoQuotations")
st.caption(f"Bienvenido, {name} — Rol: {role}")

st.markdown(
    """
    ### ¿Qué quieres hacer hoy?
    - **Explorer**: Explorar cotizaciones, filtrar y ver métricas rápidas.
    - **Daily Sheet**: Cargar/editar la hoja diaria.
    - **Upload Quotes**: Subir cotizaciones desde archivos.
    """
)

with st.sidebar:
    st.markdown("### Sesión")
    st.write(f"Usuario: **{username}**")
    st.write(f"Rol: **{role}**")
