import streamlit as st
from auth_simple import ensure_auth, current_user, logout_button

st.set_page_config(page_title="Explorer", layout="wide")

# Guard de sesión (sin login aquí)
if not ensure_auth():
    st.error("No has iniciado sesión. Ve a la página Home para ingresar.")
    st.stop()

# (Opcional) botón de logout también aquí
logout_button(location="sidebar")

username, name, role = current_user()

# ... resto de tu lógica (filtros, tablas, gráficos) ...

