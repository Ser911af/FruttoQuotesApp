# pages/0_Explorer.py
import streamlit as st
from simple_auth import ensure_login, current_user, logout_button

st.set_page_config(page_title="Explorer", page_icon="🔎", layout="wide")
st.title("🔎 Explorer")

user = ensure_login()  # corta la ejecución si no hay login

st.success(f"Bienvenido, {user}")
st.write("Contenido de prueba de Explorer…")

# Botón de logout opcional aquí también
logout_button()
