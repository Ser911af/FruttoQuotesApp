# simple_auth.py
import streamlit as st

def ensure_login() -> str:
    """Detiene la página si no hay usuario logueado. Devuelve el username."""
    user = st.session_state.get("user")
    if not user:
        st.warning("Debes iniciar sesión en Home antes de acceder a esta página.")
        st.page_link("🏠_Home.py", label="Home")
        st.stop()
    return user

def current_user() -> str | None:
    """Devuelve el username actual o None."""
    return st.session_state.get("user")

def logout_button(label: str = "Cerrar sesión"):
    """Botón para cerrar sesión (borra usuario y recarga)."""
    if st.button(label):
        st.session_state.user = None
        st.rerun()
