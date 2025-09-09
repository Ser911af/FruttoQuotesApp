import streamlit as st
from auth_simple import ensure_auth, do_login_ui, current_user, logout_button

st.set_page_config(page_title="AppFruttoQuotations", layout="wide")

# Si no hay sesión, mostramos el login AQUÍ (único lugar)
if not ensure_auth():
    do_login_ui(location="main")
    st.stop()

# Ya autenticado
logout_button(location="sidebar")

username, name, role = current_user()
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
