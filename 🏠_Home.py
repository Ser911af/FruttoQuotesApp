import streamlit as st
from auth_simple import ensure_auth, do_login_ui, current_user, logout_button
import os, sys
from pathlib import Path
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd().resolve()

st.write("🔎 Debug paths",
         {"__file__": str(__file__),
          "script_dir": str(HERE),
          "cwd": str(ROOT),
          "files_in_script_dir": [p.name for p in HERE.iterdir() if p.is_file()],
          "files_in_cwd": [p.name for p in ROOT.iterdir() if p.is_file()],
          "sys.path": sys.path[:5]})  # muestra primeros 5

st.stop()  # ← temporalmente para ver el debug

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
