# 🏠_Home.py
import streamlit as st
from typing import Dict

st.set_page_config(page_title="FruttoQuotes • Home", page_icon="🏠", layout="wide")
st.title("🏠 FruttoQuotes")
st.caption("Prototipo con login ultra-simple")

# ----------------------------
# Helpers
# ----------------------------
def get_credentials() -> Dict[str, str]:
    # Soporta ausencia de st.secrets en local
    creds = {}
    try:
        creds = dict(st.secrets.get("credentials", {}))
    except Exception:
        pass
    return creds

def do_login():
    with st.form("login_form", clear_on_submit=False):
        usuario = st.text_input("Usuario", autocomplete="username")
        clave = st.text_input("Contraseña", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Iniciar sesión", use_container_width=True)
    if submitted:
        credentials = get_credentials()
        if usuario in credentials and credentials[usuario] == clave:
            st.session_state.user = usuario
            st.success(f"Bienvenido {usuario} 👋")
            st.rerun()
        else:
            st.error("Credenciales inválidas")

def safe_page_link(path: str, label: str):
    # Renderiza link a página; si no existe, muestra aviso
    try:
        st.page_link(path, label=label)
    except Exception as e:
        with st.expander(f"⚠ No se pudo enlazar: {label}", expanded=False):
            st.write(f"Path: `{path}`")
            st.write("Motivo probable: el archivo no existe, nombre diferente o fuera de la carpeta `pages/`.")
            st.write("Error:")
            st.exception(e)

# ----------------------------
# Estado de sesión
# ----------------------------
if "user" not in st.session_state:
    st.session_state.user = None

# ----------------------------
# Auth
# ----------------------------
if st.session_state.user:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"Hola {st.session_state.user} 👋")
    with col2:
        if st.button("Cerrar sesión", use_container_width=True):
            st.session_state.user = None
            st.rerun()
else:
    do_login()

st.divider()
st.subheader("Páginas")

# ----------------------------
# Navegación (actualizada)
# ----------------------------
# IMPORTANTE: mantener los nombres EXACTOS como en /pages
links = [
    ("pages/0_Revenue.py",                 "💵 Revenue"),
    ("pages/1_Daily_Sheet.py",             "📊 Daily Sheet"),
    ("pages/2_Upload_Quotes.py",           "📤 Upload Quotes"),
    ("pages/3_Customer_Retention.py",      "🧲 Customer Retention"),
    ("pages/4_Vendor_Retention.py",        "🔁 Vendor Retention"),
    ("pages/5_Prod. Coverage.py",          "📦 Prod. Coverage"),  # Considera renombrar a 5_Prod_Coverage.py
    ("pages/6_Metrics.py",                 "📈 Metrics"),
    ("pages/7_VendorProduct_Customers.py", "🔗 VendorProduct Customers"),
]

# Render en 2 columnas para estética
left, right = st.columns(2)
for i, (path, label) in enumerate(links):
    with (left if i % 2 == 0 else right):
        safe_page_link(path, label)

# ----------------------------
# Tips de robustez (opcional)
# ----------------------------
with st.expander("💡 Recomendaciones para evitar errores de navegación"):
    st.markdown(
        """
- Ejecuta la app desde la **raíz del proyecto** donde está `🏠_Home.py`:  
  `streamlit run "🏠_Home.py"`  (o `Home.py` si decides quitar el emoji).
- Asegúrate que los archivos estén dentro de la carpeta **`pages/`** y que los nombres coincidan al 100% (mayúsculas, espacios, tildes).
- Evita caracteres especiales en nombres de archivos; en especial en **Streamlit Cloud**.
- Si renombras `5_Prod. Coverage.py` a `5_Prod_Coverage.py`, actualiza el enlace aquí también.
        """
    )
