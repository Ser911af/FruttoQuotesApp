import streamlit as st
from auth_helpers import require_login

st.set_page_config(page_title="FruttoQuotes • Home", page_icon="🏠", layout="wide")

st.title("🏠 FruttoQuotes")
st.caption("Inicio • Control de acceso con OIDC (st.login)")

# --- Encabezado de sesión (mostrar login/logout) ---
col1, col2 = st.columns([3, 1])
with col1:
    st.write("Bienvenido a FruttoQuotes. Usa el menú para ir a las secciones.")

with col2:
    if not st.user.is_logged_in:
        # Puedes cambiar el provider por "microsoft", "okta", "auth0", etc. si definiste [auth.<provider>]
        st.button("Iniciar sesión", on_click=lambda: st.login())  # usa proveedor por defecto en [auth]
    else:
        st.write(f"👤 {st.user.name}")
        st.button("Cerrar sesión", on_click=st.logout)

st.divider()

# --- Si no está logueado, muestro CTA y no enseño navegación ---
if not st.user.is_logged_in:
    st.info(
        "Debes iniciar sesión para acceder a las páginas. "
        "Haz clic en **Iniciar sesión** arriba (usa el proveedor OIDC que configuraste)."
    )
    st.stop()

# --- Contenido para usuarios autenticados ---
st.success(f"Autenticado como: {st.user.name}")
# Estos campos dependen del IdP; muchos devuelven email y sub:
st.write("Email:", getattr(st.user, "email", "—"))
st.write("ID (sub):", getattr(st.user, "sub", "—"))

st.subheader("Navegación rápida")
# Navegación multipágina (en versiones recientes puedes usar page_link)
st.page_link("pages/0_Explorer.py", label="🔎 Explorer")
st.page_link("pages/1_Daily_Sheet.py", label="📊 Daily Sheet")
st.page_link("pages/2_Upload_Quotes.py", label="📤 Upload Quotes")

st.caption("Tip: Usa la barra lateral de Streamlit para moverte entre páginas.")
