# 🏠_Home.py
import streamlit as st

st.set_page_config(page_title="FruttoQuotes • Home", page_icon="🏠", layout="wide")
st.title("🏠 FruttoQuotes")
st.caption("Prototipo con login ultra-simple")

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user:
    st.success(f"Hola {st.session_state.user} 👋")
    if st.button("Cerrar sesión"):
        st.session_state.user = None
        st.rerun()
else:
    usuario = st.text_input("Usuario")
    clave = st.text_input("Contraseña", type="password")
    if st.button("Iniciar sesión"):
        if usuario in st.secrets["credentials"] and st.secrets["credentials"][usuario] == clave:
            st.session_state.user = usuario
            st.rerun()
        else:
            st.error("Credenciales inválidas")

st.divider()
st.subheader("Páginas")
st.page_link("pages/0_Revenue.py", label="🔎 Explorer")
st.page_link("pages/1_Daily_Sheet.py", label="📊 Daily Sheet")
st.page_link("pages/2_Upload_Quotes.py", label="📤 Upload Quotes")
