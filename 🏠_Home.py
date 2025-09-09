import streamlit as st
from auth_helpers import login_and_require

st.set_page_config(page_title="FruttoQuotes • Home", page_icon="🏠", layout="wide")

st.title("🏠 FruttoQuotes")
st.caption("Prototipo con login simple (usuarios en secrets)")

# Requiere estar logueado (cualquier rol)
user = login_and_require()

st.success(f"Hola, {user['name']} · Rol: {user['role']}")

st.subheader("Navegación rápida")
st.page_link("pages/0_Explorer.py", label="🔎 Explorer")
st.page_link("pages/1_Daily_Sheet.py", label="📊 Daily Sheet")
st.page_link("pages/2_Upload_Quotes.py", label="📤 Upload Quotes")
