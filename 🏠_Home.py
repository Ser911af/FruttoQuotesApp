import streamlit as st

st.set_page_config(page_title="Login simple", page_icon="🔐")

st.title("Login simple con secrets")

# --- login manual ---
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user:
    st.success(f"Hola {st.session_state.user} 👋")
    if st.button("Cerrar sesión"):
        st.session_state.user = None
else:
    usuario = st.text_input("Usuario")
    clave = st.text_input("Contraseña", type="password")
    if st.button("Iniciar sesión"):
        if usuario in st.secrets["credentials"] and st.secrets["credentials"][usuario] == clave:
            st.session_state.user = usuario
            st.rerun()
        else:
            st.error("Credenciales inválidas")
