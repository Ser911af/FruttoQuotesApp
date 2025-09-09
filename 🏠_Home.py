import streamlit as st

st.set_page_config(page_title="Login simple", page_icon="🔐")
st.title("Login simple")

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user:
    st.success(f"Hola {st.session_state.user}")
    if st.button("Cerrar sesión"):
        st.session_state.user = None
        st.rerun()
else:
    u = st.text_input("Usuario")
    p = st.text_input("Contraseña", type="password")
    if st.button("Iniciar sesión"):
        if u in st.secrets["credentials"] and st.secrets["credentials"][u] == p:
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Credenciales inválidas")
