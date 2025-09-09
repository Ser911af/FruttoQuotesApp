import streamlit as st
import streamlit_authenticator as stauth

def get_authenticator():
    # Lee configuración desde secrets
    cookie_name = st.secrets["auth"]["cookie_name"]
    signature_key = st.secrets["auth"]["signature_key"]
    cookie_expiry_days = int(st.secrets["auth"]["cookie_expiry_days"])

    # Construye el diccionario para Streamlit-Authenticator
    credentials = {"usernames": {}}
    for u in st.secrets["users"]["list"]:
        credentials["usernames"][u["username"]] = {
            "name": u["name"],
            "email": u.get("email", ""),
            "password": u["password"],
            "role": u.get("role", "viewer"),
        }

    authenticator = stauth.Authenticate(
        credentials,
        cookie_name=cookie_name,
        key=signature_key,
        cookie_expiry_days=cookie_expiry_days,
    )
    return authenticator

def require_login(authenticator, location="main"):
    """
    location: 'main' para forzar login en área principal; 'sidebar' para poner el formulario en el sidebar.
    Retorna (name, auth_status, username).
    """
    if location == "sidebar":
        name, auth_status, username = authenticator.login("Login", "sidebar")
    else:
        name, auth_status, username = authenticator.login("Login", "main")

    if auth_status is False:
        st.error("Usuario/contraseña incorrectos.")
    elif auth_status is None:
        st.info("Ingresa tus credenciales.")

    return name, auth_status, username

def current_role(username: str) -> str:
    # Devuelve el rol del usuario actual
    for u in st.secrets["users"]["list"]:
        if u["username"] == username:
            return u.get("role", "viewer")
    return "viewer"

def logout_button(authenticator, location="sidebar"):
    if location == "sidebar":
        with st.sidebar:
            authenticator.logout("Cerrar sesión")
    else:
        authenticator.logout("Cerrar sesión")
