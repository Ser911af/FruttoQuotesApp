import streamlit as st
import streamlit_authenticator as stauth

def get_authenticator():
    # Lee configuración desde secrets.toml
    cookie_name = st.secrets["auth"]["cookie_name"]
    signature_key = st.secrets["auth"]["signature_key"]
    cookie_expiry_days = int(st.secrets["auth"]["cookie_expiry_days"])

    # Construir dict de credenciales
    credentials = {"usernames": {}}
    for u in st.secrets["users"]["list"]:
        credentials["usernames"][u["username"]] = {
            "name": u["name"],
            "email": u.get("email", ""),
            "password": u["password"],   # hash bcrypt
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
    Muestra formulario de login.
    location: 'main', 'sidebar' o 'unrendered'
    Retorna (name, auth_status, username).
    """
    # ✅ Nuevo API: primero location
    name, auth_status, username = authenticator.login(
        location,
        fields={
            "Form name": "Login",
            "Username": "Usuario",
            "Password": "Contraseña",
            "Login": "Entrar",
        },
    )

    if auth_status is False:
        st.error("Usuario/contraseña incorrectos.")
    elif auth_status is None:
        st.info("Ingresa tus credenciales.")

    return name, auth_status, username


def current_role(username: str) -> str:
    """Devuelve el rol asociado al usuario."""
    for u in st.secrets["users"]["list"]:
        if u["username"] == username:
            return u.get("role", "viewer")
    return "viewer"


def logout_button(authenticator, location="sidebar"):
    """Botón de logout."""
    if location == "sidebar":
        with st.sidebar:
            authenticator.logout("Cerrar sesión")
    else:
        authenticator.logout("Cerrar sesión")
