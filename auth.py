import streamlit as st
import streamlit_authenticator as stauth


def get_authenticator():
    """Construye el objeto authenticator desde secrets.toml"""
    cookie_name = st.secrets["auth"]["cookie_name"]
    signature_key = st.secrets["auth"]["signature_key"]
    cookie_expiry_days = int(st.secrets["auth"]["cookie_expiry_days"])

    # Construir diccionario de credenciales
    credentials = {"usernames": {}}
    for u in st.secrets["users"]["list"]:
        credentials["usernames"][u["username"]] = {
            "name": u["name"],
            "email": u.get("email", ""),
            "password": u["password"],  # hash bcrypt
            "role": u.get("role", "viewer"),
        }

    return stauth.Authenticate(
        credentials,
        cookie_name=cookie_name,
        key=signature_key,
        cookie_expiry_days=cookie_expiry_days,
    )


def require_login(authenticator, location="main"):
    """
    Muestra formulario de login.
    location: 'main' | 'sidebar' | 'unrendered'
    Retorna (name, auth_status, username).
    Se adapta a distintas versiones de streamlit-authenticator.
    """
    form_key = f"login_{location}"  # <- clave única para evitar colisiones
    form_name_fallback = f"Login ({location})"

    # 1) API nueva (>=0.4.x)
    try:
        name, auth_status, username = authenticator.login(
            location,
            fields={
                "Form name": "Login",
                "Username": "Usuario",
                "Password": "Contraseña",
                "Login": "Entrar",
            },
            key=form_key,
        )
    except TypeError:
        # 2) API intermedia (~0.3.x)
        try:
            name, auth_status, username = authenticator.login(location, key=form_key)
        except TypeError:
            # 3) API antigua (<=0.3.1)
            name, auth_status, username = authenticator.login(form_name_fallback, location)

    if auth_status is False:
        st.error("Usuario/contraseña incorrectos.")
    elif auth_status is None:
        st.info("Ingresa tus credenciales.")
    return name, auth_status, username


def current_role(username: str) -> str:
    """Devuelve el rol asociado al usuario actual."""
    for u in st.secrets["users"]["list"]:
        if u["username"] == username:
            return u.get("role", "viewer")
    return "viewer"


def logout_button(authenticator, location="sidebar"):
    """
    Renderiza botón de logout, compatible con distintas versiones.
    """
    btn_label = "Cerrar sesión"
    try:
        # API nueva: requiere location como keyword
        if location == "sidebar":
            with st.sidebar:
                authenticator.logout(btn_label, location=location, key=f"logout_{location}")
        else:
            authenticator.logout(btn_label, location=location, key=f"logout_{location}")
    except TypeError:
        # API antigua: (button_name, location)
        if location == "sidebar":
            with st.sidebar:
                authenticator.logout(btn_label, location)
        else:
            authenticator.logout(btn_label, location)
