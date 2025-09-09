import streamlit as st
import streamlit_authenticator as stauth

def get_authenticator():
    cookie_name = st.secrets["auth"]["cookie_name"]
    signature_key = st.secrets["auth"]["signature_key"]
    cookie_expiry_days = int(st.secrets["auth"]["cookie_expiry_days"])

    credentials = {"usernames": {}}
    for u in st.secrets["users"]["list"]:
        credentials["usernames"][u["username"]] = {
            "name": u["name"],
            "email": u.get("email", ""),
            "password": u["password"],   # hash bcrypt
            "role": u.get("role", "viewer"),
        }

    return stauth.Authenticate(
        credentials,
        cookie_name=cookie_name,
        key=signature_key,
        cookie_expiry_days=cookie_expiry_days,
    )

# auth.py  (reemplaza solo esta función)
import streamlit as st

def require_login(authenticator, location="main"):
    """
    location: 'main' | 'sidebar' | 'unrendered'
    Retorna (name, auth_status, username).
    Evita 'duplicate form key' usando claves únicas.
    """
    form_key = f"login_{location}"            # <- clave única
    form_name_fallback = f"Login ({location})"  # <- para API vieja

    # 1) API nueva (0.4.x): location primero + fields + key=...
    try:
        name, auth_status, username = authenticator.login(
            location,
            fields={
                "Form name": "Login",
                "Username": "Usuario",
                "Password": "Contraseña",
                "Login": "Entrar",
            },
            key=form_key,   # <- clave única
        )
    except TypeError:
        # 2) API intermedia (0.3.x reciente): login(location, key=...)
        try:
            name, auth_status, username = authenticator.login(location, key=form_key)
        except TypeError:
            # 3) API antigua (0.3.x): login(form_name, location)
            name, auth_status, username = authenticator.login(form_name_fallback, location)

    if auth_status is False:
        st.error("Usuario/contraseña incorrectos.")
    elif auth_status is None:
        st.info("Ingresa tus credenciales.")
    return name, auth_status, username

def current_role(username: str) -> str:
    for u in st.secrets["users"]["list"]:
        if u["username"] == username:
            return u.get("role", "viewer")
    return "viewer"

def logout_button(authenticator, location="sidebar"):
    """
    Botón de logout con compatibilidad de firmas.
    """
    try:
        # API nueva: location como primer arg, con button label opcional
        if location == "sidebar":
            with st.sidebar:
                authenticator.logout("Cerrar sesión", location=location)
        else:
            authenticator.logout("Cerrar sesión", location=location)
    except TypeError:
        # API antigua: (button_name, location) o (location) según versión
        try:
            if location == "sidebar":
                with st.sidebar:
                    authenticator.logout("Cerrar sesión", location)
            else:
                authenticator.logout("Cerrar sesión", location)
        except TypeError:
            if location == "sidebar":
                with st.sidebar:
                    authenticator.logout(location)
            else:
                authenticator.logout(location)
