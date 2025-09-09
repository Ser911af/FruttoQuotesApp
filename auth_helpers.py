import streamlit as st
import streamlit_authenticator as stauth
from typing import Dict, Any, Set

def _secrets_to_credentials() -> Dict[str, Any]:
    """
    Convierte tu esquema en secrets.toml a lo que necesita streamlit-authenticator.
    """
    users = st.secrets["users"]["list"]
    usernames: Dict[str, Any] = {}
    for u in users:
        usernames[u["username"]] = {
            "name": u["name"],
            "password": u["password"],  # debe ser BCRYPT hash
            "email": u.get("email", ""),
            "role": u.get("role", ""),
        }
    creds = {"usernames": usernames}
    return creds

def get_authenticator() -> stauth.Authenticate:
    creds = _secrets_to_credentials()
    cookie_name = st.secrets["auth"]["cookie_name"]
    signature_key = st.secrets["auth"]["signature_key"]
    cookie_expiry_days = int(st.secrets["auth"]["cookie_expiry_days"])
    # preauthorized no lo usamos; pasa una lista vacía
    authenticator = stauth.Authenticate(
        credentials=creds,
        cookie_name=cookie_name,
        key=signature_key,
        cookie_expiry_days=cookie_expiry_days,
        preauthorized=[]
    )
    return authenticator

def login_and_require(allowed_roles: Set[str] | None = None):
    """
    Dibuja el formulario de login (en el área principal) y corta la ejecución si no está autenticado
    o si su rol no está permitido.

    Uso típico en una página:
        user = login_and_require(allowed_roles={"Buyer", "Admin"})
    """
    authenticator = get_authenticator()

    # IMPORTANTE: usar location="main" para evitar errores de layout
    name, auth_status, username = authenticator.login(location="main", fields={
        "Form name": "Iniciar sesión",
        "Username": "Usuario",
        "Password": "Contraseña",
        "Login": "Entrar",
    })

    if auth_status is False:
        st.error("Usuario o contraseña incorrectos.")
        st.stop()
    if auth_status is None:
        # Aún no envía el form
        st.info("Ingresa tus credenciales para continuar.")
        st.stop()

    # Autenticado:
    creds = _secrets_to_credentials()
    user_info = creds["usernames"][username]
    role = user_info.get("role", "")

    if allowed_roles:
        # normaliza a mayúscula inicial
        allowed_norm = {r.strip().lower() for r in allowed_roles}
        if role.strip().lower() not in allowed_norm:
            st.error(f"No tienes permisos para esta página. Rol requerido: {', '.join(allowed_roles)}")
            st.stop()

    # Mostrar botón de logout en la barra lateral
    with st.sidebar:
        st.write(f"👤 {user_info.get('name')} ({username})")
        st.caption(user_info.get("email", ""))
        authenticator.logout(button_name="Cerrar sesión")

    # Devuelve un dict útil para usar en las páginas
    return {
        "name": user_info.get("name"),
        "username": username,
        "email": user_info.get("email"),
        "role": role,
    }
