import streamlit as st
import streamlit_authenticator as stauth
from typing import Dict, Any, Set

def _secrets_to_credentials() -> Dict[str, Any]:
    users = st.secrets["users"]["list"]
    usernames: Dict[str, Any] = {}
    for u in users:
        usernames[u["username"]] = {
            "name": u["name"],
            "password": u["password"],  # BCRYPT
            "email": u.get("email", ""),
            "role": u.get("role", ""),
        }
    return {"usernames": usernames}

def get_authenticator() -> stauth.Authenticate:
    creds = _secrets_to_credentials()
    cookie_name = st.secrets["auth"]["cookie_name"]
    signature_key = st.secrets["auth"]["signature_key"]
    cookie_expiry_days = int(st.secrets["auth"]["cookie_expiry_days"])
    return stauth.Authenticate(
        credentials=creds,
        cookie_name=cookie_name,
        key=signature_key,
        cookie_expiry_days=cookie_expiry_days,
        preauthorized=[]
    )

def login_and_require(allowed_roles: Set[str] | None = None):
    authenticator = get_authenticator()

    # 👇 Firma compatible con versiones previas (sin `fields=`)
    name, auth_status, username = authenticator.login(
        "Iniciar sesión",   # form_name
        location="main"     # 'main' | 'sidebar'
    )

    if auth_status is False:
        st.error("Usuario o contraseña incorrectos.")
        st.stop()
    if auth_status is None:
        st.info("Ingresa tus credenciales para continuar.")
        st.stop()

    # Autenticado:
    creds = _secrets_to_credentials()
    user_info = creds["usernames"][username]
    role = user_info.get("role", "")

    if allowed_roles:
        allowed_norm = {r.strip().lower() for r in allowed_roles}
        if role.strip().lower() not in allowed_norm:
            st.error(f"No tienes permisos para esta página. Rol requerido: {', '.join(allowed_roles)}")
            st.stop()

    # Logout en sidebar
    with st.sidebar:
        st.write(f"👤 {user_info.get('name')} ({username})")
        st.caption(user_info.get("email", ""))
        authenticator.logout("Cerrar sesión", location="sidebar")

    return {
        "name": user_info.get("name"),
        "username": username,
        "email": user_info.get("email"),
        "role": role,
    }
