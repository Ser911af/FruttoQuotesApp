# auth_helpers.py
import streamlit as st
import streamlit_authenticator as stauth

def _load_credentials_from_secrets():
    auth_secrets = st.secrets.get("auth", {})
    users_secrets = st.secrets.get("users", {})
    users_list = users_secrets.get("list", [])

    credentials = {"usernames": {}}
    roles_by_user = {}

    for u in users_list:
        username = u["username"]
        credentials["usernames"][username] = {
            "name": u["name"],
            "email": u.get("email", ""),
            "password": u["password"],  # HASH
        }
        roles_by_user[username] = u.get("role", "viewer")

    return credentials, roles_by_user, auth_secrets

def login_and_require(allowed_roles={"buyer", "admin"}):
    """
    Renderiza login en el sidebar y frena la app si:
    - No se autenticó
    - No tiene rol permitido
    Devuelve: dict con info del usuario autenticado.
    """
    credentials, roles_by_user, auth_secrets = _load_credentials_from_secrets()

    cookie_name = auth_secrets.get("cookie_name", "app_auth")
    signature_key = auth_secrets.get("signature_key", "please_replace_me")
    cookie_expiry_days = int(auth_secrets.get("cookie_expiry_days", 7))

    authenticator = stauth.Authenticate(
        credentials,
        cookie_name,
        signature_key,
        cookie_expiry_days,
    )

    st.sidebar.title("Acceso")
    name, auth_status, username = authenticator.login("Iniciar sesión", "sidebar")

    if auth_status is False:
        st.error("Usuario o contraseña incorrectos.")
        st.stop()
    elif auth_status is None:
        st.info("Ingresa tus credenciales para continuar.")
        st.stop()

    role = roles_by_user.get(username, "viewer")
    if role not in allowed_roles:
        st.error(f"Acceso denegado. Tu rol es '{role}'. Se requiere uno de: {', '.join(allowed_roles)}.")
        st.stop()

    with st.sidebar.expander("Sesión"):
        st.write(f"👤 {name}  \n**Rol:** {role}")
        authenticator.logout("Cerrar sesión", "sidebar")

    return {
        "name": name,
        "username": username,
        "role": role,
    }
