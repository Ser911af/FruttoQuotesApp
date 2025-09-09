# auth_helpers.py
import streamlit as st

# Asegura que el paquete esté instalado
try:
    import streamlit_authenticator as stauth
except Exception:
    st.error("Falta instalar `streamlit-authenticator` (pip install streamlit-authenticator)")
    st.stop()

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

def _do_login(authenticator):
    """
    Soporta ambas APIs de streamlit-authenticator:
      - Vieja: login("Título", "sidebar") -> (name, auth_status, username)
      - Nueva: login(location="sidebar", fields=..., max_login_attempts=...)
               -> dict o tuple dependiendo de versión
    """
    # 1) Intento API nueva (keyword-only)
    try:
        login_res = authenticator.login(
            location="sidebar",
            max_login_attempts=3,
            fields={
                "Form name": "Iniciar sesión",
                "Username": "Usuario",
                "Password": "Contraseña",
            },
        )
        # Algunas versiones devuelven tuple, otras dict
        if isinstance(login_res, tuple) and len(login_res) == 3:
            name, auth_status, username = login_res
        elif isinstance(login_res, dict):
            name = login_res.get("name")
            auth_status = login_res.get("authentication_status")
            username = login_res.get("username")
        else:
            raise ValueError("Formato de retorno inesperado en login() (API nueva).")
        return name, auth_status, username
    except TypeError:
        # Si falla por firma incompatible, probamos API vieja
        pass
    except ValueError as e:
        # Si la validación de 'location' molesta, probamos fallback a 'main'
        if "Location must be" in str(e):
            try:
                login_res = authenticator.login(
                    location="main",
                    max_login_attempts=3,
                    fields={
                        "Form name": "Iniciar sesión",
                        "Username": "Usuario",
                        "Password": "Contraseña",
                    },
                )
                if isinstance(login_res, tuple) and len(login_res) == 3:
                    return login_res
                elif isinstance(login_res, dict):
                    return (
                        login_res.get("name"),
                        login_res.get("authentication_status"),
                        login_res.get("username"),
                    )
            except Exception:
                pass
        # Si es otro error, seguimos al fallback API vieja
        pass

    # 2) Fallback API vieja (positional args)
    #    login("Iniciar sesión", "sidebar") -> (name, auth_status, username)
    try:
        return authenticator.login("Iniciar sesión", "sidebar")
    except ValueError as e:
        # Si también se queja del location, probamos en "main"
        if "Location must be" in str(e):
            return authenticator.login("Iniciar sesión", "main")
        raise

def login_and_require(allowed_roles={"buyer", "admin"}):
    """
    Renderiza login y exige que el usuario esté autenticado con un rol permitido.
    Devuelve: dict con {name, username, role}.
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

    name, auth_status, username = _do_login(authenticator)

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

    return {"name": name, "username": username, "role": role}
