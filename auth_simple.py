# auth_simple.py
import streamlit as st
import bcrypt

USERS_PATH = ("basic_auth", "users")  # ruta en secrets

def _load_users():
    """
    Devuelve un dict {username: {name, roles, password_hash}}
    Valida estructura y da errores claros en UI.
    """
    try:
        basic_auth = st.secrets.get(USERS_PATH[0], {})
        users = basic_auth.get(USERS_PATH[1], {})
        if not isinstance(users, dict) or not users:
            st.error("No hay usuarios definidos en [basic_auth.users] de secrets.toml.")
            return {}
        # Normaliza usernames a minúsculas
        norm = {}
        for uname, data in users.items():
            if not isinstance(data, dict):
                st.error(f"Entrada de usuario inválida para '{uname}'; debe ser objeto con password_hash.")
                continue
            ukey = (uname or "").strip().lower()
            if not ukey:
                st.error("Se encontró un usuario sin nombre (clave vacía) en [basic_auth.users].")
                continue
            if "password_hash" not in data:
                st.error(f"Usuario '{uname}' sin 'password_hash' en secrets.")
                continue
            norm[ukey] = {
                "name": data.get("name") or uname,
                "roles": data.get("roles", []),
                "password_hash": str(data["password_hash"]),
                "username": ukey,
            }
        return norm
    except Exception as e:
        st.error(f"Error leyendo secrets: {e}")
        return {}

def verify_credentials(username: str, password: str):
    """
    Retorna (ok: bool, user: dict|None)
    Maneja usuario no encontrado y hash ausente sin lanzar KeyError.
    """
    users = _load_users()
    if not users:
        return False, None

    uname = (username or "").strip().lower()
    if uname not in users:
        return False, None

    user = users[uname]
    ph = user.get("password_hash")
    if not ph:
        # Estructura inválida → mejor mensaje claro
        st.error(f"El usuario '{uname}' no tiene 'password_hash' en secrets.")
        return False, None

    try:
        ok = bcrypt.checkpw((password or "").encode(), ph.encode())
    except Exception as e:
        st.error(f"Error validando contraseña: {e}")
        return False, None

    return ok, user if ok else (False, None)

def do_login_ui(location: str = "main", key_prefix: str = "auth"):
    """
    Renderiza el formulario de login simple. Si éxito, guarda user en session_state.
    """
    container = st if location == "main" else st.sidebar

    # Si ya hay sesión activa, muestra perfil y botón logout
    if st.session_state.get(f"{key_prefix}_user"):
        u = st.session_state[f"{key_prefix}_user"]
        with container.expander("Sesión activa", expanded=False):
            container.write({
                "username": u.get("username"),
                "name": u.get("name"),
                "roles": u.get("roles", []),
            })
        if container.button("Cerrar sesión", key=f"{key_prefix}_logout"):
            st.session_state.pop(f"{key_prefix}_user", None)
            st.rerun()
        return

    with container.form(key=f"{key_prefix}_form", clear_on_submit=False):
        username = container.text_input("Usuario", key=f"{key_prefix}_uname")
        password = container.text_input("Contraseña", type="password", key=f"{key_prefix}_pwd")
        submitted = container.form_submit_button("Iniciar sesión")
        if submitted:
            ok, user = verify_credentials(username.strip(), password)
            if ok and user:
                st.session_state[f"{key_prefix}_user"] = user
                st.success(f"¡Bienvenido, {user.get('name', user['username'])}!")
                st.experimental_rerun()  # refresca la UI como logueado
            else:
                st.error("Usuario o contraseña incorrectos.")

def require_roles(roles: set[str], key_prefix: str = "auth") -> bool:
    """
    Retorna True si el usuario autenticado tiene alguno de los roles; False si no.
    Si no hay usuario, retorna False.
    """
    u = st.session_state.get(f"{key_prefix}_user")
    if not u:
        return False
    if not roles:
        return True
    user_roles = set(u.get("roles", []))
    return bool(user_roles.intersection(roles))
