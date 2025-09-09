# auth_simple.py
import time
import streamlit as st
import bcrypt

SESSION_KEYS = {
    "auth": "is_auth",
    "user": "auth_user",
    "name": "auth_name",
    "role": "auth_role",
    "ts":   "auth_ts",
}

def _find_user(username: str):
    for u in st.secrets["users"]["list"]:
        if u["username"] == username:
            return u
    return None

def verify_credentials(username: str, password: str) -> tuple[bool, dict | None]:
    user = _find_user(username)
    if not user:
        return False, None
    ok = bcrypt.checkpw(password.encode(), user["password_hash"].encode())
    return ok, user if ok else (False, None)

def do_login_ui(location: str = "main"):
    # Renderiza el formulario de login (solo en Home)
    container = st.sidebar if location == "sidebar" else st
    with container.form(key=f"login_form_{location}", clear_on_submit=False):
        st.subheader("Login")
        username = st.text_input("Usuario", value="", key=f"u_{location}")
        password = st.text_input("Contraseña", type="password", value="", key=f"p_{location}")
        submit   = st.form_submit_button("Entrar")

    if submit:
        ok, user = verify_credentials(username.strip(), password)
        if ok:
            st.session_state[SESSION_KEYS["auth"]] = True
            st.session_state[SESSION_KEYS["user"]] = user["username"]
            st.session_state[SESSION_KEYS["name"]] = user.get("name", user["username"])
            st.session_state[SESSION_KEYS["role"]] = user.get("role", "viewer")
            st.session_state[SESSION_KEYS["ts"]]   = int(time.time())
            st.success(f"Bienvenido, {st.session_state[SESSION_KEYS['name']]}")
            st.rerun()  # refresca la página ya autenticada
        else:
            st.error("Usuario o contraseña incorrectos.")

def ensure_auth() -> bool:
    return bool(st.session_state.get(SESSION_KEYS["auth"], False))

def current_user() -> tuple[str | None, str | None, str | None]:
    return (
        st.session_state.get(SESSION_KEYS["user"]),
        st.session_state.get(SESSION_KEYS["name"]),
        st.session_state.get(SESSION_KEYS["role"]),
    )

def logout_button(location: str = "sidebar"):
    container = st.sidebar if location == "sidebar" else st
    if container.button("Cerrar sesión", key=f"logout_{location}"):
        for k in SESSION_KEYS.values():
            st.session_state.pop(k, None)
        st.success("Sesión cerrada.")
        st.rerun()
