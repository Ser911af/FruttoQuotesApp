import streamlit as st
import streamlit_authenticator as stauth

# ---------- helpers ----------

def _login_compat(authenticator, location: str):
    """
    Llama a authenticator.login() probando las firmas conocidas según la versión instalada.
    Devuelve (name, auth_status, username).
    """
    # 1) Nuevo API (v0.4+ aprox): first arg = location, labels en 'fields'
    try:
        return authenticator.login(
            location,
            fields={
                "Form name": "Login",
                "Username": "Usuario",
                "Password": "Contraseña",
                "Login": "Entrar",
            },
        )
    except TypeError:
        pass
    except ValueError:
        # algunas variantes también arrojan ValueError si el orden no es válido
        pass

    # 2) Variante con kwargs (observada en foros): key + location
    try:
        return authenticator.login(key="Login", location=location)
    except TypeError:
        pass
    except ValueError:
        pass

    # 3) API clásico (blog original): primero el nombre del formulario, luego la ubicación
    try:
        return authenticator.login("Login", location)
    except TypeError:
        pass
    except ValueError:
        pass

    # 4) Último intento (por si alguna versión espera (location, "Login"))
    try:
        return authenticator.login(location, "Login")
    except Exception as e:
        # Si nada funcionó, mostramos detalle
        st.error(f"No se pudo renderizar el login con la versión instalada de streamlit-authenticator. Detalle: {e}")
        return None, None, None


def _logout_compat(authenticator, label="Cerrar sesión", location="sidebar"):
    """
    Llama a authenticator.logout() con variantes comunes para compatibilidad.
    """
    try:
        # más común
        if location == "sidebar":
            with st.sidebar:
                authenticator.logout(label)
        else:
            authenticator.logout(label)
    except TypeError:
        # algunas versiones aceptan (button_name, location)
        try:
            authenticator.logout(label, location)
        except Exception:
            # fallback: botón manual
            target = st.sidebar if location == "sidebar" else st
            if target.button(label):
                st.session_state.clear()
                st.rerun()


# ---------- API pública de este módulo ----------

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
    Renderiza el formulario de login en 'main' o 'sidebar'.
    Retorna (name, auth_status, username).
    """
    name, auth_status, username = _login_compat(authenticator, location)

    if auth_status is False:
        st.error("Usuario/contraseña incorrectos.")
    elif auth_status is None:
        st.info("Ingresa tus credenciales.")

    return name, auth_status, username


def current_role(username: str) -> str:
    """Devuelve el rol asociado al usuario desde secrets.toml."""
    for u in st.secrets["users"]["list"]:
        if u["username"] == username:
            return u.get("role", "viewer")
    return "viewer"


def logout_button(authenticator, location="sidebar"):
    """Muestra botón de logout en main o sidebar (con compatibilidad)."""
    _logout_compat(authenticator, label="Cerrar sesión", location=location)
