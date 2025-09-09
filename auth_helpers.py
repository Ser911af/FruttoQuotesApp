import streamlit as st

def require_login(provider: str | None = None):
    """
    Si el usuario NO está autenticado:
      - Muestra un botón para iniciar sesión con st.login(provider)
      - Corta la ejecución de la página con st.stop()
    Si está autenticado, continúa.
    """
    if not st.user.is_logged_in:
        st.warning("Esta página requiere autenticación.")
        if provider:  # Ej.: "microsoft", "okta", "auth0"
            st.button("Iniciar sesión", on_click=lambda: st.login(provider))
        else:
            st.button("Iniciar sesión", on_click=st.login)  # proveedor por defecto en [auth]
        st.stop()
