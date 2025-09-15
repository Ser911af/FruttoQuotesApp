import time
import pandas as pd
import streamlit as st
from supabase import create_client, Client
from simple_auth import ensure_login, logout_button

# =============================================
# Daily Metrics — Supabase [sales] (simple_auth)
# Requiere políticas RLS de demo (allow_insert_demo, allow_select_demo)
# =============================================

# ✅ Login obligatorio con tu simple_auth (NO Supabase Auth)
user = ensure_login()
with st.sidebar:
    logout_button()

st.set_page_config(page_title="Daily Metrics — Supabase Sales", page_icon="📈", layout="centered")
st.title("Daily Metrics — Supabase [sales] 📈")
st.caption(f"Sesión: {user}")
st.caption("Registra actividad comercial (Reached / Engaged / Closed) en el proyecto *supabase_sales*.")

# --- Credenciales desde secrets (.streamlit/secrets.toml) ---
# [supabase_sales]
# url = "https://TU-PROYECTO-SALES.supabase.co"
# anon_key = "ey..."
if "supabase_sales" not in st.secrets:
    st.error("Faltan credenciales: agrega el bloque [supabase_sales] en .streamlit/secrets.toml (url y anon_key).")
    st.stop()

SUPABASE_URL = st.secrets["supabase_sales"].get("url", "")
SUPABASE_ANON_KEY = st.secrets["supabase_sales"].get("anon_key", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Completa supabase_sales.url y supabase_sales.anon_key en secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Usuario de sesión desde simple_auth ---
session_name = str(user)  # se usará como user_name en los inserts

TABLE_NAME = "daily_metrics"  # Debe existir en el proyecto supabase_sales

with st.form("metrics_form_sales", clear_on_submit=False):
    # El nombre viene de la sesión y no es editable
    st.text_input("Nombre de usuario", value=session_name, disabled=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        clients_reached_out = st.number_input("CLIENTS REACHED OUT", min_value=0, step=1, value=0)
    with c2:
        clients_engaged = st.number_input("CLIENTS ENGAGED", min_value=0, step=1, value=0)
    with c3:
        clients_closed = st.number_input("CLIENTS CLOSED", min_value=0, step=1, value=0)

    submitted = st.form_submit_button("Guardar en SALES")

if submitted:
    if clients_reached_out < (clients_engaged + clients_closed):
        st.warning("Reached Out no puede ser menor que Engaged + Closed.")
    else:
        try:
            payload = {
                "user_name": session_name.strip(),
                "clients_reached_out": int(clients_reached_out),
                "clients_engaged": int(clients_engaged),
                "clients_closed": int(clients_closed),
            }
            res = supabase.table(TABLE_NAME).insert(payload).execute()
            if getattr(res, "data", None):
                st.success(f"¡Guardado en SALES! ID: {res.data[0].get('id', '—')}")
                time.sleep(0.3)
            else:
                st.info("Insert realizado, pero no se devolvieron datos.")
        except Exception as e:
            st.error(f"Error al insertar: {e}")

st.markdown("---")
st.subheader("Mis últimos registros (Sales)")
with st.expander("Filtros"):
    c1, c2 = st.columns([2,1])
    with c1:
        user_filter = st.text_input("Filtrar por usuario (contiene)", value=session_name)
    with c2:
        limit = st.number_input("Límites de filas", 5, 200, 50, step=5)

try:
    # Con la política select demo, podremos ver todas las filas; nos filtramos por nombre.
    query = supabase.table(TABLE_NAME).select("*").order("created_at", desc=True)
    if user_filter.strip():
        query = query.ilike("user_name", f"%{user_filter.strip()}%")

    res = query.limit(int(limit)).execute()
    rows = res.data or []

    if not rows:
        st.info("No hay registros aún en SALES para el filtro aplicado.")
    else:
        df = pd.DataFrame(rows)
        ordered_cols = [c for c in ["created_at","user_name","clients_reached_out","clients_engaged","clients_closed","id"] if c in df.columns] + [c for c in df.columns if c not in ["created_at","user_name","clients_reached_out","clients_engaged","clients_closed","id"]]
        df = df[ordered_cols]
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### Totales en pantalla (Sales)")
        k1, k2, k3 = st.columns(3)
        k1.metric("Reached Out (sum)", int(df.get("clients_reached_out", pd.Series(dtype=int)).sum()))
        k2.metric("Engaged (sum)", int(df.get("clients_engaged", pd.Series(dtype=int)).sum()))
        k3.metric("Closed (sum)", int(df.get("clients_closed", pd.Series(dtype=int)).sum()))
except Exception as e:
    st.error(f"Error al consultar datos: {e}")

st.caption("Esta página usa simple_auth (no Supabase Auth) y requiere políticas RLS de demo para INSERT/SELECT.")
