import time
import pandas as pd
import streamlit as st
from supabase import create_client, Client

st.set_page_config(page_title="Daily Metrics — Supabase Sales", page_icon="📈", layout="centered")
st.title("Daily Metrics — Supabase [sales] 📈")
st.caption("Registra actividad comercial (Reached / Engaged / Closed) en el proyecto *supabase_sales*.")

# --- Credenciales desde secrets ---
# Debes definir en .streamlit/secrets.toml una sección:
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

TABLE_NAME = "daily_metrics"  # Debe existir en el proyecto supabase_sales

with st.form("metrics_form_sales", clear_on_submit=False):
    user_name = st.text_input("Nombre de usuario", value="", placeholder="p.ej. Sergio Lopez")
    c1, c2, c3 = st.columns(3)
    with c1:
        clients_reached_out = st.number_input("CLIENTS REACHED OUT", min_value=0, step=1, value=0)
    with c2:
        clients_engaged = st.number_input("CLIENTS ENGAGED", min_value=0, step=1, value=0)
    with c3:
        clients_closed = st.number_input("CLIENTS CLOSED", min_value=0, step=1, value=0)
    submitted = st.form_submit_button("Guardar en SALES")

if submitted:
    if not user_name.strip():
        st.warning("El nombre de usuario es obligatorio.")
    elif clients_reached_out < (clients_engaged + clients_closed):
        st.warning("Reached Out no puede ser menor que Engaged + Closed.")
    else:
        try:
            payload = {
                "user_name": user_name.strip(),
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
st.subheader("Últimos registros (Sales)")
with st.expander("Filtros"):
    c1, c2 = st.columns([2,1])
    with c1:
        user_filter = st.text_input("Filtrar por usuario (contiene)", value="")
    with c2:
        limit = st.number_input("Límites de filas", 5, 200, 50, step=5)

try:
    query = supabase.table(TABLE_NAME).select("*").order("created_at", desc=True)
    if user_filter.strip():
        query = query.ilike("user_name", f"%{user_filter.strip()}%").limit(int(limit))
    else:
        query = query.limit(int(limit))
    res = query.execute()
    rows = res.data or []

    if not rows:
        st.info("No hay registros aún en SALES.")
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

st.caption("Esta página apunta al proyecto Supabase **supabase_sales** y a la tabla public.daily_metrics.")
