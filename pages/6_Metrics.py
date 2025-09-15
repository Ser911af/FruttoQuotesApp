
import os
import time
import pandas as pd
import streamlit as st

# Supabase Python client v2
from supabase import create_client, Client

st.set_page_config(page_title="Daily Metrics — FruttoFoods", page_icon="🥑", layout="centered")

st.title("Daily Metrics — Ingresos a Supabase 🥑")
st.caption("Registra clientes alcanzados, engaged y cerrados por usuario.")

# --- Configuración de credenciales ---
# Agrega esto a .streamlit/secrets.toml:
# SUPABASE_URL = "https://TU-PROYECTO.supabase.co"
# SUPABASE_ANON_KEY = "ey..."
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Faltan credenciales en st.secrets: SUPABASE_URL y SUPABASE_ANON_KEY.")
    st.stop()

# Inicializa cliente
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

TABLE_NAME = "daily_metrics"  # Debe existir en Supabase (ver SQL que compartimos antes)

with st.form("metrics_form", clear_on_submit=False):
    user_name = st.text_input("Nombre de usuario", value="", placeholder="p.ej. Sergio Lopez")
    col1, col2, col3 = st.columns(3)
    with col1:
        clients_reached_out = st.number_input("CLIENTS REACHED OUT", min_value=0, step=1, value=0)
    with col2:
        clients_engaged = st.number_input("CLIENTS ENGAGED", min_value=0, step=1, value=0)
    with col3:
        clients_closed = st.number_input("CLIENTS CLOSED", min_value=0, step=1, value=0)

    submitted = st.form_submit_button("Guardar registro")

if submitted:
    # Validaciones rápidas
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
                st.success(f"¡Guardado! ID: {res.data[0].get('id', '—')}")
                # pequeña pausa para que el índice 'created_at' se refleje
                time.sleep(0.3)
            else:
                st.info("Insert realizado, pero no se devolvieron datos.")
        except Exception as e:
            st.error(f"Error al insertar: {e}")

st.markdown("---")
st.subheader("Últimos registros")
# Filtros opcionales
with st.expander("Filtros"):
    c1, c2 = st.columns([2,1])
    with c1:
        user_filter = st.text_input("Filtrar por usuario (contiene)", value="")
    with c2:
        limit = st.number_input("Límites de filas", 5, 200, 50, step=5)

# Construcción de la query
try:
    query = supabase.table(TABLE_NAME).select("*").order("created_at", desc=True)
    if user_filter.strip():
        # ilike para búsqueda case-insensitive
        query = query.ilike("user_name", f"%{user_filter.strip()}%")

    query = query.limit(int(limit))
    res = query.execute()
    rows = res.data or []

    if not rows:
        st.info("No hay registros aún.")
    else:
        # DataFrame ordenado por fecha desc
        df = pd.DataFrame(rows)
        # Ordena columnas amigablemente si existen
        ordered_cols = [c for c in ["created_at","user_name","clients_reached_out","clients_engaged","clients_closed","id"] if c in df.columns] + [c for c in df.columns if c not in ["created_at","user_name","clients_reached_out","clients_engaged","clients_closed","id"]]
        df = df[ordered_cols]

        st.dataframe(df, use_container_width=True, hide_index=True)

        # KPIs rápidos
        st.markdown("### Totales en pantalla")
        k1, k2, k3 = st.columns(3)
        k1.metric("Reached Out (sum)", int(df.get("clients_reached_out", pd.Series(dtype=int)).sum()))
        k2.metric("Engaged (sum)", int(df.get("clients_engaged", pd.Series(dtype=int)).sum()))
        k3.metric("Closed (sum)", int(df.get("clients_closed", pd.Series(dtype=int)).sum()))
except Exception as e:
    st.error(f"Error al consultar datos: {e}")

st.markdown("---")
st.caption("Tip: Esta tabla es complementaria a tus tablas de ventas (sales). Sirve para tracking de pipeline comercial por usuario y día.")
