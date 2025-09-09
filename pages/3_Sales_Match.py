import streamlit as st
import pandas as pd

def probe_sales_one_row():
    from supabase import create_client
    blk = st.secrets.get("supabase_sales", {})
    url = blk.get("url"); key = blk.get("anon_key"); table = blk.get("table", "ventas_frutto")
    schema = blk.get("schema", "public")
    if not url or not key:
        st.error("Faltan url/anon_key en [supabase_sales].")
        return

    st.write("🔌 Probing SALES →", url.split("//")[-1].split("/")[0], "·", f"{schema}.{table}")
    sb = create_client(url, key)

    try:
        # 1) Prueba con * (1 fila)
        resp = sb.schema(schema).table(table).select("*").limit(1).execute()
        st.write("HTTP OK. data length:", len(resp.data or []))
        st.json((resp.data or [])[:1])
    except Exception as e:
        st.error("Error SELECT * LIMIT 1:")
        st.code(str(e))

    try:
        # 2) Si falla *, prueba con columnas muuuuy básicas que seguro existan
        resp2 = sb.schema(schema).table(table).select("id").limit(1).execute()
        st.write("Intento SELECT id LIMIT 1:", len(resp2.data or []))
        st.json((resp2.data or [])[:1])
    except Exception as e:
        st.error("Error SELECT id LIMIT 1:")
        st.code(str(e))

st.subheader("🔎 Diagnóstico rápido SALES")
probe_sales_one_row()
