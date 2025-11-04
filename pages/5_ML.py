# pages/4_Product_Recommendations.py
# Product-to-Product Recommendations (item–item KNN)
import math
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from simple_auth import ensure_login, logout_button

# ====== OPTIONAL: charts ======
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

# ====== Supabase client loader ======
try:
    from supabase import create_client
except Exception:
    create_client = None

# ------------------------
# PAGE CONFIG & AUTH
# ------------------------
st.set_page_config(page_title="Product Recommendations", page_icon="🧠", layout="wide")
user = ensure_login()
with st.sidebar:
    logout_button()
st.caption(f"Session: {user}")
st.title("🧠 Product Recommendations")

# ------------------------
# HELPERS (minimal)
# ------------------------
def _load_supabase_client(secret_key: str):
    sec = st.secrets.get(secret_key, None)
    if not sec or not create_client:
        return None
    url = sec.get("url")
    key = sec.get("anon_key")
    if not url or not key:
        return None
    return create_client(url, key)

def _title_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {c: c.replace("_", " ").title() for c in df.columns}
    return df.rename(columns=mapping)

# ------------------------
# LOADERS
# ------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_recos() -> pd.DataFrame:
    """Lee la tabla de resultados product_recos (model output)."""
    sb = _load_supabase_client("supabase_sales")
    if not sb:
        st.error("No Supabase client found in secrets['supabase_sales'].")
        return pd.DataFrame()

    table_name = st.secrets.get("supabase_sales", {}).get("reco_table", "product_recos")

    rows, limit, offset = [], 1000, 0
    while True:
        q = (sb.table(table_name)
               .select("*")
               .range(offset, offset + limit - 1)
               .execute())
        data = q.data or []
        rows.extend(data)
        got = len(data)
        if got < limit:
            break
        offset += got
    return pd.DataFrame(rows)

@st.cache_data(ttl=600, show_spinner=False)
def load_product_catalog() -> pd.DataFrame:
    """Lee el perfil de productos desde la vista v_product_profile."""
    sb = _load_supabase_client("supabase_sales")
    if not sb:
        return pd.DataFrame()

    view_name = st.secrets.get("supabase_sales", {}).get("product_view", "v_product_profile")

    rows, limit, offset = [], 1000, 0
    while True:
        q = (sb.table(view_name)
               .select("*")
               .range(offset, offset + limit - 1)
               .execute())
        data = q.data or []
        rows.extend(data)
        got = len(data)
        if got < limit:
            break
        offset += got

    df = pd.DataFrame(rows)
    # normaliza columnas esperadas
    expected = ["product_id", "commoditie", "label", "unit", "organic", "coo", "product_name"]
    for c in expected:
        if c not in df.columns:
            df[c] = None
    return df[expected]

@st.cache_data(ttl=600, show_spinner=False)
def load_sales_products() -> pd.DataFrame:
    """Solo para poblar el selector: lista de productos observados en ventas."""
    sb = _load_supabase_client("supabase_sales")
    if not sb:
        return pd.DataFrame()
    table_name = st.secrets.get("supabase_sales", {}).get("table", "ventas_frutto")
    q = (sb.table(table_name).select("product").execute())
    df = pd.DataFrame(q.data or [])
    if df.empty:
        return df
    s = df["product"].dropna().astype(str).unique().tolist()
    return pd.DataFrame({"product_id": sorted(s)})

# ------------------------
# DATA
# ------------------------
recos = load_recos()
catalog = load_product_catalog()
sales_products = load_sales_products()

if recos.empty:
    st.warning("No hay datos en product_recos aún.")
    st.stop()

# ------------------------
# SIDEBAR – filtros de negocio
# ------------------------
with st.sidebar:
    st.subheader("Filters")
    topk = st.slider("Top-N", min_value=3, max_value=30, value=10, step=1)
    same_commod = st.checkbox("Misma commoditie", value=True)
    same_unit = st.checkbox("Misma unidad", value=True)
    same_organic = st.checkbox("Mismo orgánico", value=False)
    min_score = st.slider("Score mínimo", 0.0, 1.0, 0.0, 0.05)

# ------------------------
# SELECTOR DE PRODUCTO
# ------------------------
# prioridad para nombres “conocidos” de ventas; si no, usa lo que haya en catalog/recos
options = []
if not sales_products.empty:
    options = sales_products["product_id"].tolist()
elif not catalog.empty:
    options = sorted(catalog["product_id"].dropna().unique().tolist())
else:
    options = sorted(recos["product_id"].dropna().astype(str).unique().tolist())

pid = st.selectbox("Elige un producto", options, index=0 if options else None, placeholder="Selecciona…")

if not pid:
    st.info("Selecciona un producto para ver sus similares.")
    st.stop()

# ------------------------
# ENSAMBLAR TABLA PARA DISPLAY
# ------------------------
# une metadatos del producto origen
src_meta = catalog[catalog["product_id"] == pid].head(1)
src_info = src_meta.iloc[0].to_dict() if not src_meta.empty else {}

# recomendaciones crudas
rec = (recos[recos["product_id"] == pid]
       .copy()
       .sort_values("rank")
       .reset_index(drop=True))

# meta de recomendados
cat2 = catalog.rename(columns={
    "product_id": "reco_product_id",
    "commoditie": "commoditie_reco",
    "label": "label_reco",
    "unit": "unit_reco",
    "organic": "organic_reco",
    "coo": "coo_reco",
    "product_name": "product_name_reco",
})
rec = rec.merge(cat2, on="reco_product_id", how="left")

# aplica filtros de negocio
if same_commod and "commoditie" in src_info and src_info.get("commoditie"):
    rec = rec[rec["commoditie_reco"] == src_info.get("commoditie")]
if same_unit and "unit" in src_info and src_info.get("unit"):
    rec = rec[rec["unit_reco"] == src_info.get("unit")]
if same_organic and "organic" in src_info and src_info.get("organic") is not None:
    rec = rec[rec["organic_reco"] == src_info.get("organic")]

rec = rec[rec["score"].fillna(0) >= min_score]
rec = rec.head(topk)

# ------------------------
# DISPLAY
# ------------------------
colA, colB = st.columns([1, 2])
with colA:
    st.markdown("**Producto seleccionado**")
    st.json({
        "product_id": pid,
        "commoditie": src_info.get("commoditie", ""),
        "label": src_info.get("label", ""),
        "unit": src_info.get("unit", ""),
        "organic": src_info.get("organic", ""),
        "origin": src_info.get("coo", ""),
        "name": src_info.get("product_name", "")
    }, expanded=False)

with colB:
    st.markdown("**Recomendaciones**")
    if rec.empty:
        st.info("No hay recomendaciones con los filtros actuales.")
    else:
        disp = rec[[
            "reco_product_id", "product_name_reco",
            "commoditie_reco", "label_reco", "unit_reco",
            "organic_reco", "coo_reco",
            "score", "rank"
        ]].rename(columns={
            "reco_product_id": "product_id",
            "product_name_reco": "name",
            "commoditie_reco": "commoditie",
            "label_reco": "label",
            "unit_reco": "unit",
            "organic_reco": "organic",
            "coo_reco": "coo",
        })
        disp = _title_cols(disp)
        st.dataframe(disp, use_container_width=True, hide_index=True)

        if ALTAIR_OK and len(rec) > 0:
            chart = alt.Chart(rec).mark_bar().encode(
                x=alt.X("score:Q", title="Similarity score"),
                y=alt.Y("product_name_reco:N", sort="-x", title="Recommended Product"),
                tooltip=["reco_product_id","product_name_reco","score","rank","commoditie_reco","unit_reco","organic_reco"]
            ).properties(height=360)
            st.altair_chart(chart, use_container_width=True)

st.caption("Fuente: product_recos (batch nocturno) + v_product_profile para metadatos.")

