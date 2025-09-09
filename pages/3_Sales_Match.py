# pages/3_Sales_Match.py
# Recommender: Vendors para clientes basado en compras recientes + Daily Sheet (quotations)

import os
import re
import math
import datetime as dt
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st

# --- Opcionales no-críticos ---
try:
    from supabase import create_client  # pip install supabase
except Exception:
    create_client = None

# ------------------------
# CONFIG & PAGE
# ------------------------
st.set_page_config(page_title="Sales Match — Vendor Recommender", page_icon="🧩", layout="wide")
st.title("🧩 Sales Match: Recomendador de Proveedores")
st.caption("Cruza ventas recientes con cotizaciones del Daily Sheet para sugerir vendors por cliente y producto.")

# ------------------------
# HELPERS: Secrets & Clientes
# ------------------------
def _load_supabase_client(secret_key: str):
    """Crea cliente Supabase desde st.secrets[secret_key] con keys: url, anon_key, schema(optional)"""
    sec = st.secrets.get(secret_key, None)
    if not sec or not create_client:
        return None
    url = sec.get("url")
    key = sec.get("anon_key")
    if not url or not key:
        return None
    return create_client(url, key)

def _safe_to_datetime(s):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(None)

def _coerce_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return False
    s = str(x).strip().lower()
    return s in {"true", "t", "1", "yes", "y", "og"}  # tratamos "OG" como verdadero

def _normalize_txt(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.lower()
    s = re.sub(r"[\s/_\-]+", " ", s)
    s = re.sub(r"[^\w\s\.\+]", "", s)  # deja alfanum, espacio y puntos/+
    return s.strip()

# --- Diccionarios de sinónimos/mapeos (ajústalos a tu realidad) ---
PRODUCT_SYNONYMS = {
    # “familias”/commodities
    "round tomato": {"round tomato", "tomato round", "tomato rounds", "tomato", "vr round tomato", "1 layer vr round tomato", "1layer vr round tomato", "vr tomato round"},
    "beef tomato": {"beef", "beef tomato", "beefsteak", "beef steak tomato"},
    "kabocha": {"kabocha", "kabocha squash"},
    "spaghetti": {"spaghetti", "spaghetti squash"},
    "acorn": {"acorn", "acorn squash"},
    "butter": {"butter", "butternut", "butternut squash"},
    "eggplant": {"eggplant", "berenjena"},
    # agrega tus productos fuertes aquí...
}

def _build_reverse_synonyms(syno: Dict[str, set]) -> Dict[str, str]:
    rev = {}
    for canon, variants in syno.items():
        for v in variants:
            rev[_normalize_txt(v)] = canon
    return rev

REV_SYNONYMS = _build_reverse_synonyms(PRODUCT_SYNONYMS)

SIZE_PATTERN = re.compile(r"(\d+\s?lb|\d+\s?ct|\d+\s?[xX]\s?\d+|bulk|jbo|xl|lg|med|fancy|4x4|4x5|5x5|60cs|15 lb|25 lb|2 layers?)", re.IGNORECASE)

def _extract_size(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = SIZE_PATTERN.search(s)
    return m.group(1).lower() if m else ""

def _canonical_product(txt: str) -> str:
    n = _normalize_txt(txt)
    # mapea por sinónimos; si no encuentra, devuelve palabra “principal” del texto
    if n in REV_SYNONYMS:
        return REV_SYNONYMS[n]
    # intenta con tokens claves
    for canon, variants in PRODUCT_SYNONYMS.items():
        if any(_normalize_txt(v) in n for v in variants):
            return canon
    # fallback: primera palabra
    return n.split(" ")[0] if n else ""

def _vendor_clean(s: str) -> str:
    return _normalize_txt(s)

def _loc_clean(s: str) -> str:
    return _normalize_txt(s)

# ------------------------
# LOADERS
# ------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_quotations() -> pd.DataFrame:
    """
    Carga cotizaciones (Daily Sheet) desde Supabase.
    Tolerante a distintos nombres de columnas (alias_map).
    """
    sb = _load_supabase_client("supabase_quotes")
    if not sb:
        st.error("No se pudo crear cliente Supabase para cotizaciones (st.secrets['supabase_quotes']).")
        return pd.DataFrame()

    table_name = st.secrets.get("supabase_quotes", {}).get("table", "quotations")
    # pull en páginas si es grande
    rows = []
    limit = 1000
    offset = 0
    while True:
        q = sb.table(table_name).select("*").range(offset, offset + limit - 1).execute()
        data = q.data or []
        rows.extend(data)
        if len(data) < limit:
            break
        offset += limit

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Aliases comunes → estándar
    alias_map = {
        "cotization_date": ["cotization_date", "date", "Date"],
        "organic": ["organic", "OG/CV", "og_cv", "ogcv"],
        "product": ["product", "Product"],
        "price": ["price", "Price", "price$"],
        "location": ["location", "Where", "where"],
        "vendorclean": ["vendorclean", "Vendor", "Shipper", "vendor"],
        "size": ["size", "Size", "volume_standard"],  # si no hay size, guarda volume_standard
    }
    # crea columnas “estándar” con fallbacks
    std = {}
    for std_col, candidates in alias_map.items():
        for c in candidates:
            if c in df.columns:
                std[std_col] = df[c]
                break
        if std_col not in std:
            std[std_col] = pd.Series([None]*len(df))
    qdf = pd.DataFrame(std)

    # Casting / normalización
    qdf["date"] = pd.to_datetime(qdf["cotization_date"], errors="coerce")
    qdf["is_organic"] = qdf["organic"].apply(_coerce_bool)
    qdf["price"] = pd.to_numeric(qdf["price"], errors="coerce")
    qdf["product_raw"] = qdf["product"].astype(str)
    qdf["product_canon"] = qdf["product_raw"].apply(_canonical_product)
    qdf["size_std"] = qdf["size"].astype(str).apply(_extract_size)
    qdf["vendor_c"] = qdf["vendorclean"].astype(str).apply(_vendor_clean)
    qdf["loc_c"] = qdf["location"].astype(str).apply(_loc_clean)

    # Filtro de saneo
    qdf = qdf.dropna(subset=["price"])
    return qdf[["date", "is_organic", "price", "product_raw", "product_canon", "size_std", "vendor_c", "loc_c"]]

@st.cache_data(ttl=300, show_spinner=False)
def load_sales() -> pd.DataFrame:
    """
    Carga ventas. Intenta Supabase; si no, deja ganchos para SQL Server/Azure.
    """
    # 1) Supabase
    sb = _load_supabase_client("supabase_sales")
    if sb:
        table_name = st.secrets.get("supabase_sales", {}).get("table", "sales")
        rows, limit, offset = [], 1000, 0
        while True:
            q = sb.table(table_name).select("*").range(offset, offset + limit - 1).execute()
            data = q.data or []
            rows.extend(data)
            if len(data) < limit:
                break
            offset += limit
        df = pd.DataFrame(rows)
    else:
        # 2) Azure SQL (opcional) — deja el esqueleto por si lo estás usando
        ms = st.secrets.get("mssql_sales", None)
        if ms:
            st.warning("Loader MSSQL no implementado aquí por brevedad. Usa sqlalchemy/pyodbc y devuélveme un DataFrame.")
            return pd.DataFrame()
        else:
            st.error("No se encontró fuente de ventas (st.secrets['supabase_sales'] o ['mssql_sales']).")
            return pd.DataFrame()

    if df.empty:
        return df

    # Campos del ejemplo del usuario (tolerantes)
    alias_map = {
        "received_date": ["received_date", "reqs_date", "created_at", "sale_date"],
        "product": ["product", "commoditie", "buyer_product"],
        "organic": ["organic", "is_organic", "OG/CV"],
        "unit": ["unit"],
        "customer": ["customer", "client", "buyer"],
        "vendor": ["vendor", "shipper", "supplier"],
        "sale_location": ["sale_location", "lot_location"],
        "quantity": ["quantity", "qty"],
        "price_per_unit": ["price_per_unit", "price", "unit_price", "sell_price"],
    }
    std = {}
    for std_col, candidates in alias_map.items():
        for c in candidates:
            if c in df.columns:
                std[std_col] = df[c]
                break
        if std_col not in std:
            std[std_col] = pd.Series([None]*len(df))

    sdf = pd.DataFrame(std)
    sdf["date"] = pd.to_datetime(sdf["received_date"], errors="coerce")
    sdf["is_organic"] = sdf["organic"].apply(_coerce_bool)
    sdf["product_raw"] = sdf["product"].astype(str)
    sdf["product_canon"] = sdf["product_raw"].apply(_canonical_product)
    sdf["size_std"] = sdf["unit"].astype(str).apply(_extract_size)
    sdf["customer_c"] = sdf["customer"].astype(str).apply(_normalize_txt)
    sdf["vendor_c"] = sdf["vendor"].astype(str).apply(_vendor_clean)
    sdf["loc_c"] = sdf["sale_location"].astype(str).apply(_loc_clean)
    sdf["quantity"] = pd.to_numeric(sdf["quantity"], errors="coerce")
    sdf["price_per_unit"] = pd.to_numeric(sdf["price_per_unit"], errors="coerce")

    return sdf[[
        "date", "is_organic", "product_raw", "product_canon", "size_std",
        "customer_c", "vendor_c", "loc_c", "quantity", "price_per_unit"
    ]]

# ------------------------
# MATCHING & SCORING
# ------------------------
def recent_purchases(sales: pd.DataFrame, days_window: int) -> pd.DataFrame:
    if sales.empty:
        return sales
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_window)
    # Maneja tz-naive
    sales_dates = sales["date"]
    # reemplaza NaT por -inf para no romper filtro
    sales["date"] = pd.to_datetime(sales_dates, errors="coerce")
    filt = sales["date"] >= cutoff
    out = sales.loc[filt].copy()
    # Producto representativo por cliente (suma qty)
    agg = (out
           .groupby(["customer_c", "product_canon", "is_organic", "size_std", "loc_c"], dropna=False)[["quantity"]]
           .sum()
           .reset_index())
    return agg

def candidate_vendors(quot: pd.DataFrame, prod_canon: str, is_og: bool, loc: str, size_hint: str) -> pd.DataFrame:
    if quot.empty:
        return quot
    df = quot.copy()
    # Filtros suaves: misma familia + orgánico igual
    df = df[df["product_canon"] == prod_canon]
    df = df[df["is_organic"] == bool(is_og)]

    # Boost por ubicación y por tamaño
    df["loc_match"] = (df["loc_c"] == loc).astype(int)
    if size_hint:
        df["size_match"] = (df["size_std"] == _normalize_txt(size_hint)).astype(int)
    else:
        df["size_match"] = 0

    # Precio relativo (zscore) dentro del subconjunto
    df["price_z"] = (df["price"] - df["price"].mean()) / (df["price"].std(ddof=0) + 1e-9)

    # Score base: precio bajo (z negativo), + ubicación, + tamaño
    # Escala y pesos ajustables
    df["score"] = (-1.0 * df["price_z"]) + (0.5 * df["loc_match"]) + (0.3 * df["size_match"])
    return df

def add_familiarity_score(cands: pd.DataFrame, prior_vendors: List[str]) -> pd.DataFrame:
    if cands.empty:
        return cands
    pv = {_vendor_clean(v) for v in prior_vendors if isinstance(v, str)}
    c = cands.copy()
    c["familiar"] = c["vendor_c"].apply(lambda v: 1 if v in pv else 0)
    c["score"] = c["score"] + 0.2 * c["familiar"]  # pequeño plus por familiaridad
    return c

def build_recommendations(sales_recent: pd.DataFrame, sales_all: pd.DataFrame, quotations: pd.DataFrame, top_k:int = 5) -> pd.DataFrame:
    """
    Para cada (customer, product), trae candidatos de quotations y arma top_k por score.
    """
    recs = []
    if sales_recent.empty or quotations.empty:
        return pd.DataFrame(columns=["customer", "product", "is_organic", "size", "loc",
                                     "vendor", "price", "score", "why"])

    # vendors previos por cliente (familiaridad)
    hist = (sales_all.groupby(["customer_c"])["vendor_c"]
            .agg(lambda s: list(set([_vendor_clean(x) for x in s if isinstance(x,str)])))
            .to_dict())

    for _, row in sales_recent.iterrows():
        cust = row["customer_c"]
        prod = row["product_canon"]
        is_og = bool(row["is_organic"])
        size = row.get("size_std", "")
        loc = row.get("loc_c", "")

        cands = candidate_vendors(quotations, prod, is_og, loc, size)
        if cands.empty:
            continue

        prior = hist.get(cust, [])
        cands = add_familiarity_score(cands, prior)

        top = (cands
               .sort_values(["score", "price"], ascending=[False, True])
               .head(top_k)
               .copy())

        for _, r in top.iterrows():
            why_bits = []
            if r["loc_match"] == 1:
                why_bits.append("ubicación coincide")
            if r["size_match"] == 1:
                why_bits.append("tamaño coincide")
            if r.get("familiar", 0) == 1:
                why_bits.append("proveedor ya conocido")
            why = ", ".join(why_bits) if why_bits else "mejor relación precio/filtros"

            recs.append({
                "customer": cust,
                "product": prod,
                "is_organic": is_og,
                "size": size or "",
                "loc": loc or "",
                "vendor": r["vendor_c"],
                "price": float(r["price"]),
                "score": float(r["score"]),
                "why": why,
            })

    recdf = pd.DataFrame(recs)
    if recdf.empty:
        return recdf
    # Ordena dentro de cada (customer, product)
    recdf = recdf.sort_values(["customer", "product", "score", "price"], ascending=[True, True, False, True])
    return recdf

# ------------------------
# UI CONTROLS
# ------------------------
with st.sidebar:
    st.subheader("Filtros")
    days = st.slider("Ventana de días (ventas recientes)", min_value=3, max_value=60, value=14, step=1)
    topk = st.slider("Top-K recomendaciones", min_value=3, max_value=20, value=5, step=1)

# ------------------------
# DATA PIPELINE
# ------------------------
qdf = load_quotations()
sdf = load_sales()

col1, col2 = st.columns(2)
with col1:
    st.metric("Cotizaciones cargadas", value=len(qdf))
with col2:
    st.metric("Registros de ventas", value=len(sdf))

if qdf.empty or sdf.empty:
    st.stop()

# Selección de clientes (solo los que tienen ventas en ventana)
sdf_recent = recent_purchases(sdf, days_window=days)
customers = sorted(sdf_recent["customer_c"].dropna().unique().tolist())
sel_customers = st.multiselect("Clientes", customers, default=customers[: min(10, len(customers))])

if not sel_customers:
    st.info("Selecciona al menos un cliente para ver recomendaciones.")
    st.stop()

subset_recent = sdf_recent[sdf_recent["customer_c"].isin(sel_customers)].copy()
st.write("**Compras recientes por cliente y producto (agregado por cantidad):**")
st.dataframe(subset_recent.rename(columns={
    "customer_c": "customer", "product_canon": "product", "is_organic": "organic",
    "size_std": "size", "loc_c": "location", "quantity": "qty"
}))

# ------------------------
# BUILD & SHOW RECOMMENDATIONS
# ------------------------
recs = build_recommendations(subset_recent, sdf, qdf, top_k=topk)

if recs.empty:
    st.warning("No hay candidatos de vendors en las cotizaciones que cumplan con tus filtros.")
    st.stop()

# Vista general
st.subheader("Recomendaciones")
st.caption("Score combina precio (mejor = mayor score), coincidencia de ubicación/tamaño y familiaridad (si ya compró a ese vendor).")

# Orden amigable
recs_show = recs.copy()
recs_show["organic"] = recs_show["is_organic"].map({True: "OG", False: "CV"})
recs_show = recs_show.drop(columns=["is_organic"], errors="ignore")

# Tabla principal
st.dataframe(
    recs_show[["customer", "product", "organic", "size", "loc", "vendor", "price", "score", "why"]]
)

# Insight rápido: vendors más recomendados
st.subheader("Vendors más recomendados (resumen)")
vend_counts = (recs_show.groupby("vendor")
               .agg(recs=("vendor", "count"), avg_price=("price", "mean"), avg_score=("score", "mean"))
               .reset_index()
               .sort_values(["avg_score", "recs"], ascending=[False, False]))
st.dataframe(vend_counts)

# ------------------------
# NOTAS DE AJUSTE (para ti):
# ------------------------
st.markdown("""
**Notas/ajustes que puedes tocar rápido:**
- `PRODUCT_SYNONYMS`: añade variantes que veas en ventas/cotizaciones para mejorar el “match”.
- Pesos del score en `candidate_vendors`: precio (z) = -1.0, ubicación = 0.5, tamaño = 0.3. Familiaridad suma +0.2.
- Si quieres filtrar cotizaciones por fecha reciente (p.ej. últimos 7 días), puedes filtrar `qdf` antes de usarlo.
- Si tu tabla de ventas vive en SQL Server/Azure, completa `st.secrets["mssql_sales"]` y reemplaza el loader por tu query SQL.
- Si en tus cotizaciones **no** tienes `size`, el sistema usará `volume_standard` o ignorará el match de tamaño.
""")
