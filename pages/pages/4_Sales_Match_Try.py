# pages/3_Sales_Match.py
# Recommender y Match de Vendors contra Daily Sheet del día
# - Muestra por defecto SOLO las cotizaciones del día seleccionado (hoy por defecto, zona horaria Bogotá)
# - Hace match de vendors: ¿quién nos ha vendido recientemente y aparece en el Daily Sheet de hoy?
# - Mantiene sugerencias por cliente/producto usando solo el Daily Sheet del día (opcional)

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
st.caption("Cruza ventas recientes con cotizaciones del Daily Sheet para sugerir vendors y destacar quiénes ya nos vendieron.")

# ------------------------
# HELPERS: Secrets & Normalización
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
    if n in REV_SYNONYMS:
        return REV_SYNONYMS[n]
    for canon, variants in PRODUCT_SYNONYMS.items():
        if any(_normalize_txt(v) in n for v in variants):
            return canon
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

    alias_map = {
        "cotization_date": ["cotization_date", "date", "Date"],
        "organic": ["organic", "OG/CV", "og_cv", "ogcv"],
        "product": ["product", "Product"],
        "price": ["price", "Price", "price$"],
        "location": ["location", "Where", "where"],
        "vendorclean": ["vendorclean", "Vendor", "Shipper", "vendor"],
        "size": ["size", "Size", "volume_standard"],
    }
    std = {}
    for std_col, candidates in alias_map.items():
        for c in candidates:
            if c in df.columns:
                std[std_col] = df[c]
                break
        if std_col not in std:
            std[std_col] = pd.Series([None] * len(df))

    qdf = pd.DataFrame(std)
    qdf["date"] = pd.to_datetime(qdf["cotization_date"], errors="coerce")
    qdf["is_organic"] = qdf["organic"].apply(_coerce_bool)
    qdf["price"] = pd.to_numeric(qdf["price"], errors="coerce")
    qdf["product_raw"] = qdf["product"].astype(str)
    qdf["product_canon"] = qdf["product_raw"].apply(_canonical_product)
    qdf["size_std"] = qdf["size"].astype(str).apply(_extract_size)
    qdf["vendor_c"] = qdf["vendorclean"].astype(str).apply(_vendor_clean)
    qdf["loc_c"] = qdf["location"].astype(str).apply(_loc_clean)

    qdf = qdf.dropna(subset=["price"])
    return qdf[["date", "is_organic", "price", "product_raw", "product_canon", "size_std", "vendor_c", "loc_c"]]


@st.cache_data(ttl=300, show_spinner=False)
def load_sales() -> pd.DataFrame:
    """Carga ventas. Intenta Supabase; si no, deja ganchos para SQL Server/Azure."""
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
        ms = st.secrets.get("mssql_sales", None)
        if ms:
            st.warning("Loader MSSQL no implementado aquí por brevedad. Usa sqlalchemy/pyodbc y devuélveme un DataFrame.")
            return pd.DataFrame()
        else:
            st.error("No se encontró fuente de ventas (st.secrets['supabase_sales'] o ['mssql_sales']).")
            return pd.DataFrame()

    if df.empty:
        return df

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
            std[std_col] = pd.Series([None] * len(df))

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
# MATCHING & SCORING (por cliente/producto)
# ------------------------
def recent_purchases(sales: pd.DataFrame, days_window: int) -> pd.DataFrame:
    if sales.empty:
        return sales
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_window)
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    out = sales.loc[sales["date"] >= cutoff].copy()
    agg = (
        out.groupby(["customer_c", "product_canon", "is_organic", "size_std", "loc_c"], dropna=False)[["quantity"]]
        .sum()
        .reset_index()
    )
    return agg


def candidate_vendors(quot: pd.DataFrame, prod_canon: str, is_og: bool, loc: str, size_hint: str) -> pd.DataFrame:
    if quot.empty:
        return quot
    df = quot.copy()
    df = df[df["product_canon"] == prod_canon]
    df = df[df["is_organic"] == bool(is_og)]
    df["loc_match"] = (df["loc_c"] == loc).astype(int)
    if size_hint:
        df["size_match"] = (df["size_std"] == _normalize_txt(size_hint)).astype(int)
    else:
        df["size_match"] = 0
    df["price_z"] = (df["price"] - df["price"].mean()) / (df["price"].std(ddof=0) + 1e-9)
    df["score"] = (-1.0 * df["price_z"]) + (0.5 * df["loc_match"]) + (0.3 * df["size_match"])
    return df


def add_familiarity_score(cands: pd.DataFrame, prior_vendors: List[str]) -> pd.DataFrame:
    if cands.empty:
        return cands
    pv = {_vendor_clean(v) for v in prior_vendors if isinstance(v, str)}
    c = cands.copy()
    c["familiar"] = c["vendor_c"].apply(lambda v: 1 if v in pv else 0)
    c["score"] = c["score"] + 0.2 * c["familiar"]
    return c


def build_recommendations(sales_recent: pd.DataFrame, sales_all: pd.DataFrame, quotations: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    recs = []
    if sales_recent.empty or quotations.empty:
        return pd.DataFrame(columns=["customer", "product", "is_organic", "size", "loc",
                                     "vendor", "price", "score", "why"])

    hist = (
        sales_all.groupby(["customer_c"])['vendor_c']
        .agg(lambda s: list(set([_vendor_clean(x) for x in s if isinstance(x, str)])))
        .to_dict()
    )

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

        top = (
            cands.sort_values(["score", "price"], ascending=[False, True]).head(top_k).copy()
        )
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
    recdf = recdf.sort_values(["customer", "product", "score", "price"], ascending=[True, True, False, True])
    return recdf


# ------------------------
# NUEVO: MATCH DE VENDORS POR DÍA (Daily Sheet)
# ------------------------
def _bogota_today() -> dt.date:
    # Bogotá = UTC-5 sin DST
    return (dt.datetime.utcnow() + dt.timedelta(hours=-5)).date()


def recent_vendor_stats(sales: pd.DataFrame, days_window: int) -> pd.DataFrame:
    """Estadísticas por vendor en ventas recientes (últimos N días)."""
    if sales.empty:
        return pd.DataFrame(columns=["vendor_c", "last_sale", "n_customers", "n_products", "qty_total"])    
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_window)
    s = sales.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.loc[s["date"] >= cutoff]
    if s.empty:
        return pd.DataFrame(columns=["vendor_c", "last_sale", "n_customers", "n_products", "qty_total"])    
    grp = s.groupby("vendor_c")
    out = pd.DataFrame({
        "vendor_c": grp.size().index,
        "last_sale": grp["date"].max().values,
        "n_customers": grp["customer_c"].nunique().values,
        "n_products": grp["product_canon"].nunique().values,
        "qty_total": grp["quantity"].sum().values,
    })
    return out


# ------------------------
# UI CONTROLS
# ------------------------
with st.sidebar:
    st.subheader("Filtros")
    days = st.slider("Ventana de días (ventas recientes)", min_value=3, max_value=60, value=14, step=1)
    topk = st.slider("Top-K recomendaciones", min_value=3, max_value=20, value=5, step=1)

    st.markdown("---")
    # Fecha del Daily Sheet (por defecto, hoy Bogotá)
    default_day = _bogota_today()
    selected_day = st.date_input("Fecha del Daily Sheet", value=default_day)
    only_matches = st.checkbox("Mostrar SOLO vendors con match (han vendido en la ventana)", value=True)

# ------------------------
# DATA PIPELINE
# ------------------------
qdf_all = load_quotations()
sdf = load_sales()

col1, col2 = st.columns(2)
with col1:
    st.metric("Cotizaciones cargadas", value=len(qdf_all))
with col2:
    st.metric("Registros de ventas", value=len(sdf))

if qdf_all.empty or sdf.empty:
    st.stop()

# 1) Filtrar Daily Sheet al día seleccionado
qdf_day = qdf_all[qdf_all["date"].dt.date == pd.to_datetime(selected_day).date()].copy()
if qdf_day.empty:
    st.warning("No hay cotizaciones para la fecha seleccionada.")

# 2) Vendors recientes en ventas (últimos N días)
vstats = recent_vendor_stats(sdf, days_window=days)
recent_vendors_set = set(vstats["vendor_c"].dropna().unique())

# 3) Match: vendors del Daily Sheet del día ∩ vendors con ventas recientes
if not qdf_day.empty:
    vendors_today = (
        qdf_day.groupby("vendor_c").agg(
            avg_price=("price", "mean"),
            best_price=("price", "min"),
            n_quotes=("vendor_c", "count"),
            locations=("loc_c", lambda s: ", ".join(sorted(set([x for x in s if isinstance(x, str) and x])))),
        ).reset_index()
    )
    vendors_today["match_recent_sales"] = vendors_today["vendor_c"].apply(lambda v: v in recent_vendors_set)
    vendors_today = vendors_today.merge(vstats, on="vendor_c", how="left")
    if only_matches:
        vendors_today_show = vendors_today[vendors_today["match_recent_sales"]].copy()
    else:
        vendors_today_show = vendors_today.copy()

    st.subheader("Vendors del Daily Sheet (día seleccionado)")
    st.caption("Señalamos cuáles ya nos vendieron en los últimos días y sus estadísticas recientes.")
    st.dataframe(
        vendors_today_show.sort_values(["match_recent_sales", "best_price"], ascending=[False, True])
    )
else:
    vendors_today_show = pd.DataFrame()

# ------------------------
# (Opcional) Compras recientes por cliente/producto + Recs usando SOLO el Daily Sheet del día
# ------------------------
sdf_recent_cust = recent_purchases(sdf, days_window=days)
customers = sorted(sdf_recent_cust["customer_c"].dropna().unique().tolist())

with st.expander("Compras recientes por cliente/producto y recomendaciones (usando Daily Sheet del día)", expanded=False):
    if not customers:
        st.info("No hay compras recientes en la ventana seleccionada.")
    else:
        sel_customers = st.multiselect("Clientes", customers, default=customers[: min(10, len(customers))])
        subset_recent = sdf_recent_cust[sdf_recent_cust["customer_c"].isin(sel_customers)].copy()
        st.write("**Compras recientes (agregado por cantidad):**")
        st.dataframe(subset_recent.rename(columns={
            "customer_c": "customer", "product_canon": "product", "is_organic": "organic",
            "size_std": "size", "loc_c": "location", "quantity": "qty"
        }))

        if not qdf_day.empty and not subset_recent.empty:
            recs = build_recommendations(subset_recent, sdf, qdf_day, top_k=topk)
            if recs.empty:
                st.warning("No hay candidatos de vendors en el Daily Sheet del día que cumplan con tus filtros.")
            else:
                st.subheader("Recomendaciones (limitadas al Daily Sheet del día)")
                recs_show = recs.copy()
                recs_show["organic"] = recs_show["is_organic"].map({True: "OG", False: "CV"})
                recs_show = recs_show.drop(columns=["is_organic"], errors="ignore")
                st.dataframe(recs_show[["customer", "product", "organic", "size", "loc", "vendor", "price", "score", "why"]])

                st.subheader("Vendors más recomendados (resumen)")
                vend_counts = (
                    recs_show.groupby("vendor")
                    .agg(recs=("vendor", "count"), avg_price=("price", "mean"), avg_score=("score", "mean"))
                    .reset_index()
                    .sort_values(["avg_score", "recs"], ascending=[False, False])
                )
                st.dataframe(vend_counts)

# ------------------------
# NOTAS DE AJUSTE
# ------------------------
st.markdown(
    """
**Notas/ajustes rápidos:**
- La sección principal ahora filtra por **fecha del Daily Sheet** (por defecto, hoy Bogotá) y hace **match de vendors** con ventas recientes (ventana configurable).
- Si deseas que el match solo considere cotizaciones del mismo **Location**, agrega `merge` adicional por `loc_c` o filtra `qdf_day` por `loc_c` antes del `groupby`.
- Los pesos de score se ajustan en `candidate_vendors`. La familiaridad suma +0.2.
- Para cotizaciones muy grandes, considera filtrar por fecha también en `load_quotations()` vía un `.gte()` en Supabase para reducir tráfico.
    """
)
