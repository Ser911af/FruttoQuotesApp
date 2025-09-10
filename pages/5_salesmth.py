# pages/3_Sales_Match.py
# Sales–Quotes MATCH (read-only): empareja cotizaciones del día vs historial de ventas
# - No modifica DB (solo SELECTs)
# - Empareja por product exacto + OG/CV, con señal de ubicación

import os
import math
import datetime as dt
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Config & Página ----------
st.set_page_config(page_title="Sales Match (Read-only)", page_icon="🧩", layout="wide")
st.title("🧩 Sales Match — Read-only (quotations × ventas)")
st.caption("Empareja **cotizaciones del día** con el historial de **ventas** por cliente usando `product` exacto (talla incluida).")

# ---------- Dependencias opcionales ----------
try:
    from supabase import create_client  # pip install supabase
except Exception:
    create_client = None

# ---------- Helpers de normalización (solo Python, sin tocar DB) ----------
def _norm_txt(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = " ".join(s.split())
    return s

def _norm_loc(s: Optional[str]) -> str:
    # igual a _norm_txt pero quitando comas
    return _norm_txt(str(s).replace(",", "") if s is not None else s)

def _as_organic(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = _norm_txt(str(x))
    return s in {"og", "organic", "true", "t", "1", "yes", "y"}

def _money_to_float(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x)
    # quita símbolos, comas
    s = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
    try:
        return float(s) if s != "" else None
    except Exception:
        return None

def _bogota_today() -> dt.date:
    return (dt.datetime.utcnow() + dt.timedelta(hours=-5)).date()

# ---------- Clientes Supabase (read-only) ----------
def _sb_client(secret_key: str):
    if not create_client:
        return None
    sec = st.secrets.get(secret_key, {})
    url, key = sec.get("url"), sec.get("anon_key")
    if not url or not key:
        return None
    return create_client(url, key)

# ---------- Loaders (solo SELECT) ----------
@st.cache_data(ttl=300, show_spinner=False)
def load_quotations_by_date(selected_date: dt.date) -> pd.DataFrame:
    """
    Lee tabla quotations (ya existente) filtrando por fecha en ctization_date.
    No escribe nada. Devuelve columnas normalizadas: product,is_organic,price,loc_c,vendor_c.
    """
    sb = _sb_client("supabase_quotes")
    if not sb:
        st.error("No se pudo crear cliente Supabase para cotizaciones (st.secrets['supabase_quotes']).")
        return pd.DataFrame()

    table = "quotations"  # nombre real según el usuario
    # Filtra por ctization_date == selected_date
    try:
        # Si la columna es texto con formato M/D/YYYY, podemos traer el día completo y filtrar en pandas.
        # Primero intentamos filtrar en Supabase si guarda como date exacta:
        resp = sb.table(table).select("*").eq("ctization_date", str(selected_date)).execute()
        data = resp.data or []
        df = pd.DataFrame(data)
        if df.empty:
            # fallback: trae del día +/- y filtra en pandas (por si ctization_date es texto)
            resp2 = sb.table(table).select("*").execute()
            df = pd.DataFrame(resp2.data or [])
    except Exception:
        # fallback sin filtro remoto
        resp2 = sb.table(table).select("*").execute()
        df = pd.DataFrame(resp2.data or [])

    if df.empty:
        return df

    # Asegura parseo de fecha y filtra por el día exacto
    if "ctization_date" in df.columns:
        df["_q_date"] = pd.to_datetime(df["ctization_date"], errors="coerce")
    else:
        # alias por si acaso
        for cand in ["cotization_date", "date", "Date", "request_date", "Request Date"]:
            if cand in df.columns:
                df["_q_date"] = pd.to_datetime(df[cand], errors="coerce")
                break
    if "_q_date" in df.columns:
        df = df[df["_q_date"].dt.date == pd.to_datetime(selected_date).date()].copy()

    # Alias mínimos (no escribimos a DB)
    def pick(*cols):
        for c in cols:
            if c in df.columns:
                return df[c]
        return pd.Series([None] * len(df))

    q = pd.DataFrame({
        "product":        pick("product", "Product").astype(str).map(_norm_txt),
        "is_organic":     pick("organic", "OG/CV", "og_cv").map(_as_organic),
        "price":          pick("price", "Price", "Price$").map(_money_to_float),
        "loc_c":          pick("Where", "where", "location").astype(str).map(_norm_loc),
        "vendor_c":       pick("Shipper", "Vendor", "vendor", "vendorclean").astype(str).map(_norm_txt),
    })
    q = q.dropna(subset=["product", "price"])
    return q


@st.cache_data(ttl=300, show_spinner=False)
def load_sales_readonly() -> pd.DataFrame:
    """
    Lee ventas desde Supabase (ventas_frutto).
    Si no hay credenciales, devuelve DF vacío (app sigue funcionando sin escribir nada).
    """
    sb = _sb_client("supabase_sales")
    if not sb:
        st.warning("No hay cliente Supabase para ventas (st.secrets['supabase_sales']). Configura o usa otro loader.")
        return pd.DataFrame()

    table = "ventas_frutto"
    rows, limit, offset = [], 2000, 0
    while True:
        resp = sb.table(table).select(
            "reqs_date, product, organic, sale_location, lot_location, customer, vendor, quantity, price_per_unit"
        ).range(offset, offset + limit - 1).execute()
        data = resp.data or []
        rows.extend(data)
        if len(data) < limit:
            break
        offset += limit

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalización local (no escribimos a DB)
    df["date"]          = pd.to_datetime(df["reqs_date"], errors="coerce")
    df["product"]       = df["product"].astype(str).map(_norm_txt)
    df["is_organic"]    = df["organic"].map(_as_organic)
    df["loc_c"]         = df["sale_location"].astype(str).map(_norm_loc) if "sale_location" in df.columns else ""
    if "lot_location" in df.columns:
        # usa sale_location si existe; si está vacío, usa lot_location
        df["loc_c"] = np.where(df["loc_c"].astype(str).str.len() > 0,
                               df["loc_c"],
                               df["lot_location"].astype(str).map(_norm_loc))
    df["customer_c"]    = df["customer"].astype(str).map(_norm_txt) if "customer" in df.columns else ""
    df["vendor_c"]      = df["vendor"].astype(str).map(_norm_txt) if "vendor" in df.columns else ""
    df["quantity"]      = pd.to_numeric(df["quantity"], errors="coerce")
    df["price_per_unit"]= pd.to_numeric(df["price_per_unit"], errors="coerce")

    keep = ["date","product","is_organic","loc_c","customer_c","vendor_c","quantity","price_per_unit"]
    return df[keep]


# ---------- Core analytics (solo pandas) ----------
def build_benchmark(sales: pd.DataFrame, bench_days: int = 30) -> pd.DataFrame:
    if sales.empty:
        return pd.DataFrame(columns=["customer_c","product","is_organic","bench_price"])
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=bench_days)
    s = sales.loc[pd.to_datetime(sales["date"], errors="coerce") >= cutoff].copy()
    if s.empty:
        return pd.DataFrame(columns=["customer_c","product","is_organic","bench_price"])
    bench = (
        s.groupby(["customer_c","product","is_organic"])["price_per_unit"]
         .median()
         .reset_index()
         .rename(columns={"price_per_unit":"bench_price"})
    )
    return bench

def recent_buys(sales: pd.DataFrame, recent_days: int = 14) -> pd.DataFrame:
    if sales.empty:
        return pd.DataFrame(columns=["customer_c","product","is_organic","qty_recent","last_buy_date","loc_any"])
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=recent_days)
    s = sales.loc[pd.to_datetime(sales["date"], errors="coerce") >= cutoff].copy()
    if s.empty:
        return pd.DataFrame(columns=["customer_c","product","is_organic","qty_recent","last_buy_date","loc_any"])
    agg = (
        s.groupby(["customer_c","product","is_organic"], dropna=False)
         .agg(qty_recent=("quantity","sum"),
              last_buy_date=("date","max"))
         .reset_index()
    )
    # catastro de ubicaciones históricas para “match suave”
    locs = s.groupby(["customer_c","product","is_organic"])["loc_c"].agg(lambda x: list(sorted(set([_norm_loc(v) for v in x if isinstance(v,str)])))).reset_index().rename(columns={"loc_c":"loc_list"})
    out = agg.merge(locs, on=["customer_c","product","is_organic"], how="left")
    return out

def _loc_soft_match(q_loc: str, loc_list: list) -> float:
    if not isinstance(q_loc, str) or not loc_list:
        return 0.0
    q = _norm_loc(q_loc)
    for L in loc_list:
        if q == L:
            return 1.0
        if (q in L) or (L in q):
            return 0.5
    return 0.0

def match_quotes_with_offers(quotes_day: pd.DataFrame,
                             sales: pd.DataFrame,
                             recent_days: int = 14,
                             bench_days: int = 30,
                             only_active_vendors: bool = False,
                             topk_per_pair: int = 5) -> pd.DataFrame:
    if quotes_day.empty or sales.empty:
        return pd.DataFrame()

    # Vendors activos (últimos 30 días)
    if only_active_vendors:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=30)
        active_vendors = (sales.loc[sales["date"] >= cutoff, "vendor_c"]
                               .dropna().unique().tolist())
        quotes_day = quotes_day[quotes_day["vendor_c"].isin(active_vendors)].copy()

    rb = recent_buys(sales, recent_days=recent_days)   # por cliente+producto
    bench = build_benchmark(sales, bench_days=bench_days)

    # Join por product + is_organic (exacto, como es tu clave operativa)
    # Luego para cada (cliente,producto) comparamos contra TODAS las quotes de ese producto en el día
    # y calculamos score por ubicación y mejora de precio vs bench.
    # Primero pre-join por (product,is_organic) para reducir combinaciones:
    pre = quotes_day.merge(
        rb[["customer_c","product","is_organic","loc_list"]],
        on=["product","is_organic"],
        how="inner"
    )

    if pre.empty:
        return pd.DataFrame()

    out_rows = []
    # Mapa bench para lookup rápido
    bench_map = bench.set_index(["customer_c","product","is_organic"])["bench_price"].to_dict()

    for idx, row in pre.iterrows():
        cust = row["customer_c"]
        prod = row["product"]
        isog = bool(row["is_organic"])
        q_price = row["price"]
        q_vendor = row["vendor_c"]
        q_loc   = row["loc_c"]
        loc_list= row.get("loc_list", [])

        bench_price = bench_map.get((cust, prod, isog), np.nan)

        # Señales
        loc_score = _loc_soft_match(q_loc, loc_list)   # 0.0, 0.5, 1.0
        if pd.notnull(bench_price) and bench_price > 0 and pd.notnull(q_price):
            improvement = (bench_price - q_price) / bench_price
        else:
            improvement = np.nan

        score = (0 if np.isnan(improvement) else improvement) + (0.3 if loc_score > 0 else 0)

        out_rows.append({
            "customer": cust,
            "product": prod,
            "organic": "OG" if isog else "CV",
            "vendor": q_vendor,
            "quote_loc": q_loc,
            "quote_price": q_price,
            "bench_price": None if np.isnan(bench_price) else float(bench_price),
            "price_improvement": None if np.isnan(improvement) else float(improvement),
            "score": float(score),
        })

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    # Orden y recorte por top-K por (customer,product)
    out = out.sort_values(["customer","product","score","quote_price"], ascending=[True, True, False, True])
    out = out.groupby(["customer","product"], as_index=False).head(topk_per_pair)
    return out

# ---------- UI ----------
with st.sidebar:
    st.subheader("Parámetros")
    default_day = _bogota_today()
    q_day = st.date_input("Fecha de cotizaciones (ctization_date)", value=default_day)
    recent_days = st.slider("Ventana 'ventas recientes' (días)", 7, 60, 14)
    bench_days  = st.slider("Ventana benchmark (días)", 14, 90, 30)
    only_active = st.checkbox("Solo vendors activos (30d)", value=False)
    topk = st.slider("Top-K ofertas por cliente/producto", 3, 20, 5)

# ---------- Pipeline ----------
qdf = load_quotations_by_date(q_day)
sdf = load_sales_readonly()

m1, m2, m3 = st.columns(3)
with m1: st.metric("Cotizaciones (día)", len(qdf))
with m2: st.metric("Ventas (filas)", len(sdf))
with m3:
    custs = sdf["customer_c"].nunique() if not sdf.empty else 0
    st.metric("Clientes con ventas", custs)

if qdf.empty:
    st.warning("No se encontraron cotizaciones para la fecha seleccionada en 'quotations.ctization_date'.")
    st.stop()
if sdf.empty:
    st.warning("No hay ventas cargadas (ventas_frutto). Configura st.secrets['supabase_sales'] para lectura.")
    st.stop()

offers = match_quotes_with_offers(
    quotes_day=qdf,
    sales=sdf,
    recent_days=recent_days,
    bench_days=bench_days,
    only_active_vendors=only_active,
    topk_per_pair=topk
)

st.subheader("Ofertas sugeridas (solo lectura)")
if offers.empty:
    st.info("Sin empates para los parámetros actuales.")
else:
    st.dataframe(
        offers.style.format({
            "quote_price": "${:,.2f}",
            "bench_price": "${:,.2f}",
            "price_improvement": "{:.1%}",
            "score": "{:.3f}",
        }),
        use_container_width=True
    )
    csv = offers.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", csv, file_name=f"offers_{q_day}.csv", mime="text/csv")

st.caption("Empareje por `product` exacto + `OG/CV`. Ubicación con match suave (igualdad o inclusión). Todo en memoria — sin escribir a DB.")
