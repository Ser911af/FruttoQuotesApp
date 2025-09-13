# pages/5_salesmth.py
# 🧩 Sales Match — Vendor Recommender (UI simplificada + comparador precios + Reps con detalle)
# - UI en tabs: Resumen / Ofertas / Comparar / Reps / Actividad / Ayuda
# - Comparación directa: cotización vs venta (spread y %)
# - Reps: resumen + **detalle de deals** (customer, product, vendor, qty, buy/sell price), CSV export
# - SIN matplotlib (usamos column_config)
# - TZ Bogotá

import os
import re
import math
import datetime as dt
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st

# ✅ 1) Login obligatorio
from simple_auth import ensure_login, logout_button

user = ensure_login()
with st.sidebar:
    logout_button()

st.set_page_config(page_title="Sales Match — Vendor Recommender", page_icon="🧩", layout="wide")
st.caption(f"Sesión: {user}")

# (Opcional) Supabase
try:
    from supabase import create_client  # pip install supabase
except Exception:
    create_client = None

# ------------------------
# TÍTULO + RESUMEN
# ------------------------
st.title("🧩 Sales Match: Customer Offer Recommender")
st.caption(
    "Compará lo que **se cotiza hoy** con lo que **se vendió recientemente** para decidir qué ofrecer a cada cliente. "
    "Además, revisá spreads (cotización vs venta) y un resumen/DETALLE de los últimos deals por representante."
)

# ========================
# HELPERS: Normalización
# ========================
def _load_supabase_client(secret_key: str):
    sec = st.secrets.get(secret_key, None)
    if not sec or not create_client:
        return None
    url = sec.get("url"); key = sec.get("anon_key")
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
    return s in {"true", "t", "1", "yes", "y", "og"}  # OG -> True

def _normalize_txt(s: Optional[str]) -> str:
    if s is None: return ""
    s = str(s).lower()
    s = re.sub(r"[\s/_\-]+", " ", s)
    s = re.sub(r"[^\w\s\.\+]", "", s)
    return s.strip()

PRODUCT_SYNONYMS = {
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
    if not isinstance(s, str): return ""
    m = SIZE_PATTERN.search(s)
    return m.group(1).lower() if m else ""

def _canonical_product(txt: str) -> str:
    n = _normalize_txt(txt)
    if n in REV_SYNONYMS: return REV_SYNONYMS[n]
    for canon, variants in PRODUCT_SYNONYMS.items():
        if any(_normalize_txt(v) in n for v in variants):
            return canon
    return n.split(" ")[0] if n else ""

def _vendor_clean(s: str) -> str: return _normalize_txt(s)
def _loc_clean(s: str) -> str: return _normalize_txt(s)

# ========================
# LOADERS
# ========================
@st.cache_data(ttl=300, show_spinner=False)
def load_quotations() -> pd.DataFrame:
    sb = _load_supabase_client("supabase_quotes")
    if not sb:
        st.error("No se pudo crear cliente Supabase para cotizaciones (st.secrets['supabase_quotes']).")
        return pd.DataFrame()
    table_name = st.secrets.get("supabase_quotes", {}).get("table", "quotations")
    rows, limit, offset = [], 1000, 0
    while True:
        q = sb.table(table_name).select("*").range(offset, offset + limit - 1).execute()
        data = q.data or []
        rows.extend(data)
        if len(data) < limit: break
        offset += limit
    df = pd.DataFrame(rows)
    if df.empty: return df

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
            if c in df.columns: std[std_col] = df[c]; break
        if std_col not in std: std[std_col] = pd.Series([None]*len(df))

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
    return qdf[["date","is_organic","price","product_raw","product_canon","size_std","vendor_c","loc_c"]]

@st.cache_data(ttl=300, show_spinner=False)
def load_sales() -> pd.DataFrame:
    sb = _load_supabase_client("supabase_sales")
    if sb:
        table_name = st.secrets.get("supabase_sales", {}).get("table", "sales")
        rows, limit, offset = [], 1000, 0
        while True:
            q = sb.table(table_name).select("*").range(offset, offset + limit - 1).execute()
            data = q.data or []
            rows.extend(data)
            if len(data) < limit: break
            offset += limit
        df = pd.DataFrame(rows)
    else:
        ms = st.secrets.get("mssql_sales", None)
        if ms:
            st.warning("Loader MSSQL no implementado aquí por brevedad. Usa sqlalchemy/pyodbc y devolvé un DataFrame.")
            return pd.DataFrame()
        else:
            st.error("No se encontró fuente de ventas (st.secrets['supabase_sales'] o ['mssql_sales']).")
            return pd.DataFrame()

    if df.empty: return df

    alias_map = {
        "received_date": ["received_date","reqs_date","created_at","sale_date"],
        "product": ["product","commoditie","buyer_product"],
        "organic": ["organic","is_organic","OG/CV"],
        "unit": ["unit"],
        "customer": ["customer","client","buyer"],
        "vendor": ["vendor","shipper","supplier"],
        "sale_location": ["sale_location","lot_location"],
        "quantity": ["quantity","qty"],
        "price_per_unit": ["price_per_unit","price","unit_price","sell_price"],
        "sales_rep": ["sales_rep","rep","seller","account_exec","account_manager"],
        "cost_per_unit": ["cost_per_unit","buy_price","purchase_price","cost"],
    }
    std = {}
    for std_col, candidates in alias_map.items():
        for c in candidates:
            if c in df.columns: std[std_col] = df[c]; break
        if std_col not in std: std[std_col] = pd.Series([None]*len(df))

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
    sdf["sales_rep"] = sdf["sales_rep"].astype(str) if "sales_rep" in sdf.columns else None
    sdf["cost_per_unit"] = pd.to_numeric(sdf["cost_per_unit"], errors="coerce") if "cost_per_unit" in sdf.columns else np.nan

    return sdf[[
        "date","is_organic","product_raw","product_canon","size_std",
        "customer_c","vendor_c","loc_c","quantity","price_per_unit",
        "sales_rep","cost_per_unit"
    ]]

# ========================
# CORE LÓGICA
# ========================
def recent_purchases(sales: pd.DataFrame, days_window: int) -> pd.DataFrame:
    if sales.empty: return sales
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_window)
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    out = sales.loc[sales["date"] >= cutoff].copy()
    agg = (
        out.groupby(["customer_c","product_canon","is_organic","size_std","loc_c"], dropna=False)[["quantity","price_per_unit"]]
        .agg(quantity=("quantity","sum"), last_price=("price_per_unit","last"))
        .reset_index()
    )
    return agg

def candidate_vendors(quot: pd.DataFrame, prod_canon: str, is_og: bool, loc: str, size_hint: str) -> pd.DataFrame:
    if quot.empty: return quot
    df = quot.copy()
    df = df[df["product_canon"] == prod_canon]
    df = df[df["is_organic"] == bool(is_og)]
    df["loc_match"] = (df["loc_c"] == loc).astype(int)
    df["size_match"] = (df["size_std"] == _normalize_txt(size_hint)).astype(int) if size_hint else 0
    df["price_z"] = (df["price"] - df["price"].mean()) / (df["price"].std(ddof=0) + 1e-9)
    df["score"] = (-1.0 * df["price_z"]) + (0.5 * df["loc_match"]) + (0.3 * df["size_match"])
    return df

def add_familiarity_score(cands: pd.DataFrame, prior_vendors: List[str]) -> pd.DataFrame:
    if cands.empty: return cands
    pv = {_vendor_clean(v) for v in prior_vendors if isinstance(v, str)}
    c = cands.copy()
    c["familiar"] = c["vendor_c"].apply(lambda v: 1 if v in pv else 0)
    c["score"] += 0.2 * c["familiar"]
    return c

def build_recommendations(sales_recent: pd.DataFrame, sales_all: pd.DataFrame, quotations: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    recs = []
    if sales_recent.empty or quotations.empty:
        return pd.DataFrame(columns=["customer","product","is_organic","size","loc","vendor","price","score","why"])
    hist = (
        sales_all.groupby(["customer_c"])['vendor_c']
        .agg(lambda s: list(set([_vendor_clean(x) for x in s if isinstance(x, str)])))
        .to_dict()
    )
    for _, row in sales_recent.iterrows():
        cust = row["customer_c"]; prod = row["product_canon"]; is_og = bool(row["is_organic"])
        size = row.get("size_std",""); loc = row.get("loc_c","")
        cands = candidate_vendors(quotations, prod, is_og, loc, size)
        if cands.empty: continue
        prior = hist.get(cust, [])
        cands = add_familiarity_score(cands, prior)
        top = cands.sort_values(["score","price"], ascending=[False,True]).head(top_k).copy()
        for _, r in top.iterrows():
            why_bits = []
            if r["loc_match"] == 1: why_bits.append("ubicación coincide")
            if r["size_match"] == 1: why_bits.append("tamaño coincide")
            if r.get("familiar",0) == 1: why_bits.append("proveedor ya conocido")
            why = ", ".join(why_bits) if why_bits else "mejor relación precio/filtros"
            recs.append({
                "customer": cust, "product": prod, "is_organic": is_og,
                "size": size or "", "loc": loc or "", "vendor": r["vendor_c"],
                "price": float(r["price"]), "score": float(r["score"]), "why": why,
            })
    recdf = pd.DataFrame(recs)
    if recdf.empty: return recdf
    return recdf.sort_values(["customer","product","score","price"], ascending=[True,True,False,True])

def _bogota_today() -> dt.date:
    return (dt.datetime.utcnow() + dt.timedelta(hours=-5)).date()

def recent_vendor_stats(sales: pd.DataFrame, days_window: int) -> pd.DataFrame:
    if sales.empty:
        return pd.DataFrame(columns=["vendor_c","last_sale","n_customers","n_products","qty_total"])
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_window)
    s = sales.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.loc[s["date"] >= cutoff]
    if s.empty:
        return pd.DataFrame(columns=["vendor_c","last_sale","n_customers","n_products","qty_total"])
    grp = s.groupby("vendor_c")
    out = pd.DataFrame({
        "vendor_c": grp.size().index,
        "last_sale": grp["date"].max().values,
        "n_customers": grp["customer_c"].nunique().values,
        "n_products": grp["product_canon"].nunique().values,
        "qty_total": grp["quantity"].sum().values,
    })
    return out

# ====== Scoring de ofertas ======
def score_offer(row_q, row_hist, bench_price: Optional[float]) -> Tuple[float, str]:
    why = []; score = 0.0
    if pd.notnull(bench_price) and bench_price > 0 and pd.notnull(row_q["price"]):
        improvement = (bench_price - row_q["price"]) / bench_price
        score += 1.2 * improvement
        if improvement > 0: why.append(f"precio {improvement*100:.1f}% por debajo de su referencia")
    if row_q.get("loc_c","") == row_hist.get("loc_c",""): score += 0.3;  why.append("misma ubicación")
    if row_q.get("size_std","") and row_q.get("size_std","") == row_hist.get("size_std",""): score += 0.2;  why.append("mismo tamaño")
    if bool(row_q.get("is_organic",False)) == bool(row_hist.get("is_organic",False)): score += 0.2;  why.append("misma condición OG/CV")
    if not why: why.append("coincide en commodity y condiciones básicas")
    return score, ", ".join(why)

def build_customer_offers(qdf_day: pd.DataFrame, sales_recent: pd.DataFrame, sales_all: pd.DataFrame, bench_days: int = 30, top_k: int = 5) -> pd.DataFrame:
    if qdf_day.empty or sales_recent.empty:
        return pd.DataFrame(columns=["customer","product","organic","size_hist","loc_hist","vendor","price","score","why"])
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=bench_days)
    s = sales_all.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.loc[s["date"] >= cutoff]
    bench = (
        s.groupby(["customer_c","product_canon","is_organic","size_std","loc_c"])['price_per_unit']
        .median().rename('bench_price').reset_index()
    )
    offers = []
    for _, row in sales_recent.iterrows():
        cust = row["customer_c"]; prod = row["product_canon"]; is_og = bool(row["is_organic"])
        size_h = row.get("size_std",""); loc_h = row.get("loc_c","")
        subset = qdf_day[(qdf_day["product_canon"] == prod) & (qdf_day["is_organic"] == is_og)].copy()
        if subset.empty: continue
        b = bench[(bench["customer_c"]==cust) & (bench["product_canon"]==prod) & (bench["is_organic"]==is_og)]
        if size_h: b = b[b["size_std"]==size_h]
        if loc_h:  b = b[b["loc_c"]==loc_h]
        bench_price = b["bench_price"].iloc[0] if len(b)>0 else np.nan
        for _, q in subset.iterrows():
            score, why = score_offer(q, {"size_std": size_h, "loc_c": loc_h, "is_organic": is_og}, bench_price)
            offers.append({
                "customer": cust, "product": prod, "organic": "OG" if is_og else "CV",
                "size_hist": size_h, "loc_hist": loc_h, "vendor": q["vendor_c"],
                "price": float(q["price"]), "score": float(score), "why": why,
            })
    if not offers: return pd.DataFrame()
    out = pd.DataFrame(offers).sort_values(["customer","product","score","price"], ascending=[True,True,False,True])
    return out.groupby(["customer","product"], as_index=False).head(top_k)

# ========================
# COMPARADOR COTIZACIÓN VS VENTA (sin matplotlib)
# ========================
def match_quote_for_sale(qdf_all: pd.DataFrame, sale_row: pd.Series, day_tolerance: int = 2) -> Optional[pd.Series]:
    if qdf_all.empty: return None
    base = qdf_all[
        (qdf_all["product_canon"] == sale_row["product_canon"]) &
        (qdf_all["is_organic"] == bool(sale_row["is_organic"])) &
        (qdf_all["size_std"] == sale_row.get("size_std","")) &
        (qdf_all["loc_c"] == sale_row.get("loc_c",""))
    ].copy()
    if base.empty:
        base = qdf_all[
            (qdf_all["product_canon"] == sale_row["product_canon"]) &
            (qdf_all["is_organic"] == bool(sale_row["is_organic"])) &
            (qdf_all["loc_c"] == sale_row.get("loc_c",""))
        ].copy()
        if base.empty: return None
    sale_date = pd.to_datetime(sale_row["date"], errors="coerce")
    if pd.isna(sale_date): return base.sort_values("date", ascending=False).head(1).iloc[0]
    same_day = base[base["date"].dt.date == sale_date.date()]
    if not same_day.empty: return same_day.sort_values("price").head(1).iloc[0]
    base["day_diff"] = (base["date"] - sale_date).abs()
    base = base[base["day_diff"] <= pd.Timedelta(days=day_tolerance)]
    if base.empty: return None
    return base.sort_values(["day_diff","price"], ascending=[True,True]).head(1).iloc[0]

def build_quote_vs_sale(qdf_all: pd.DataFrame, sales: pd.DataFrame, days_window: int = 14) -> pd.DataFrame:
    if qdf_all.empty or sales.empty: return pd.DataFrame()
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_window)
    s = sales.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.loc[s["date"] >= cutoff].dropna(subset=["price_per_unit"])
    if s.empty: return pd.DataFrame()
    bench_cut = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=30)
    bench_src = sales.copy()
    bench_src["date"] = pd.to_datetime(bench_src["date"], errors="coerce")
    bench_src = bench_src.loc[bench_src["date"] >= bench_cut]
    bench = (
        bench_src.groupby(["customer_c","product_canon","is_organic","size_std","loc_c"])["price_per_unit"]
        .median().rename("bench_price").reset_index()
    )
    out_rows = []
    for _, row in s.iterrows():
        q = match_quote_for_sale(qdf_all, row, day_tolerance=2)
        if q is None or pd.isna(q.get("price", np.nan)): continue
        sell = float(row["price_per_unit"]); quote = float(q["price"])
        spread_abs = sell - quote
        spread_pct_vs_quote = (sell - quote) / quote if quote > 0 else np.nan
        b = bench[
            (bench["customer_c"] == row["customer_c"]) &
            (bench["product_canon"] == row["product_canon"]) &
            (bench["is_organic"] == bool(row["is_organic"])) &
            (bench["size_std"] == row.get("size_std","")) &
            (bench["loc_c"] == row.get("loc_c",""))
        ]
        bench_price = b["bench_price"].iloc[0] if len(b)>0 else np.nan
        mejora_vs_bench = sell - bench_price if pd.notna(bench_price) else np.nan
        out_rows.append({
            "date_sale": pd.to_datetime(row["date"]),
            "customer": row["customer_c"],
            "sales_rep": row.get("sales_rep", None),
            "product": row["product_canon"],
            "organic": "OG" if bool(row["is_organic"]) else "CV",
            "size": row.get("size_std",""),
            "loc": row.get("loc_c",""),
            "vendor_quote": q["vendor_c"],
            "quote_date": pd.to_datetime(q["date"]),
            "quote_price": quote,
            "sell_price": sell,
            "qty": row.get("quantity", np.nan),
            "spread_abs": spread_abs,
            "spread_pct_vs_quote": spread_pct_vs_quote,
            "bench_price": bench_price,
            "delta_vs_bench": mejora_vs_bench,
            "cost_per_unit": row.get("cost_per_unit", np.nan)
        })
    out = pd.DataFrame(out_rows)
    if out.empty: return out
    if "cost_per_unit" in out.columns and out["cost_per_unit"].notna().any():
        out["gross_margin_abs"] = out["sell_price"] - out["cost_per_unit"]
        out["gross_margin_pct"] = (out["sell_price"] - out["cost_per_unit"]) / out["sell_price"]
    return out.sort_values("date_sale", ascending=False)

# ========================
# HELPERS UI (sin matplotlib)
# ========================
def show_offers_table(df: pd.DataFrame):
    if df.empty:
        st.info("Sin ofertas para mostrar."); return
    min_s = float(df["score"].min()); max_s = float(df["score"].max())
    st.dataframe(
        df, hide_index=True, use_container_width=True,
        column_config={
            "price": st.column_config.NumberColumn("price", format="$%.2f"),
            "score": st.column_config.ProgressColumn(
                "score", help="Score vs benchmark + match condiciones.",
                min_value=min_s, max_value=max_s,
            ),
        },
    )

def show_compare_table(comp: pd.DataFrame):
    if comp.empty:
        st.info("No hay matches cotización↔venta en la ventana seleccionada."); return
    show_cols = [
        "date_sale","customer","sales_rep","product","organic","size","loc",
        "vendor_quote","quote_date","quote_price","sell_price","qty",
        "spread_abs","spread_pct_vs_quote","bench_price","delta_vs_bench"
    ]
    show_cols = [c for c in show_cols if c in comp.columns]
    st.dataframe(
        comp[show_cols], hide_index=True, use_container_width=True,
        column_config={
            "date_sale": st.column_config.DatetimeColumn("date_sale", format="YYYY-MM-DD HH:mm"),
            "quote_date": st.column_config.DatetimeColumn("quote_date", format="YYYY-MM-DD HH:mm"),
            "quote_price": st.column_config.NumberColumn("quote_price", format="$%.2f"),
            "sell_price": st.column_config.NumberColumn("sell_price", format="$%.2f"),
            "spread_abs": st.column_config.NumberColumn("spread_abs", format="$%.2f"),
            "spread_pct_vs_quote": st.column_config.NumberColumn("% vs quote", format="%.1f%%"),
            "bench_price": st.column_config.NumberColumn("bench_price", format="$%.2f"),
            "delta_vs_bench": st.column_config.NumberColumn("delta_vs_bench", format="$%.2f"),
            "qty": st.column_config.NumberColumn("qty", format="%.0f"),
        },
    )

# ========================
# CONTROLES (sidebar)
# ========================
with st.sidebar:
    st.subheader("🎛️ Filtros")
    days = st.slider("Sales lookback (days)", min_value=3, max_value=60, value=14, step=1, key="win_days")
    topk = st.slider("Top-K offers por cliente/producto", min_value=3, max_value=20, value=5, step=1, key="topk_recs")
    st.markdown("---")
    default_day = _bogota_today()
    selected_day = st.date_input("Fecha de cotizaciones (Daily Sheet)", value=default_day, key="ds_day")
    only_matches = st.checkbox("Sólo vendors con match de ventas recientes", value=True, key="only_matches_chk")

# ========================
# DATA
# ========================
qdf_all = load_quotations()
sdf = load_sales()

m1, m2 = st.columns(2)
with m1: st.metric("Cotizaciones cargadas", value=len(qdf_all))
with m2: st.metric("Registros de ventas", value=len(sdf))
if qdf_all.empty or sdf.empty: st.stop()

qdf_day = qdf_all[qdf_all["date"].dt.date == pd.to_datetime(selected_day).date()].copy()
if qdf_day.empty: st.warning("No hay cotizaciones para la fecha seleccionada.")
sdf_recent_cust = recent_purchases(sdf, days_window=days)

# ========================
# UI EN TABS
# ========================
tab_resumen, tab_ofertas, tab_compare, tab_reps, tab_activity, tab_help = st.tabs(
    ["📌 Resumen", "🎯 Ofertas (hoy)", "💱 Comparar precios", "🧑‍💼 Reps", "📈 Actividad", "❓ Ayuda"]
)

# ---------- RESUMEN ----------
with tab_resumen:
    st.subheader("Fotografía rápida")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Clientes con compras recientes", value=int(sdf_recent_cust["customer_c"].nunique()) if not sdf_recent_cust.empty else 0)
    with c2: st.metric("Productos activos (ventana)", value=int(sdf_recent_cust["product_canon"].nunique()) if not sdf_recent_cust.empty else 0)
    with c3: st.metric("Vendors en ventas recientes", value=int(sdf["vendor_c"].nunique()))
    with c4: st.metric("OG ratio (últimas ventas)", value=f"{100*float(sdf['is_organic'].mean()):.1f}%" if "is_organic" in sdf and len(sdf)>0 else "N/A")
    st.markdown("**Tip:** mové el slider de *lookback* para ajustar el universo reciente.")

# ---------- OFERTAS (hoy) ----------
with tab_ofertas:
    st.subheader("Customer offers para el Daily Sheet seleccionado")
    if qdf_day.empty or sdf_recent_cust.empty:
        st.info("Cargá cotizaciones del día y asegurate de tener ventas recientes en la ventana.")
    else:
        offers = build_customer_offers(qdf_day, sdf_recent_cust, sdf, bench_days=30, top_k=topk)
        if offers.empty:
            st.warning("No se encontraron ofertas relevantes usando las cotizaciones del día.")
        else:
            show_offers_table(offers)

    with st.expander("Vendors más recomendados (resumen)", expanded=False):
        if qdf_day.empty or sdf_recent_cust.empty:
            st.info("Sin datos.")
        else:
            if 'offers' in locals() and not offers.empty:
                recs_show = offers.copy()
                vend_counts = (
                    recs_show.groupby("vendor")
                    .agg(recs=("vendor","count"), avg_price=("price","mean"), avg_score=("score","mean"))
                    .reset_index()
                    .sort_values(["avg_score","recs"], ascending=[False,False])
                )
                st.dataframe(
                    vend_counts, hide_index=True, use_container_width=True,
                    column_config={
                        "avg_price": st.column_config.NumberColumn("avg_price", format="$%.2f"),
                        "avg_score": st.column_config.NumberColumn("avg_score", format="%.2f"),
                        "recs": st.column_config.NumberColumn("recs", format="%.0f"),
                    },
                )
            else:
                st.info("No hay recomendaciones para resumir.")

# ---------- COMPARAR PRECIOS ----------
with tab_compare:
    st.subheader("Cotización vs Venta (spread y %)")
    comp = build_quote_vs_sale(qdf_all, sdf, days_window=days)
    if comp.empty:
        st.info("No se pudo construir el comparador (falta match por producto/OG/size/loc o no hay ventas en ventana).")
    else:
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("Deals con match", value=len(comp))
        with k2: st.metric("Spread medio vs cotización", value=f"${comp['spread_abs'].mean():,.2f}")
        with k3: st.metric("% medio vs cotización", value=f"{100*comp['spread_pct_vs_quote'].mean():.1f}%")
        with k4:
            has_bench = comp["bench_price"].notna().any()
            st.metric("Con bench (30d)", value=int(comp["bench_price"].notna().sum()) if has_bench else 0)
        show_compare_table(comp)
        # Export
        csv = comp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar comparador (CSV)", data=csv, file_name="compare_quote_vs_sale.csv", mime="text/csv")

# ---------- REPS (Resumen + DETALLE) ----------
with tab_reps:
    st.subheader("Resumen y detalle de últimos deals por Sales Rep")

    if "sales_rep" not in sdf.columns or sdf["sales_rep"].isna().all():
        st.info("No hay columna de representante en las ventas (alias: sales_rep, rep, seller, account_exec, account_manager).")
    else:
        cut = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
        s = sdf.loc[pd.to_datetime(sdf["date"], errors="coerce") >= cut].copy()

        if s.empty:
            st.info("No hay ventas recientes en la ventana seleccionada.")
        else:
            # ---------- Resumen por rep ----------
            comp = build_quote_vs_sale(qdf_all, sdf, days_window=days)
            comp_rep = pd.DataFrame()
            if not comp.empty and "sales_rep" in comp.columns:
                comp_rep = comp.groupby("sales_rep").agg(
                    deals=("sales_rep","count"),
                    avg_spread=("spread_abs","mean"),
                    avg_spread_pct=("spread_pct_vs_quote","mean")
                ).reset_index()

            rep_summary = (
                s.groupby("sales_rep")
                .agg(
                    deals=("sales_rep","count"),
                    customers=("customer_c","nunique"),
                    products=("product_canon","nunique"),
                    qty_total=("quantity","sum"),
                    avg_price=("price_per_unit","mean")
                )
                .reset_index()
            )
            if not comp_rep.empty:
                rep_summary = rep_summary.merge(comp_rep, on="sales_rep", how="left")

            st.markdown("**Resumen por representante**")
            st.dataframe(
                rep_summary.fillna({"avg_spread":0, "avg_spread_pct":0})
                .sort_values("deals", ascending=False),
                hide_index=True, use_container_width=True,
                column_config={
                    "qty_total": st.column_config.NumberColumn("qty_total", format="%.0f"),
                    "avg_price": st.column_config.NumberColumn("avg_price", format="$%.2f"),
                    "avg_spread": st.column_config.NumberColumn("avg_spread", format="$%.2f"),
                    "avg_spread_pct": st.column_config.NumberColumn("avg_spread_pct", format="%.1f%%"),
                },
            )

            # ---------- DETALLE de registros ----------
            st.markdown("**Detalle de deals (registros)**")
            reps = ["(Todos)"] + sorted([r for r in s["sales_rep"].dropna().unique().tolist() if r])
            sel_rep = st.selectbox("Filtrar por Sales Rep", reps, index=0)

            detail = s.copy()
            if sel_rep != "(Todos)":
                detail = detail[detail["sales_rep"] == sel_rep]

            # Selección de columnas pedidas + extras útiles
            cols = [
                "date", "sales_rep", "customer_c", "product_canon",
                "vendor_c", "quantity", "cost_per_unit", "price_per_unit"
            ]
            cols = [c for c in cols if c in detail.columns]
            detail = detail[cols].sort_values("date", ascending=False)

            # Márgenes si hay costo
            if "cost_per_unit" in detail.columns and detail["cost_per_unit"].notna().any():
                detail["gross_margin_abs"] = detail["price_per_unit"] - detail["cost_per_unit"]
                detail["gross_margin_pct"] = (detail["price_per_unit"] - detail["cost_per_unit"]) / detail["price_per_unit"]

            st.dataframe(
                detail, hide_index=True, use_container_width=True,
                column_config={
                    "date": st.column_config.DatetimeColumn("date", format="YYYY-MM-DD HH:mm"),
                    "customer_c": st.column_config.TextColumn("customer"),
                    "product_canon": st.column_config.TextColumn("product"),
                    "vendor_c": st.column_config.TextColumn("vendor"),
                    "quantity": st.column_config.NumberColumn("quantity", format="%.0f"),
                    "cost_per_unit": st.column_config.NumberColumn("buy_price", format="$%.2f"),
                    "price_per_unit": st.column_config.NumberColumn("sell_price", format="$%.2f"),
                    "gross_margin_abs": st.column_config.NumberColumn("gross_margin_abs", format="$%.2f"),
                    "gross_margin_pct": st.column_config.NumberColumn("gross_margin_pct", format="%.1f%%"),
                },
            )

            # Export CSV del detalle
            csv_detail = detail.to_csv(index=False).encode("utf-8")
            fname = f"sales_rep_detail_{'all' if sel_rep=='(Todos)' else sel_rep}.csv".replace(" ", "_")
            st.download_button("⬇️ Descargar detalle (CSV)", data=csv_detail, file_name=fname, mime="text/csv")

# ---------- ACTIVIDAD ----------
with tab_activity:
    st.subheader("Compras recientes por cliente/producto")
    if sdf_recent_cust.empty:
        st.info("No hay ventas recientes en la ventana seleccionada.")
    else:
        customers = sorted(sdf_recent_cust["customer_c"].dropna().unique().tolist())
        sel_customers = st.multiselect("Filtrar clientes", customers, default=customers[: min(10, len(customers))], key="customers_main")
        subset_recent = sdf_recent_cust[sdf_recent_cust["customer_c"].isin(sel_customers)].copy()
        subset_recent = subset_recent.rename(columns={
            "customer_c":"customer","product_canon":"product","is_organic":"organic",
            "size_std":"size","loc_c":"location","quantity":"qty"
        })
        st.dataframe(subset_recent, hide_index=True, use_container_width=True)

    with st.expander("Recomendaciones (limitadas a Daily Sheet del día)", expanded=False):
        if qdf_day.empty or sdf_recent_cust.empty:
            st.info("Cargá cotizaciones del día y ventas recientes.")
        else:
            recs = build_recommendations(sdf_recent_cust, sdf, qdf_day, top_k=topk)
            if recs.empty:
                st.warning("No hay candidatos de vendors que cumplan con los filtros.")
            else:
                recs_show = recs.copy()
                recs_show["organic"] = recs_show["is_organic"].map({True:"OG", False:"CV"})
                recs_show = recs_show.drop(columns=["is_organic"], errors="ignore")
                st.dataframe(
                    recs_show[["customer","product","organic","size","loc","vendor","price","score","why"]],
                    hide_index=True, use_container_width=True
                )

# ---------- AYUDA ----------
with tab_help:
    st.subheader("¿Cómo usar esta página?")
    st.markdown("""
**Flujo (3 pasos):**
1) Elegí la fecha de **cotización** (Daily Sheet) en la barra lateral.  
2) Ajustá la ventana de **ventas recientes** (*Sales lookback*).  
3) Revisá **Ofertas (hoy)** y **Comparar precios**. En **Reps**, tenés **resumen y el detalle** exportable.

**Glosario:**
- **product_canon**: commodity normalizado.
- **OG/CV**: orgánico vs convencional.
- **size**: patrón de tamaño/pack detectado automáticamente.
- **loc**: mercado/ubicación normalizado.
- **spread vs cotización**: `sell_price - quote_price` (positivo: vendimos por encima de la cotización).
- **bench 30d**: mediana del precio pagado por ese cliente en ~30 días para ese commodity/condiciones.
""")
    st.info("Si querés, agrego filtros por commodity/OG/loc también en Reps → Detalle.")
