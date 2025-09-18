# pages/3_Customer_Retention.py
# Customer Retention Rankings — by Company and by Sales Rep
# Mantiene: Login + acceso a DB de ventas (Supabase/Azure) con el esquema `ventas_frutto`

import os
import math
import re
import datetime as dt
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st

# ✅ 1) Login obligatorio antes de cargar nada pesado
from simple_auth import ensure_login, logout_button

user = ensure_login()   # Si no hay sesión, este call bloquea la página (st.stop)
with st.sidebar:
    logout_button()

st.caption(f"Sesión: {user}")

# --- Opcional: Altair para gráficos (no crítico)
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

# --- Opcional: Supabase
try:
    from supabase import create_client  # pip install supabase
except Exception:
    create_client = None

# ------------------------
# CONFIG & PAGE
# ------------------------
st.set_page_config(page_title="Customer Retention Rankings", page_icon="📈", layout="wide")
st.title("📈 Customer Retention Rankings")
st.caption("Rankings de retención por Cliente y por Sales Rep, con métricas RFM + regularidad de compra.")

# ------------------------
# HELPERS (normas/limpieza)
# ------------------------
def _normalize_txt(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _coerce_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return False
    s = str(x).strip().lower()
    return s in {"true", "t", "1", "yes", "y", "og"}

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

# ------------------------
# LOADER VENTAS (ventas_frutto)
# ------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_sales() -> pd.DataFrame:
    """
    Lee ventas desde Supabase (preferente) o ganchos MSSQL (Azure).
    Normaliza a columnas snake_case del esquema `ventas_frutto`.
    """
    # 1) Supabase
    sb = _load_supabase_client("supabase_sales")
    if sb:
        table_name = st.secrets.get("supabase_sales", {}).get("table", "ventas_frutto")
        rows, limit, offset = [], 2000, 0
        while True:
            q = sb.table(table_name).select("*").range(offset, offset + limit - 1).execute()
            data = q.data or []
            rows.extend(data)
            if len(data) < limit:
                break
            offset += limit
        df = pd.DataFrame(rows)
    else:
        # 2) Azure SQL (opcional)
        ms = st.secrets.get("mssql_sales", None)
        if ms:
            st.warning("Loader MSSQL no implementado aquí por brevedad. Usa sqlalchemy/pyodbc y devuélveme un DataFrame.")
            return pd.DataFrame()
        else:
            st.error("No se encontró fuente de ventas (st.secrets['supabase_sales'] o ['mssql_sales']).")
            return pd.DataFrame()

    if df.empty:
        return df

    # Alias → estándar (según CREATE TABLE ventas_frutto)
    alias_map = {
        "reqs_date": ["reqs_date"],
        "most_recent_invoice_paid_date": ["most_recent_invoice_paid_date"],
        "received_date": ["received_date"],
        "use_by_date": ["use_by_date"],
        "pack_date": ["pack_date"],
        "id_hash": ["id_hash"],
        "product": ["product", "buyer_product", "commoditie"],
        "organic": ["organic"],
        "unit": ["unit"],
        "label": ["label"],
        "coo": ["coo"],
        "sales_order": ["sales_order"],
        "invoice_num": ["invoice_num", "invoice #", "invoice"],
        "invoice_payment_status": ["invoice_payment_status"],
        "sale_location": ["sale_location", "lot_location"],
        "sales_rep": ["sales_rep", "cus_sales_rep"],
        "customer": ["customer"],
        "lot": ["lot"],
        "vendor": ["vendor", "shipper", "supplier"],
        "source": ["source"],
        "lot_location": ["lot_location"],
        "oficina": ["oficina"],
        "number_market": ["number_market"],
        "quantity": ["quantity", "qty"],
        "cost_per_unit": ["cost_per_unit"],
        "price_per_unit": ["price_per_unit", "sell_price", "unit_price", "price"],
        "total_cost": ["total_cost"],
        "total_sold_lot_expenses": ["total_sold_lot_expenses"],
        "total_revenue": ["total_revenue"],
        "total_profit_usd": ["total_profit_usd", "total_profit_$"],
        "total_profit_pct": ["total_profit_pct", "total_profit_%"],
        "created_at": ["created_at"],
    }

    std = {}
    for std_col, candidates in alias_map.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        std[std_col] = df[found] if found else pd.Series([None]*len(df))

    sdf = pd.DataFrame(std)
    # Casts principales
    for c in ["reqs_date","most_recent_invoice_paid_date","received_date","use_by_date","pack_date","created_at"]:
        sdf[c] = pd.to_datetime(sdf[c], errors="coerce")
    for c in ["quantity","cost_per_unit","price_per_unit","total_cost","total_sold_lot_expenses","total_revenue","total_profit_usd","total_profit_pct"]:
        sdf[c] = pd.to_numeric(sdf[c], errors="coerce")

    # Normalizaciones útiles
    sdf["customer_c"] = sdf["customer"].astype(str).map(_normalize_txt)
    sdf["sales_rep_c"] = sdf["sales_rep"].astype(str).map(_normalize_txt)
    sdf["sale_location_c"] = sdf["sale_location"].astype(str).map(_normalize_txt)
    sdf["oficina_c"] = sdf["oficina"].astype(str).map(_normalize_txt)
    sdf["is_organic"] = sdf["organic"].map(_coerce_bool)

    # Fecha operativa base (prioridad recibida > reqs_date)
    sdf["date"] = sdf["received_date"].fillna(sdf["reqs_date"])
    sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")

    # Identificador de orden (si existe)
    sdf["order_id"] = sdf["invoice_num"].astype(str)
    sdf.loc[sdf["order_id"].isin(["nan","none",""]), "order_id"] = np.nan

    return sdf

# ------------------------
# MÉTRICAS DE RETENCIÓN
# ------------------------
def _week_key(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    # Año-Week ISO (más estable para "regularidad")
    iso = ts.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"

def make_retention_metrics(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Calcula RFM + regularidad por cliente dentro del rango [start, end].
    Regularidad = semanas activas / semanas posibles (entre min y max en el rango).
    """
    if df.empty:
        return pd.DataFrame()

    d = df[(df["date"] >= start) & (df["date"] < end)].copy()
    if d.empty:
        return pd.DataFrame()

    # Weeks por fila
    d["week_key"] = d["date"].apply(_week_key)

    # Órdenes por invoice (si no hay invoice_num, cuenta filas como proxy)
    if d["order_id"].notna().any():
        orders = d.groupby(["customer_c","order_id"], dropna=False).agg(
            order_revenue=("total_revenue", "sum")
        ).reset_index()
        orders_per_customer = orders.groupby("customer_c").agg(
            n_orders=("order_id","nunique"),
            aov=("order_revenue","mean")
        )
    else:
        # Fallback
        orders_per_customer = d.groupby("customer_c").agg(
            n_orders=("date","count"),
            aov=("total_revenue","mean")
        )

    agg = d.groupby("customer_c").agg(
        total_revenue=("total_revenue","sum"),
        total_qty=("quantity","sum"),
        last_sale=("date","max"),
        first_sale=("date","min"),
        active_weeks=("week_key","nunique")
    )

    # Semanas posibles (entre primera y última venta en el rango)
    # Si solo una semana activa, span_weeks = 1 (evita división por cero)
    span_weeks = []
    for cust, row in agg.iterrows():
        f = row["first_sale"]
        l = row["last_sale"]
        if pd.isna(f) or pd.isna(l):
            span_weeks.append(1)
        else:
            # ISO week span aproximado: días/7 + 1
            days = max(0, (l.normalize() - f.normalize()).days)
            span_weeks.append(max(1, (days // 7) + 1))
    agg["weeks_span"] = span_weeks

    # Regularidad como densidad de semanas activas
    agg["regularity_ratio"] = (agg["active_weeks"] / agg["weeks_span"]).clip(upper=1.0)

    # Recency (días desde última compra)
    today = pd.Timestamp.now(tz="America/Bogota").tz_localize(None)
    agg["recency_days"] = (today - agg["last_sale"]).dt.days

    # Une órdenes/AOV
    agg = agg.join(orders_per_customer, how="left")
    agg["aov"] = agg["aov"].fillna(0)
    agg["n_orders"] = agg["n_orders"].fillna(0)

    # RFM-like score (z-scores con winsorization ligera)
    def z(x):
        x_ = x.clip(lower=x.quantile(0.02), upper=x.quantile(0.98))
        mu, sd = x_.mean(), x_.std(ddof=0)
        return (x - mu) / (sd + 1e-9)

    agg["z_rev"] = z(agg["total_revenue"].fillna(0))
    agg["z_freq"] = z(agg["n_orders"].fillna(0))
    agg["z_recency"] = z((-agg["recency_days"].fillna(365)))  # recency menor = mejor
    agg["z_regular"] = z(agg["regularity_ratio"].fillna(0))

    # Score compuesto ajustable
    # Pesos: Revenue 0.35, Frequency 0.25, Recency 0.25, Regularity 0.15
    agg["retention_score"] = (
        0.35*agg["z_rev"] +
        0.25*agg["z_freq"] +
        0.25*agg["z_recency"] +
        0.15*agg["z_regular"]
    )

    # Reset index y orden
    agg = agg.reset_index().rename(columns={"customer_c":"customer"})
    cols = [
        "customer","total_revenue","n_orders","aov","total_qty",
        "first_sale","last_sale","recency_days","active_weeks","weeks_span","regularity_ratio",
        "retention_score"
    ]
    agg = agg[cols].sort_values(["retention_score","total_revenue"], ascending=[False, False])

    return agg

def make_salesrep_rankings(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Ranking por Sales Rep: clientes activos, revenue, regularidad y score promedio ponderado.
    """
    if df.empty:
        return pd.DataFrame()
    d = df[(df["date"] >= start) & (df["date"] < end)].copy()
    if d.empty:
        return pd.DataFrame()

    # Métricas a nivel cliente primero (para ponderar)
    cust = make_retention_metrics(df, start, end)
    if cust.empty:
        return pd.DataFrame()

    # Mapear rep de cada venta al cliente (usa el rep "más frecuente" del rango)
    rep_map = (d.groupby(["customer_c","sales_rep_c"])
                 .size().reset_index(name="cnt"))
    top_rep_per_customer = rep_map.sort_values(["customer_c","cnt"], ascending=[True,False]) \
                                  .drop_duplicates("customer_c") \
                                  .rename(columns={"sales_rep_c":"sales_rep"})
    cust = cust.merge(top_rep_per_customer[["customer_c","sales_rep"]].rename(columns={"customer_c":"customer"}),
                      on="customer", how="left")

    # Agregado por rep
    rep = (d.groupby("sales_rep_c")
             .agg(
                 total_revenue=("total_revenue","sum"),
                 total_qty=("quantity","sum"),
                 active_customers=("customer_c","nunique")
             ).reset_index().rename(columns={"sales_rep_c":"sales_rep"}))

    # Promedios de retención de sus clientes (ponderado por revenue)
    tmp = cust.copy()
    tmp["rev_weight"] = tmp["total_revenue"].clip(lower=0)
    grp = tmp.groupby("sales_rep").apply(
        lambda g: pd.Series({
            "avg_retention_score": np.average(g["retention_score"], weights=(g["rev_weight"]+1e-9)),
            "avg_regularity": np.average(g["regularity_ratio"], weights=(g["rev_weight"]+1e-9)),
            "avg_recency_days": np.average(g["recency_days"], weights=(g["rev_weight"]+1e-9)),
            "customers": g["customer"].nunique()
        })
    ).reset_index()

    rep = rep.merge(grp, on="sales_rep", how="left")
    rep["avg_retention_score"] = rep["avg_retention_score"].fillna(0)
    rep["rank_score"] = (
        0.5 * (rep["avg_retention_score"]) +
        0.3 * (rep["total_revenue"].rank(pct=True, method="average")) +
        0.2 * (rep["active_customers"].rank(pct=True, method="average"))
    )

    rep = rep.sort_values(["rank_score","total_revenue"], ascending=[False, False])
    return rep

def customers_by_rep(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, rep: str) -> pd.DataFrame:
    """Ranking de clientes para un Sales Rep concreto."""
    if not rep:
        return pd.DataFrame()
    cust = make_retention_metrics(df, start, end)
    if cust.empty:
        return pd.DataFrame()

    # Rep dominante por cliente (en el rango)
    d = df[(df["date"] >= start) & (df["date"] < end)].copy()
    rep_map = (d.groupby(["customer_c","sales_rep_c"])
                 .size().reset_index(name="cnt"))
    top_rep_per_customer = rep_map.sort_values(["customer_c","cnt"], ascending=[True,False]) \
                                  .drop_duplicates("customer_c") \
                                  .rename(columns={"sales_rep_c":"sales_rep"})
    out = cust.merge(top_rep_per_customer.rename(columns={"customer_c":"customer"}),
                     on="customer", how="left")
    out = out[out["sales_rep"] == _normalize_txt(rep)]
    return out.sort_values(["retention_score","total_revenue"], ascending=[False, False])

# ------------------------
# UI CONTROLS
# ------------------------
with st.sidebar:
    st.subheader("Filtros")

    # Rango de fechas (default: últimos 90 días)
    today_bo = pd.Timestamp.now(tz="America/Bogota").normalize()
    default_start = (today_bo - pd.Timedelta(days=90)).tz_localize(None)
    default_end = (today_bo + pd.Timedelta(days=1)).tz_localize(None)

    date_range = st.date_input("Rango de fechas", value=(default_start, default_end))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)  # end exclusivo
    else:
        start_date, end_date = default_start, default_end

    oficina_filter = st.text_input("Oficina (contiene):", "")
    location_filter = st.text_input("Sale Location (contiene):", "")
    organic_only = st.checkbox("Solo Orgánico (OG)", value=False)

# ------------------------
# DATA
# ------------------------
sdf = load_sales()
col1, col2 = st.columns(2)
with col1:
    st.metric("Registros de ventas", value=len(sdf))
with col2:
    st.metric("Clientes únicos", value=int(sdf["customer_c"].nunique()) if not sdf.empty else 0)

if sdf.empty:
    st.stop()

# Filtros
mask = (sdf["date"] >= start_date) & (sdf["date"] < end_date)
if oficina_filter:
    mask &= sdf["oficina_c"].str.contains(_normalize_txt(oficina_filter))
if location_filter:
    mask &= sdf["sale_location_c"].str.contains(_normalize_txt(location_filter))
if organic_only:
    mask &= (sdf["is_organic"] == True)

sdf_f = sdf[mask].copy()
st.caption(f"Filtrado: {len(sdf_f)} filas en rango seleccionado.")

if sdf_f.empty:
    st.warning("No hay datos en el rango/filtros seleccionados.")
    st.stop()

# ------------------------
# RANKING: CUSTOMERS
# ------------------------
st.subheader("🏢 Customer Retention Rankings (por Compañía)")
cust_rank = make_retention_metrics(sdf, start_date, end_date)
if cust_rank.empty:
    st.info("Sin datos suficientes para calcular retención por cliente.")
else:
    show_cols = ["customer","total_revenue","n_orders","aov","total_qty","last_sale","recency_days","active_weeks","weeks_span","regularity_ratio","retention_score"]
    st.dataframe(cust_rank[show_cols], use_container_width=True)

    if ALTAIR_OK:
        chart = alt.Chart(cust_rank.head(25)).mark_bar().encode(
            x=alt.X("retention_score:Q", title="Retention Score"),
            y=alt.Y("customer:N", sort="-x", title="Cliente"),
            tooltip=["customer","total_revenue","n_orders","recency_days","regularity_ratio","retention_score"]
        ).properties(height=500)
        st.altair_chart(chart, use_container_width=True)

# ------------------------
# RANKING: SALES REPS
# ------------------------
st.subheader("🧑‍💼 Sales Rep Retention Rankings")
rep_rank = make_salesrep_rankings(sdf, start_date, end_date)
if rep_rank.empty:
    st.info("Sin datos suficientes para calcular ranking por Sales Rep.")
else:
    show_cols_rep = ["sales_rep","active_customers","total_revenue","avg_retention_score","avg_regularity","avg_recency_days","rank_score"]
    st.dataframe(rep_rank[show_cols_rep], use_container_width=True)

    if ALTAIR_OK:
        chart2 = alt.Chart(rep_rank.head(20)).mark_bar().encode(
            x=alt.X("rank_score:Q", title="Rank Score"),
            y=alt.Y("sales_rep:N", sort="-x", title="Sales Rep"),
            tooltip=["sales_rep","active_customers","total_revenue","avg_retention_score","avg_regularity","avg_recency_days","rank_score"]
        ).properties(height=420)
        st.altair_chart(chart2, use_container_width=True)

# ------------------------
# CUSTOMERS BY SALES REP
# ------------------------
st.subheader("🔎 Customers by Sales Rep")
all_reps = sorted([r for r in sdf["sales_rep_c"].dropna().unique().tolist() if r])
rep_sel = st.selectbox("Selecciona un Sales Rep", options=[""] + all_reps, index=0, placeholder="Elegir...")

if rep_sel:
    cr = customers_by_rep(sdf, start_date, end_date, rep_sel)
    if cr.empty:
        st.info("Ese Sales Rep no tiene clientes en el rango.")
    else:
        st.dataframe(
            cr[["customer","total_revenue","n_orders","aov","last_sale","recency_days","active_weeks","weeks_span","regularity_ratio","retention_score"]],
            use_container_width=True
        )

# ------------------------
# NOTAS DE AJUSTE
# ------------------------
st.markdown("""
**Notas para ajuste rápido:**
- Pesos del `retention_score` en `make_retention_metrics`: Revenue (0.35), Frequency (0.25), Recency (0.25), Regularity (0.15).
- Regularidad = semanas activas / semanas entre primera y última venta en el rango.
- Si `invoice_num` no está poblado, la cuenta de órdenes usa filas como proxy.
- Puedes filtrar por `oficina`, `sale_location` y orgánico desde la barra lateral.
- Si tu tabla en Supabase se llama distinto, ajusta `st.secrets['supabase_sales'].table`.
""")
