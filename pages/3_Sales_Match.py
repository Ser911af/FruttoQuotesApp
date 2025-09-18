# pages/3_Customer_Retention.py
# Customer Retention Rankings — By Company and By Sales Rep
# Mantiene login + loaders (Supabase/MSSQL) y construye un dashboard de retención

import os
import math
import datetime as dt
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# ✅ 1) Login obligatorio antes de cargar nada pesado
from simple_auth import ensure_login, logout_button
user = ensure_login()
with st.sidebar:
    logout_button()
st.caption(f"Sesión: {user}")

# --- Opcionales (no críticos) ---
try:
    from supabase import create_client  # pip install supabase
except Exception:
    create_client = None

try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

# ------------------------
# CONFIG & PAGE
# ------------------------
st.set_page_config(page_title="Customer Retention Rankings", page_icon="📈", layout="wide")
st.title("📈 Customer Retention Rankings")
st.caption("Rankings de retención por compañía (customer) y por sales rep usando ventas históricas.")

# ------------------------
# HELPERS
# ------------------------
def _load_supabase_client(secret_key: str):
    sec = st.secrets.get(secret_key, None)
    if not sec or not create_client:
        return None
    url = sec.get("url"); key = sec.get("anon_key")
    if not url or not key:
        return None
    return create_client(url, key)

def _coerce_date(s) -> pd.Timestamp:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(None)

def _normalize_txt(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = " ".join(s.split())
    return s

# ------------------------
# SALES LOADER (Soporta tu esquema snake_case)
# ------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_sales() -> pd.DataFrame:
    """
    Carga ventas desde Supabase (st.secrets['supabase_sales']) con paginación.
    Alternativamente, puedes implementar MSSQL si defines st.secrets['mssql_sales'].
    Homologa columnas clave a estándar.
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
        # 2) Azure SQL (si prefieres)
        ms = st.secrets.get("mssql_sales", None)
        if ms:
            st.warning("Implementa el loader MSSQL aquí (sqlalchemy/pyodbc) y devuelve un DataFrame con las columnas esperadas.")
            return pd.DataFrame()
        else:
            st.error("No se encontró fuente de ventas (st.secrets['supabase_sales'] o ['mssql_sales']).")
            return pd.DataFrame()

    if df.empty:
        return df

    # Alias para tolerar nombres (tu tabla snake_case ya cuadra con estos)
    alias_map = {
        "received_date": [
            "received_date", "reqs_date", "created_at", "sale_date"
        ],
        "customer": [
            "customer", "client", "buyer"
        ],
        "sales_rep": [
            "sales_rep", "cus_sales_rep", "buyer_assigned"
        ],
        "total_revenue": [
            "total_revenue", "total_sold_lot_expenses"  # fallback débil
        ],
        "invoice_num": ["invoice_num", "invoice", "invoice #"],
        "invoice_payment_status": ["invoice_payment_status", "payment_status"],
        "id_hash": ["id_hash"],
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

    # Normalizaciones
    sdf["date"] = pd.to_datetime(sdf["received_date"], errors="coerce")
    sdf["customer"] = sdf["customer"].astype(str)
    sdf["customer_c"] = sdf["customer"].apply(_normalize_txt)
    sdf["sales_rep"] = sdf["sales_rep"].fillna("UNASSIGNED").astype(str)
    sdf["sales_rep_c"] = sdf["sales_rep"].apply(_normalize_txt)
    sdf["total_revenue"] = pd.to_numeric(sdf["total_revenue"], errors="coerce")
    sdf["invoice_num"] = sdf["invoice_num"].astype(str)
    sdf["invoice_payment_status"] = sdf["invoice_payment_status"].astype(str)
    sdf["id_hash"] = sdf["id_hash"].astype(str)

    # Filtrado básico
    sdf = sdf.dropna(subset=["date", "customer"])
    # Derivados de tiempo
    sdf["date"] = sdf["date"].dt.tz_localize(None)  # asegurar naive
    sdf["ym"] = sdf["date"].dt.to_period("M").astype(str)   # YYYY-MM

    return sdf[[
        "date", "ym", "customer", "customer_c", "sales_rep", "sales_rep_c",
        "total_revenue", "invoice_num", "invoice_payment_status", "id_hash"
    ]]

# ------------------------
# RETENTION METRICS
# ------------------------
def last_full_month(today: dt.date) -> dt.date:
    # Devuelve el día 1 del mes anterior al actual
    first_this = today.replace(day=1)
    prev_month_end = first_this - dt.timedelta(days=1)
    return prev_month_end.replace(day=1)

def month_to_month_retention(df: pd.DataFrame) -> Dict[str, float]:
    """
    Retención mes-a-mes global y por sales rep.
    Definición: clientes activos en (último mes completo) que también estuvieron activos en (mes-1).
      retention = |clientes_intersección| / |clientes_mes-1|
    """
    if df.empty:
        return {"global": np.nan}

    today = dt.date.today()
    m2 = last_full_month(today)                          # último mes completo (inicio)
    m1 = (m2 - pd.offsets.MonthBegin(1)).date()         # mes previo (inicio)

    m1_str = pd.Period(m1, freq="M").strftime("%Y-%m")  # YYYY-MM
    m2_str = pd.Period(m2, freq="M").strftime("%Y-%m")

    df_m1 = df[df["ym"] == m1_str]
    df_m2 = df[df["ym"] == m2_str]

    # Global
    c1 = set(df_m1["customer_c"].unique().tolist())
    c2 = set(df_m2["customer_c"].unique().tolist())
    global_ret = (len(c1 & c2) / len(c1)) if len(c1) > 0 else np.nan

    # Por sales rep
    rep_rates = {}
    for rep, sub1 in df_m1.groupby("sales_rep_c"):
        prev_customers = set(sub1["customer_c"].unique().tolist())
        curr_customers = set(df_m2[df_m2["sales_rep_c"] == rep]["customer_c"].unique().tolist())
        rep_rates[rep] = (len(prev_customers & curr_customers) / len(prev_customers)) if len(prev_customers) > 0 else np.nan

    return {"global": global_ret, "by_rep": rep_rates, "m1": m1_str, "m2": m2_str}

def customer_retention_table(df: pd.DataFrame, months_window: int = 12, min_orders:int = 2) -> pd.DataFrame:
    """
    Ranking de clientes por proxy de retención:
      - Ventana de meses (default 12).
      - Métricas: recency (días desde última compra), frecuencia (#ordenes),
        active_months (meses con compra), retention_rate = active_months / meses_desde_primera_compra_en_ventana,
        revenue_total y AOV.
    """
    if df.empty:
        return pd.DataFrame()

    end_date = pd.Timestamp.today().normalize()
    start_date = (end_date - pd.DateOffset(months=months_window)).normalize()
    sub = df[(df["date"] >= start_date) & (df["date"] < end_date)].copy()
    if sub.empty:
        return pd.DataFrame()

    # Agregados por customer
    g = sub.groupby("customer", dropna=False)
    agg = g.agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        orders=("invoice_num", "nunique"),
        months_active=("ym", "nunique"),
        revenue=("total_revenue", "sum")
    ).reset_index()

    agg["recency_days"] = (end_date - agg["last_date"]).dt.days
    agg["months_since_first"] = ((end_date.dt.to_period("M").astype(int)) - (agg["first_date"].dt.to_period("M").astype(int)) + 1).clip(lower=1)
    agg["retention_rate"] = (agg["months_active"] / agg["months_since_first"]).astype(float)
    agg["aov"] = agg["revenue"] / agg["orders"].replace({0: np.nan})

    # Filtro de significancia
    agg = agg[agg["orders"] >= min_orders].copy()

    # Score de ranking (ajustable): retención alta, recency baja, frecuencia alta, revenue alto
    # Normalización robusta
    def _nz(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    agg["score"] = (
        0.45 * _nz(agg["retention_rate"]) +
        0.25 * (1 - _nz(agg["recency_days"])) +
        0.20 * _nz(agg["orders"]) +
        0.10 * _nz(agg["revenue"])
    )

    agg = agg.sort_values(["score", "retention_rate", "orders", "revenue"], ascending=[False, False, False, False])

    # Columnas amigables
    out = agg[[
        "customer", "orders", "months_active", "months_since_first", "retention_rate",
        "recency_days", "revenue", "aov", "first_date", "last_date", "score"
    ]].rename(columns={
        "customer": "Customer",
        "orders": "Orders",
        "months_active": "Active Months",
        "months_since_first": "Months Since First",
        "retention_rate": "Retention Rate",
        "recency_days": "Recency (days)",
        "revenue": "Revenue (sum)",
        "aov": "AOV",
        "first_date": "First Purchase",
        "last_date": "Last Purchase",
        "score": "Rank Score"
    })
    return out

def sales_rep_retention_table(df: pd.DataFrame, months_window:int = 12, min_customers:int = 5) -> pd.DataFrame:
    """
    Ranking de sales reps por retención:
      - Cohortes de clientes por rep dentro de la ventana.
      - Métricas: clientes atendidos, clientes activos (meses), retención mes-a-mes propia,
        frecuencia promedio por cliente, revenue total y por cliente.
    """
    if df.empty:
        return pd.DataFrame()

    end_date = pd.Timestamp.today().normalize()
    start_date = (end_date - pd.DateOffset(months=months_window)).normalize()
    sub = df[(df["date"] >= start_date) & (df["date"] < end_date)].copy()
    if sub.empty:
        return pd.DataFrame()

    # Retención m2 vs m1 por rep
    m = month_to_month_retention(df)
    rep_mmr = m.get("by_rep", {}) if isinstance(m, dict) else {}

    # Agregados por rep
    g = sub.groupby("sales_rep", dropna=False)
    agg = g.agg(
        customers=("customer", "nunique"),
        orders=("invoice_num", "nunique"),
        months_active=("ym", "nunique"),
        revenue=("total_revenue", "sum"),
        last_date=("date", "max")
    ).reset_index()

    # Métricas derivadas
    agg["orders_per_customer"] = agg["orders"] / agg["customers"].replace({0: np.nan})
    agg["revenue_per_customer"] = agg["revenue"] / agg["customers"].replace({0: np.nan})
    agg["recency_days"] = (pd.Timestamp.today().normalize() - agg["last_date"]).dt.days
    agg["mmr"] = agg["sales_rep"].apply(lambda r: rep_mmr.get(_normalize_txt(r), np.nan))

    # Filtro reps con base suficiente
    agg = agg[agg["customers"] >= min_customers].copy()

    # Score de ranking: MMR (más peso), engagement y valor
    def _nz(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    # Relleno MMR nan con mediana para no penalizar en exceso
    if agg["mmr"].notna().any():
        mmr_med = agg["mmr"].median(skipna=True)
        agg["mmr"] = agg["mmr"].fillna(mmr_med)
    else:
        agg["mmr"] = np.nan

    agg["score"] = (
        0.45 * _nz(agg["mmr"].fillna(agg["mmr"].median() if agg["mmr"].notna().any() else 0.0)) +
        0.20 * (1 - _nz(agg["recency_days"])) +
        0.20 * _nz(agg["orders_per_customer"]) +
        0.15 * _nz(agg["revenue_per_customer"])
    )

    agg = agg.sort_values(["score", "mmr", "orders_per_customer", "revenue_per_customer"], ascending=[False, False, False, False])

    out = agg[[
        "sales_rep", "customers", "orders", "months_active",
        "orders_per_customer", "revenue", "revenue_per_customer",
        "mmr", "recency_days", "score"
    ]].rename(columns={
        "sales_rep": "Sales Rep",
        "customers": "Customers",
        "orders": "Orders",
        "months_active": "Active Months",
        "orders_per_customer": "Orders / Customer",
        "revenue": "Revenue (sum)",
        "revenue_per_customer": "Revenue / Customer",
        "mmr": "MoM Retention",
        "recency_days": "Recency (days)",
        "score": "Rank Score"
    })
    return out

# ------------------------
# UI CONTROLS
# ------------------------
with st.sidebar:
    st.subheader("Parámetros")
    months_window = st.slider("Ventana (meses)", min_value=3, max_value=24, value=12, step=1)
    min_orders_cust = st.slider("Mín. órdenes por cliente (ranking clientes)", 1, 10, 2, 1)
    min_customers_rep = st.slider("Mín. clientes por rep (ranking reps)", 1, 50, 5, 1)

# ------------------------
# DATA PIPELINE
# ------------------------
sdf = load_sales()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Registros de ventas", value=len(sdf))
with col2:
    st.metric("Clientes únicos", value=sdf["customer"].nunique() if not sdf.empty else 0)
with col3:
    st.metric("Sales reps únicos", value=sdf["sales_rep"].nunique() if not sdf.empty else 0)

if sdf.empty:
    st.stop()

# ------------------------
# GLOBAL MOM-TO-MOM RETENTION KPI
# ------------------------
mom = month_to_month_retention(sdf)
if isinstance(mom, dict):
    m1 = mom.get("m1", "prev"); m2 = mom.get("m2", "last")
    st.subheader("📅 Retención mes-a-mes (global)")
    st.caption(f"Clientes activos en {m2} que también estuvieron activos en {m1} / clientes activos en {m1}.")
    st.metric(f"Global MoM Retention ({m1}→{m2})", value=f"{mom['global']*100:.1f}%" if pd.notna(mom['global']) else "N/A")

# ------------------------
# CUSTOMER RANKING
# ------------------------
st.subheader("🏢 Customer Retention Ranking")
cust_table = customer_retention_table(sdf, months_window=months_window, min_orders=min_orders_cust)
if cust_table.empty:
    st.info("Sin suficientes datos para el ranking de clientes con los parámetros actuales.")
else:
    st.dataframe(cust_table, use_container_width=True)

    if ALTAIR_OK:
        st.markdown("**Distribución: Retention Rate vs Recency**")
        chart = alt.Chart(cust_table.astype({"Retention Rate": float, "Recency (days)": float})).mark_circle(size=60).encode(
            x=alt.X("Retention Rate:Q", title="Retention Rate"),
            y=alt.Y("Recency (days):Q", title="Recency (days)"),
            tooltip=["Customer","Orders","Active Months","Retention Rate","Recency (days)","Revenue (sum)","AOV"]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

# ------------------------
# SALES REP RANKING
# ------------------------
st.subheader("🧑‍💼 Sales Rep Retention Ranking")
rep_table = sales_rep_retention_table(sdf, months_window=months_window, min_customers=min_customers_rep)
if rep_table.empty:
    st.info("Sin suficientes datos para el ranking de reps con los parámetros actuales.")
else:
    st.dataframe(rep_table, use_container_width=True)

    if ALTAIR_OK:
        st.markdown("**MoM Retention vs Revenue/Customer**")
        plot_df = rep_table.rename(columns={"Revenue / Customer": "Revenue_per_Customer"})
        chart2 = alt.Chart(plot_df.astype({"MoM Retention": float, "Revenue_per_Customer": float})).mark_circle(size=80).encode(
            x=alt.X("MoM Retention:Q", title="MoM Retention"),
            y=alt.Y("Revenue_per_Customer:Q", title="Revenue per Customer"),
            tooltip=["Sales Rep","Customers","Orders","MoM Retention","Revenue (sum)","Revenue_per_Customer","Orders / Customer"]
        ).interactive()
        st.altair_chart(chart2, use_container_width=True)

# ------------------------
# NOTAS Y AJUSTES
# ------------------------
st.markdown("""
**Notas de método y ajustes:**
- **Retención global y por rep (MoM):** mide qué porcentaje de clientes activos en el mes **M-1** siguen activos en **M** (últimos **2 meses completos**).
- **Ranking de clientes:** usa un **Rank Score** que pondera *Retention Rate* (meses con compra / meses desde primera compra en la ventana), *Recency inversa*, *Frecuencia* y *Revenue*. Ajusta pesos en el código si lo prefieres.
- **Parámetros clave en la barra lateral:**
  - *Ventana (meses):* periodo de análisis para métricas de cliente y rep.
  - *Mín. órdenes por cliente:* filtra outliers con muy poca señal.
  - *Mín. clientes por rep:* asegura base suficiente para comparar reps.
- **DB:** por defecto toma `st.secrets['supabase_sales']` y la tabla `"ventas_frutto"`. Cambia `table` en secrets si aplicara.
""")
