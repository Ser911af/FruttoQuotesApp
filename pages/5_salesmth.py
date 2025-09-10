# --- MATCH COTIZACIONES (quotations) vs VENTAS (ventas_frutto) EN MEMORIA (SOLO LECTURA) ---

import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st

try:
    from supabase import create_client
except Exception:
    create_client = None

st.markdown("## 🔎 Match diario: Cotizaciones vs Ventas (solo lectura)")

# ------------------------
# Helpers
# ------------------------
def _normalize_txt(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = " ".join(s.split())
    return s

def _as_bool_organic(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().lower()
    return s in {"og", "organic", "true", "t", "1", "yes", "y"} or (isinstance(x, (int, np.integer)) and int(x) == 1)

def _money_to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = "".join(ch for ch in s if (ch.isdigit() or ch in ".-"))
    try:
        return float(s) if s != "" else np.nan
    except Exception:
        return np.nan

def _loc_match_soft(a: str, b: str) -> float:
    """1.0 si iguales; 0.5 si una contiene a la otra; 0.0 si distintas."""
    A, B = _normalize_txt(a), _normalize_txt(b)
    if not A or not B:
        return 0.0
    if A == B:
        return 1.0
    if A in B or B in A:
        return 0.5
    return 0.0

# ------------------------
# Loaders (solo SELECT)
# ------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_quotations_from_supabase() -> pd.DataFrame:
    """Lee toda la tabla quotations en Supabase (solo lectura)."""
    sec = st.secrets.get("supabase_quotes", {})
    if not create_client or not sec:
        st.error("Falta configurar st.secrets['supabase_quotes'] (url, anon_key, table).")
        return pd.DataFrame()
    sb = create_client(sec["url"], sec["anon_key"])
    table_name = sec.get("table", "quotations")

    rows, limit, offset = [], 2000, 0
    while True:
        resp = sb.table(table_name).select("*").range(offset, offset+limit-1).execute()
        data = resp.data or []
        rows.extend(data)
        if len(data) < limit:
            break
        offset += limit
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalización
    q = pd.DataFrame({
        "q_date_str": df.get("cotization_date"),
        "product": df.get("product"),
        "organic_int": df.get("organic"),
        "price": df.get("price"),
        "loc": df.get("location"),
        "vendor": df.get("vendorclean"),
    })
    # Parse fecha: M/D/YYYY
    q["q_date"] = pd.to_datetime(q["q_date_str"], errors="coerce", format="%m/%d/%Y")
    q["product_n"] = q["product"].astype(str).map(_normalize_txt)
    q["is_organic"] = q["organic_int"].map(_as_bool_organic)
    q["price"] = pd.to_numeric(q["price"], errors="coerce")
    q["loc_n"] = q["loc"].astype(str).map(_normalize_txt)
    q["vendor_n"] = q["vendor"].astype(str).map(_normalize_txt)

    q = q.dropna(subset=["q_date", "product_n", "price"])
    # Nos quedamos con las columnas que usaremos
    return q[["q_date", "product_n", "is_organic", "price", "loc_n", "vendor_n"]]

@st.cache_data(ttl=300, show_spinner=False)
def load_sales_readonly() -> pd.DataFrame:
    """
    Lee ventas de tu tabla 'ventas_frutto' **sin escribir**.
    Prioridad:
      1) Supabase (st.secrets['supabase_sales'] con table='ventas_frutto')
      2) MSSQL (st.secrets['mssql_sales'] con DSN/conn string) -> opcional
    """
    # 1) Supabase
    sec = st.secrets.get("supabase_sales", {})
    if create_client and sec:
        sb = create_client(sec["url"], sec["anon_key"])
        table_name = sec.get("table", "ventas_frutto")
        rows, limit, offset = [], 5000, 0
        while True:
            resp = sb.table(table_name).select("*").range(offset, offset+limit-1).execute()
            data = resp.data or []
            rows.extend(data)
            if len(data) < limit:
                break
            offset += limit
        df = pd.DataFrame(rows)
    else:
        # 2) MSSQL (opcional — dejar esqueleto sin ejecutar si no hay secretos)
        ms = st.secrets.get("mssql_sales", None)
        if ms:
            st.warning("Conector MSSQL no implementado en este snippet. Usa pyodbc/sqlalchemy en solo lectura.")
            return pd.DataFrame()
        else:
            st.error("No se encontró fuente de ventas (st.secrets['supabase_sales'] o ['mssql_sales']).")
            return pd.DataFrame()

    if df.empty:
        return df

    # Normalización mínima según tu esquema
    s = pd.DataFrame({
        "date_str": df.get("reqs_date"),
        "product": df.get("product"),
        "organic": df.get("organic"),
        "sale_location": df.get("sale_location"),
        "lot_location": df.get("lot_location"),
        "customer": df.get("customer"),
        "vendor": df.get("vendor"),
        "quantity": pd.to_numeric(df.get("quantity"), errors="coerce"),
        "price_per_unit": pd.to_numeric(df.get("price_per_unit"), errors="coerce"),
    })
    s["date"] = pd.to_datetime(s["date_str"], errors="coerce")  # reqs_date ya es fecha base
    s["product_n"] = s["product"].astype(str).map(_normalize_txt)
    s["is_organic"] = s["organic"].map(_as_bool_organic)
    s["loc_n"] = s["sale_location"].fillna(s["lot_location"]).astype(str).map(_normalize_txt)
    s["customer_n"] = s["customer"].astype(str).map(_normalize_txt)
    s["vendor_n"] = s["vendor"].astype(str).map(_normalize_txt)

    s = s.dropna(subset=["date", "product_n"])
    return s[[
        "date","product_n","is_organic","loc_n","customer_n","vendor_n","quantity","price_per_unit"
    ]]

# ------------------------
# UI: parámetros
# ------------------------
with st.sidebar:
    st.markdown("### ⚙️ Parámetros (solo lectura)")
    # fecha del daily sheet (por defecto: ayer Bogotá)
    bogota_today = (dt.datetime.utcnow() + dt.timedelta(hours=-5)).date()
    default_qday = bogota_today - dt.timedelta(days=1)
    q_day = st.date_input("Fecha de cotizaciones", value=default_qday, key="q_day_match")

    recent_days = st.slider("Ventas recientes (benchmark) — días", 7, 60, 30, 1)
    active_days = st.slider("Actividad cliente/producto — días", 7, 30, 14, 1)
    only_active_vendors = st.checkbox("Solo vendors que han vendido en últimos 30 días", value=False)

# ------------------------
# Pipeline: cargar y filtrar
# ------------------------
qdf = load_quotations_from_supabase()
sdf = load_sales_readonly()

colA, colB = st.columns(2)
with colA: st.metric("Cotizaciones (total)", len(qdf))
with colB: st.metric("Ventas (total)", len(sdf))
if qdf.empty or sdf.empty:
    st.stop()

# Cotizaciones del día elegido
qdf_day = qdf[qdf["q_date"].dt.date == pd.to_datetime(q_day).date()].copy()
st.write(f"**Cotizaciones del {q_day}:** {len(qdf_day)}")
if qdf_day.empty:
    st.info("No hay cotizaciones para la fecha seleccionada.")
    st.stop()

# (Opcional) vendors activos
if only_active_vendors:
    cutoff_v = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=30)
    vend_act = sdf.loc[sdf["date"] >= cutoff_v, "vendor_n"].dropna().unique().tolist()
    qdf_day = qdf_day[qdf_day["vendor_n"].isin(vend_act)].copy()
    st.caption(f"Filtrando a {len(vend_act)} vendors activos (30 días). Cotizaciones día después del filtro: {len(qdf_day)}")

# Compras recientes por cliente+producto (actividad) en ventana active_days
cutoff_recent = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=active_days)
s_recent = sdf.loc[sdf["date"] >= cutoff_recent].copy()
recent_buys = (
    s_recent.groupby(["customer_n","product_n","is_organic"], dropna=False)
    .agg(qty_recent=("quantity","sum"), last_buy=("date","max"))
    .reset_index()
)

# Benchmark 30 días (o recent_days elegido) por cliente+producto exacto
cutoff_bench = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=recent_days)
s_bench = sdf.loc[sdf["date"] >= cutoff_bench].copy()
bench = (
    s_bench.groupby(["customer_n","product_n","is_organic"], dropna=False)["price_per_unit"]
    .median()
    .rename("bench_price")
    .reset_index()
)

# ------------------------
# Match: por product exacto + OG/CV; ubicación suave
# ------------------------
# Expandimos ubicaciones históricas por cliente+producto
cust_prod_locs = (
    sdf.groupby(["customer_n","product_n","is_organic","loc_n"], dropna=False)
    .size().reset_index(name="n")
[["customer_n","product_n","is_organic","loc_n"]]
)

# Construimos candidatos: combinamos cada (cliente, producto) reciente con cotizaciones del día del mismo producto/OG
candidates = (
    recent_buys.merge(qdf_day, how="inner", on=["product_n","is_organic"])
               .merge(cust_prod_locs, how="left", on=["customer_n","product_n","is_organic"])
)

if candidates.empty:
    st.warning("No hay empates (cliente+producto recientes) con las cotizaciones del día.")
    st.stop()

# Score por cercanía de ubicación
candidates["loc_match"] = candidates.apply(lambda r: _loc_match_soft(r.get("loc_n_y"), r.get("loc_n_x")), axis=1)
# Nota: tras el merge, loc_n_x viene del histórico (cust_prod_locs), loc_n_y viene de qdf_day
# renombraremos para claridad
candidates = candidates.rename(columns={"loc_n_x":"loc_hist","loc_n_y":"loc_quote"})

# Adjuntamos benchmark por cliente+producto
candidates = candidates.merge(bench, how="left", on=["customer_n","product_n","is_organic"])

# price_improvement y score
def _price_improv(row):
    bp, pq = row.get("bench_price"), row.get("price")
    if pd.notnull(bp) and bp > 0 and pd.notnull(pq):
        return (bp - pq) / bp
    return np.nan

candidates["price_improvement"] = candidates.apply(_price_improv, axis=1)

# Score final: precio (peso 1.0) + ubicación (0.3)
candidates["score"] = candidates["price_improvement"].fillna(0.0) + 0.3 * candidates["loc_match"].fillna(0.0)

# Orden y selección de columnas
out = candidates[[
    "customer_n","product_n","is_organic","vendor_n","loc_quote","price","bench_price","price_improvement","score","qty_recent","last_buy","loc_hist"
]].copy()

out = out.sort_values(["customer_n","product_n","score","price"], ascending=[True, True, False, True])

# ------------------------
# Mostrar resultados
# ------------------------
st.markdown("### 📋 Ofertas candidatas por cliente + producto (día seleccionado)")
if out.empty:
    st.info("Sin resultados bajo los filtros actuales.")
else:
    out_show = out.copy()
    out_show["OG/CV"] = np.where(out_show["is_organic"], "OG", "CV")
    out_show = out_show.drop(columns=["is_organic"])
    # Formatos
    st.dataframe(
        out_show.rename(columns={
            "customer_n":"customer","product_n":"product","vendor_n":"vendor",
            "loc_quote":"location_quote","loc_hist":"location_hist"
        }).style.format({
            "price":"${:,.2f}",
            "bench_price":"${:,.2f}",
            "price_improvement":"{:.1%}",
            "score":"{:.3f}"
        })
    )

    # Descarga CSV
    csv = out_show.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar CSV (match día)", data=csv, file_name=f"match_cotizaciones_{q_day}.csv", mime="text/csv")

# Diagnóstico rápido
with st.expander("🔬 Diagnóstico (inputs normalizados)"):
    st.write("Cotizaciones (día):", qdf_day.head(20))
    st.write("Compras recientes (cliente+producto):", recent_buys.head(20))
    st.write("Benchmark (cliente+producto):", bench.head(20))
