# pages/5_VendorProduct_Customers.py
# Vendor–Product → Customers Explorer

import re
import math
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from simple_auth import ensure_login, logout_button

user = ensure_login()
with st.sidebar:
    logout_button()
st.caption(f"Session: {user}")

try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

try:
    from supabase import create_client  # pip install supabase
except Exception:
    create_client = None

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Vendor–Product → Customers", page_icon="🧭", layout="wide")
st.title("🧭 Vendor–Product → Customers (Compras/Ventas)")

# ------------------------
# HELPERS
# ------------------------
def _normalize_txt(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00A0", " ")  # NBSP -> space
    s = s.strip().lower()
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

def _fmt_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            df[c] = s.dt.strftime("%Y-%m-%d").fillna("")
    return df

def _round_cols(df: pd.DataFrame, spec: dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    for c, n in spec.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(n)
    return df

def _styled_table(df_disp: pd.DataFrame, styles: dict) -> "pd.io.formats.style.Styler":
    # Return a Styler so Streamlit formats strings without converting data types
    return df_disp.style.format(styles, na_rep="")

def _week_key(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    iso = ts.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"

# ------------------------
# SALES LOADER
# ------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_sales() -> pd.DataFrame:
    """
    Carga ventas desde Supabase (o error visible).
    Estandariza alias para columnas clave e incluye dimensiones de producto.
    """
    sb = _load_supabase_client("supabase_sales")
    if sb:
        table_name = st.secrets.get("supabase_sales", {}).get("table", "ventas_frutto")
        rows, limit, offset = [], 1000, 0
        while True:
            q = sb.table(table_name).select("*").range(offset, offset + limit - 1).execute()
            data = q.data or []
            rows.extend(data)
            got = len(data)
            if got < limit:
                break
            offset += got
        df = pd.DataFrame(rows)
    else:
        st.error("No sales source found (st.secrets['supabase_sales']).")
        return pd.DataFrame()

    if df.empty:
        return df

    alias_map = {
        # Dates
        "reqs_date": ["reqs_date"],
        "received_date": ["received_date", "date_received", "rec_date"],
        "created_at": ["created_at"],
        # IDs
        "invoice_num": ["invoice_num", "invoice #", "invoice"],
        "sales_order": ["sales_order", "so_number", "order_number"],
        # Core dimensions
        "vendor": ["vendor", "shipper", "supplier"],
        "customer": ["customer", "customer_name", "client", "buyer_name"],
        "buyer_assigned": ["buyer_assigned", "buyer_asigned", "buyer"],
        "sales_rep": ["sales_rep", "cus_sales_rep"],
        "lot_location": ["lot_location"],
        # Product dimensions (robust aliases)
        "product": ["product", "item", "commodity", "sku", "item_desc", "description", "product_name"],
        "brand": ["brand"],
        "pack": ["pack", "pack_size", "package"],
        "variety": ["variety", "sub_commodity"],
        # Metrics
        "quantity": ["quantity", "qty"],
        "total_revenue": ["total_revenue", "amount", "total"],
        "price_per_unit": ["price_per_unit", "sell_price", "unit_price", "price"],
    }

    std = {}
    for std_col, candidates in alias_map.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        std[std_col] = df[found] if found else pd.Series([None] * len(df))

    sdf = pd.DataFrame(std)

    # Types
    for c in ["reqs_date", "received_date", "created_at"]:
        sdf[c] = pd.to_datetime(sdf[c], errors="coerce")
    for c in ["quantity", "total_revenue", "price_per_unit"]:
        sdf[c] = pd.to_numeric(sdf[c], errors="coerce")

    # Canonical normalized keys
    sdf["vendor_c"] = sdf["vendor"].astype(str).map(_normalize_txt)
    sdf["customer_c"] = sdf["customer"].astype(str).map(_normalize_txt)
    sdf["buyer_c"] = sdf["buyer_assigned"].astype(str).map(_normalize_txt)

    # Producto canónico: combinamos columna principal + variedad/brand/pack si existen
    # para diferenciar "Roma Tomato 25lb" vs "Roma Tomato 10lb"
    parts = []
    for col in ["product", "variety", "brand", "pack"]:
        if col in sdf.columns:
            parts.append(sdf[col].astype(str))
    if parts:
        combo = parts[0]
        for p in parts[1:]:
            combo = combo.where(p.isna() | (p.astype(str).str.strip() == ""), combo + " | " + p.astype(str))
    else:
        combo = sdf["product"].astype(str)
    sdf["product_full"] = combo.replace("nan", "").replace("None", "")

    sdf["product_c"] = sdf["product_full"].astype(str).map(_normalize_txt)

    # Display labels
    sdf["vendor_disp"] = sdf["vendor"].astype(str).str.title()
    sdf["customer_disp"] = sdf["customer"].astype(str).str.title()
    sdf["product_disp"] = sdf["product_full"].astype(str).str.strip()

    # Unified date
    sdf["date"] = sdf["received_date"].fillna(sdf["reqs_date"]).fillna(sdf["created_at"])  # fallback chain
    sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")

    # Order identifier: prefer invoice_num; if absent, fallback to SO or NaN
    sdf["order_id"] = sdf["invoice_num"].astype(str)
    sdf.loc[sdf["order_id"].isin(["nan", "none", ""]), "order_id"] = np.nan
    if sdf["order_id"].isna().all() and "sales_order" in sdf:
        so = sdf["sales_order"].astype(str)
        so = so.where(~so.isin(["nan", "none", ""]), np.nan)
        sdf["order_id"] = so

    # Useful derived
    sdf["week_key"] = sdf["date"].apply(_week_key)

    return sdf

# ------------------------
# SIDEBAR FILTERS
# ------------------------
with st.sidebar:
    st.subheader("Filtros")
    today_bo = pd.Timestamp.now(tz="America/Bogota").normalize()
    default_start = (today_bo - pd.Timedelta(days=90)).tz_localize(None)
    default_end = today_bo.tz_localize(None)

    date_range = st.date_input("Date range", value=(default_start, default_end))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)  # end exclusive
    else:
        start_date, end_date = default_start, default_end + pd.Timedelta(days=1)

# Adaptive as_of date
today_norm = pd.Timestamp.now(tz="America/Bogota").normalize().tz_localize(None)
end_minus_1 = (end_date - pd.Timedelta(days=1)).normalize()
as_of_date = min(today_norm, end_minus_1)
st.caption(f"Recency as-of: **{as_of_date.date()}**")

# ------------------------
# DATA
# ------------------------
sdf = load_sales()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Sales records", value=len(sdf))
with col2:
    st.metric("Unique vendors", value=int(sdf["vendor_c"].nunique()) if not sdf.empty else 0)
with col3:
    st.metric("Unique products", value=int(sdf["product_c"].nunique()) if not sdf.empty else 0)
with col4:
    st.metric("Unique customers", value=int(sdf["customer_c"].nunique()) if not sdf.empty else 0)

if sdf.empty:
    st.stop()

# Date filter
mask = (sdf["date"] >= start_date) & (sdf["date"] < end_date)
sdf_f = sdf[mask].copy()
st.caption(f"Filtered rows: {len(sdf_f)} in selected range.")
if sdf_f.empty:
    st.warning("No data in the selected range.")
    st.stop()

# ------------------------
# UI: CASCADING SELECTORS
# ------------------------
st.subheader("🎯 Selección Vendor y Producto")

# Vendor options
vend_map = (
    sdf_f[["vendor_c", "vendor_disp"]]
    .drop_duplicates()
    .sort_values("vendor_disp", kind="mergesort")
)
vend_options = [""] + vend_map["vendor_disp"].tolist()
vend_disp = st.selectbox("Vendor", options=vend_options, index=0, placeholder="Selecciona un Vendor…")

selected_vendor_c = ""
if vend_disp:
    selected_vendor_c = vend_map.loc[vend_map["vendor_disp"] == vend_disp, "vendor_c"].iloc[0]

# Product options filtered by vendor (si no hay vendor, lista global truncada para performance)
if selected_vendor_c:
    prod_map = (
        sdf_f.loc[sdf_f["vendor_c"] == selected_vendor_c, ["product_c", "product_disp"]]
        .drop_duplicates()
        .sort_values("product_disp", kind="mergesort")
    )
else:
    # fallback global top-N (por si quieren ver algo rápido)
    prod_map = (
        sdf_f[["product_c", "product_disp"]]
        .value_counts()
        .reset_index()
        .sort_values(0, ascending=False)
        .head(500)[["product_c", "product_disp"]]
        .drop_duplicates()
    )

prod_options = [""] + prod_map["product_disp"].fillna("").tolist()
prod_disp = st.selectbox("Producto (depende del Vendor)", options=prod_options, index=0, placeholder="Selecciona un Producto…")

selected_product_c = ""
if prod_disp:
    selected_product_c = prod_map.loc[prod_map["product_disp"] == prod_disp, "product_c"].iloc[0]

# ------------------------
# CORE: CUSTOMERS WHO BOUGHT (VENDOR, PRODUCT)
# ------------------------
st.subheader("👥 Customers que compraron el Vendor–Producto")
def customers_for_vendor_product(df: pd.DataFrame, vendor_c: str, product_c: str) -> pd.DataFrame:
    if not vendor_c or not product_c:
        return pd.DataFrame()

    d = df[(df["vendor_c"] == vendor_c) & (df["product_c"] == product_c)].copy()
    if d.empty:
        return pd.DataFrame()

    # Orders: group by invoice if present; otherwise count rows
    if d["order_id"].notna().any():
        orders = d.groupby(["customer_c", "order_id"], dropna=False).agg(
            order_rev=("total_revenue", "sum"),
            order_qty=("quantity", "sum"),
            order_price=("price_per_unit", "mean"),
            last_date=("date", "max"),
            first_date=("date", "min"),
        ).reset_index()
        by_cust = orders.groupby("customer_c").agg(
            n_orders=("order_id", "nunique"),
            total_revenue=("order_rev", "sum"),
            total_qty=("order_qty", "sum"),
            aov=("order_rev", "mean"),
            last_purchase=("last_date", "max"),
            first_purchase=("first_date", "min"),
        )
    else:
        by_cust = d.groupby("customer_c").agg(
            n_orders=("date", "count"),
            total_revenue=("total_revenue", "sum"),
            total_qty=("quantity", "sum"),
            aov=("total_revenue", "mean"),
            last_purchase=("date", "max"),
            first_purchase=("date", "min"),
        )

    # Price metrics within the pair
    price_stats = d.groupby("customer_c").agg(
        avg_unit_price=("price_per_unit", "mean"),
        median_unit_price=("price_per_unit", "median"),
    )

    # Recency
    out = by_cust.join(price_stats, how="left")
    out["recency_days"] = (as_of_date.normalize() - out["last_purchase"].dt.normalize()).dt.days

    # Display labels
    disp_map_c = d[["customer_c", "customer_disp"]].drop_duplicates()
    out = (
        out.reset_index()
        .merge(disp_map_c, on="customer_c", how="left")
        .rename(columns={"customer_disp": "customer"})
        .sort_values(["total_revenue", "total_qty"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # Share dentro del Vendor–Producto (quién se lleva la tajada del pastel 🍰)
    total_rev_pair = out["total_revenue"].sum()
    out["rev_share_vendor_product"] = np.where(
        total_rev_pair > 0, out["total_revenue"] / total_rev_pair, 0.0
    )

    cols = [
        "customer", "n_orders", "total_qty", "total_revenue",
        "aov", "avg_unit_price", "median_unit_price",
        "first_purchase", "last_purchase", "recency_days",
        "rev_share_vendor_product",
    ]
    return out[cols]

def vendor_product_overview(df: pd.DataFrame, vendor_c: str, product_c: str) -> pd.DataFrame:
    if not vendor_c or not product_c:
        return pd.DataFrame()
    d = df[(df["vendor_c"] == vendor_c) & (df["product_c"] == product_c)].copy()
    if d.empty:
        return pd.DataFrame()

    # Totales y temporalidad
    total_rev = d["total_revenue"].sum()
    total_qty = d["quantity"].sum()
    first_dt = d["date"].min()
    last_dt = d["date"].max()
    active_weeks = d["week_key"].nunique()
    span_days = (last_dt.normalize() - first_dt.normalize()).days if pd.notna(first_dt) and pd.notna(last_dt) else 0
    weeks_span = (span_days // 7) + 1 if span_days >= 0 else 0
    regularity = (active_weeks / weeks_span) if weeks_span > 0 else np.nan

    # #Orders
    if d["order_id"].notna().any():
        n_orders = d["order_id"].nunique()
        aov = d.groupby("order_id")["total_revenue"].sum().mean()
    else:
        n_orders = len(d)
        aov = d["total_revenue"].mean()

    # Precio
    avg_price = d["price_per_unit"].mean()
    med_price = d["price_per_unit"].median()

    # #Customers
    n_customers = d["customer_c"].nunique()

    ov = pd.DataFrame([{
        "total_revenue": total_rev,
        "total_qty": total_qty,
        "n_orders": n_orders,
        "aov": aov,
        "avg_unit_price": avg_price,
        "median_unit_price": med_price,
        "first_sale": first_dt,
        "last_sale": last_dt,
        "recency_days": (as_of_date.normalize() - last_dt.normalize()).days if pd.notna(last_dt) else np.nan,
        "active_weeks": active_weeks,
        "weeks_span": weeks_span,
        "regularity_ratio": np.clip(regularity, 0, 1) if pd.notna(regularity) else np.nan,
        "n_customers": n_customers,
    }])
    return ov

# Compute
cust_tbl = customers_for_vendor_product(sdf_f, selected_vendor_c, selected_product_c)
ov_tbl = vendor_product_overview(sdf_f, selected_vendor_c, selected_product_c)

# Labels visibles
if selected_vendor_c:
    st.caption(f"Vendor seleccionado: **{vend_disp}**")
if selected_product_c:
    st.caption(f"Producto seleccionado: **{prod_disp}**")

# Overview cards
if not ov_tbl.empty:
    o = _fmt_dates(ov_tbl.copy(), ["first_sale", "last_sale"])
    o = _round_cols(o, {
        "aov": 2, "avg_unit_price": 2, "median_unit_price": 2, "regularity_ratio": 4
    })
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Revenue", f"${o['total_revenue'].iloc[0]:,.0f}")
    c2.metric("Total Qty", f"{o['total_qty'].iloc[0]:,.0f}")
    c3.metric("# Órdenes", f"{int(o['n_orders'].iloc[0])}")
    c4.metric("Avg Unit Price", f"${o['avg_unit_price'].iloc[0]:,.2f}")
    c5.metric("Clientes únicos", f"{int(o['n_customers'].iloc[0])}")
    c6.metric("Recency (días)", f"{int(o['recency_days'].iloc[0]) if pd.notna(o['recency_days'].iloc[0]) else 0}")

    with st.expander("Detalle Vendor–Producto", expanded=False):
        o_disp = _title_cols(o[[
            "total_revenue","total_qty","n_orders","aov","avg_unit_price","median_unit_price",
            "first_sale","last_sale","recency_days","active_weeks","weeks_span","regularity_ratio","n_customers"
        ]])
        o_styles = {
            "Total Revenue": "${:,.0f}",
            "Total Qty": "{:,.0f}",
            "N Orders": "{:,.0f}",
            "Aov": "${:,.2f}",
            "Avg Unit Price": "${:,.2f}",
            "Median Unit Price": "${:,.2f}",
            "Recency Days": "{:,.0f}",
            "Active Weeks": "{:,.0f}",
            "Weeks Span": "{:,.0f}",
            "Regularity Ratio": "{:.2%}",
            "N Customers": "{:,.0f}",
        }
        st.dataframe(_styled_table(o_disp, o_styles), use_container_width=True, hide_index=True)

# Customers table
if selected_vendor_c and selected_product_c:
    if cust_tbl.empty:
        st.info("No hay compras para ese Vendor–Producto en el rango seleccionado. (O el producto solo existe en universos paralelos 🪐)")
    else:
        t = cust_tbl.copy()
        t = _fmt_dates(t, ["first_purchase", "last_purchase"])
        t = _round_cols(t, {
            "aov": 2, "avg_unit_price": 2, "median_unit_price": 2, "rev_share_vendor_product": 4
        })
        t = t.rename(columns={
            "customer": "customer",
            "n_orders": "n_orders",
            "total_qty": "total_qty",
            "total_revenue": "total_revenue",
            "aov": "aov",
            "avg_unit_price": "avg_unit_price",
            "median_unit_price": "median_unit_price",
            "first_purchase": "first_purchase",
            "last_purchase": "last_purchase",
            "recency_days": "recency_days",
            "rev_share_vendor_product": "rev_share_vendor_product",
        })

        t_disp = _title_cols(t)
        styles = {
            "N Orders": "{:,.0f}",
            "Total Qty": "{:,.0f}",
            "Total Revenue": "${:,.0f}",
            "Aov": "${:,.2f}",
            "Avg Unit Price": "${:,.2f}",
            "Median Unit Price": "${:,.2f}",
            "Recency Days": "{:,.0f}",
            "Rev Share Vendor Product": "{:.2%}",
        }
        st.dataframe(_styled_table(t_disp, styles), use_container_width=True, hide_index=True)

        # Download
        csv = t_disp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV (Customers Vendor–Producto)", data=csv, file_name="customers_vendor_product.csv", mime="text/csv")

        # Charts
        if ALTAIR_OK:
            top_n = min(25, len(t))
            cA, cB = st.columns(2)
            with cA:
                chart_rev = alt.Chart(t.head(top_n)).mark_bar().encode(
                    x=alt.X("total_revenue:Q", title="Revenue"),
                    y=alt.Y("customer:N", sort="-x", title="Customer"),
                    tooltip=["customer","n_orders","total_qty","total_revenue","aov","avg_unit_price","recency_days"]
                ).properties(height=420)
                st.altair_chart(chart_rev, use_container_width=True)
            with cB:
                chart_qty = alt.Chart(t.head(top_n)).mark_bar().encode(
                    x=alt.X("total_qty:Q", title="Qty"),
                    y=alt.Y("customer:N", sort="-x", title="Customer"),
                    tooltip=["customer","n_orders","total_qty","total_revenue","avg_unit_price","recency_days"]
                ).properties(height=420)
                st.altair_chart(chart_qty, use_container_width=True)

# ------------------------
# EXPLICACIÓN
# ------------------------
with st.expander("ℹ️ ¿Qué hace este módulo?", expanded=False):
    st.markdown(
        """
**Objetivo:** Dado un *Vendor* y un *Producto* específico, listar **qué clientes** compraron ese par, con métricas clave:

- **n_orders**: número de órdenes (o filas si no hay `invoice_num`).
- **total_qty** y **total_revenue** del par.
- **AOV** (Average Order Value) dentro de ese par.
- **Avg/Median Unit Price**: precio promedio/mediano.
- **First/Last purchase** y **Recency (días)**.
- **Rev Share Vendor Product**: % del revenue del par que aporta cada cliente.

**Notas**
- El rango de fechas es **inicio inclusivo** y **fin exclusivo** (+1 día internamente).
- La recencia se mide respecto a la **fecha de referencia** (*as-of date*).
- Selectores en cascada: primero **Vendor**, luego **Producto** disponible para ese Vendor.
- Las etiquetas muestran *Title Case*, pero la lógica usa claves normalizadas (`*_c`) para consistencia.
        """
    )
