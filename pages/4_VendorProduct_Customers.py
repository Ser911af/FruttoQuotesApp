import re
import math
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional deps
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
st.set_page_config(page_title="Vendor × Commodity → Customers", page_icon="🧾", layout="wide")
st.title("🧾 Vendor × Commodity → Customers")
st.caption("Encuentra qué clientes compraron una *commodity específica* de un *vendor* dado, con métricas y detalle.")

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


def _winsorized_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0)
    # soft winsorization at 2–98%
    lo, hi = x.quantile(0.02), x.quantile(0.98)
    xw = x.clip(lower=lo, upper=hi)
    mu, sd = xw.mean(), xw.std(ddof=0)
    return (x - mu) / (sd + 1e-9)


def _round_cols(df: pd.DataFrame, spec: dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    for c, n in spec.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(n)
    return df


def _fmt_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            df[c] = s.dt.strftime("%Y-%m-%d").fillna("")
    return df


def _styled_table(df_disp: pd.DataFrame, styles: dict) -> "pd.io.formats.style.Styler":
    return df_disp.style.format(styles, na_rep="")


def _load_supabase_client(secret_key: str):
    sec = st.secrets.get(secret_key, None)
    if not sec or not create_client:
        return None
    url = sec.get("url")
    key = sec.get("anon_key")
    if not url or not key:
        return None
    return create_client(url, key)


# ------------------------
# SALES LOADER (extends alias map with product + commodity fields)
# ------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_sales() -> pd.DataFrame:
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
        "received_date": ["received_date"],
        "created_at": ["created_at"],
        # IDs
        "invoice_num": ["invoice_num", "invoice #", "invoice"],
        "sales_order": ["sales_order"],
        # Dimensions
        "vendor": ["vendor", "shipper", "supplier"],
        "buyer_assigned": ["buyer_assigned", "buyer_asigned", "buyer"],
        "customer": ["customer", "customer_name", "client", "cliente"],
        "product": [
            "product", "item", "item_name", "product_name", "desc", "description",
            "sku", "upc", "gtin"
        ],
        # NEW: commodity (includes common typos / synonyms)
        "commodity": ["commodity", "comodity", "commoditie", "cooditie", "category", "family", "product_group"],
        "lot_location": ["lot_location"],
        # Metrics
        "quantity": ["quantity", "qty"],
        "total_revenue": ["total_revenue", "total", "line_total", "revenue"],
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
    sdf["product_c"] = sdf["product"].astype(str).map(_normalize_txt)
    sdf["customer_c"] = sdf["customer"].astype(str).map(_normalize_txt)
    sdf["commodity_c"] = sdf["commodity"].astype(str).map(_normalize_txt)

    # Display labels (Title Case)
    sdf["vendor_disp"] = sdf["vendor"].astype(str).str.title()
    sdf["product_disp"] = sdf["product"].astype(str)
    sdf["customer_disp"] = sdf["customer"].astype(str).str.title()
    sdf["commodity_disp"] = sdf["commodity"].astype(str).str.title()

    # Unified date
    sdf["date"] = sdf["received_date"].fillna(sdf["reqs_date"]).fillna(sdf["created_at"])  # fallback chain
    sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")

    # Order identifier: prefer invoice_num; if absent, we'll count rows as proxy later
    sdf["order_id"] = sdf["invoice_num"].astype(str)
    sdf.loc[sdf["order_id"].isin(["nan", "none", "", "NaT"]), "order_id"] = np.nan

    return sdf


# ------------------------
# SIDEBAR FILTERS
# ------------------------
with st.sidebar:
    st.subheader("Filtros")
    today_bo = pd.Timestamp.now(tz="America/Bogota").normalize()
    default_start = (today_bo - pd.Timedelta(days=15)).tz_localize(None)
    default_end = today_bo.tz_localize(None)

    date_range = st.date_input("Date range", value=(default_start, default_end))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)  # end exclusive
    else:
        start_date, end_date = default_start, default_end + pd.Timedelta(days=1)

    st.caption("Primero el vendor, luego la commodity — como tacos y salsa.")

# Adaptive as_of_date
today_norm = pd.Timestamp.now(tz="America/Bogota").normalize().tz_localize(None)
end_minus_1 = (end_date - pd.Timedelta(days=1)).normalize()
as_of_date = min(today_norm, end_minus_1)
st.caption(f"Recency as-of: **{as_of_date.date()}**")

# ------------------------
# DATA
# ------------------------
sdf = load_sales()

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Sales records", value=len(sdf))
with m2:
    st.metric("Unique vendors", value=int(sdf["vendor_c"].nunique()) if not sdf.empty else 0)
with m3:
    st.metric("Unique commodities", value=int(sdf["commodity_c"].nunique()) if not sdf.empty else 0)
with m4:
    st.metric("Unique customers", value=int(sdf["customer_c"].nunique()) if not sdf.empty else 0)

if sdf.empty:
    st.stop()

mask = (sdf["date"] >= start_date) & (sdf["date"] < end_date)
sdf_f = sdf[mask].copy()
st.caption(f"Filtered rows: {len(sdf_f)} in selected range.")
if sdf_f.empty:
    st.warning("No data in the selected range.")
    st.stop()

# ------------------------
# UI: VENDOR → COMMODITY selectors (dependent)
# ------------------------
left, right = st.columns([1, 1])
with left:
    vend_opts = [""] + sorted(sdf_f["vendor_disp"].dropna().unique().tolist())
    vendor_sel_disp = st.selectbox("Vendor", options=vend_opts, index=0, placeholder="Select...")

if vendor_sel_disp:
    vendor_norm = (
        sdf_f.loc[sdf_f["vendor_disp"] == vendor_sel_disp, "vendor_c"].dropna().iloc[0]
        if (sdf_f["vendor_disp"] == vendor_sel_disp).any()
        else _normalize_txt(vendor_sel_disp)
    )
    vmask = sdf_f["vendor_c"] == vendor_norm

    # List of commodities available for that vendor
    comms = sdf_f.loc[vmask, ["commodity_c", "commodity_disp"]].dropna().drop_duplicates()
    comm_opts = [""] + sorted(comms["commodity_disp"].astype(str).unique().tolist())

    with right:
        commodity_sel_disp = st.selectbox("Commodity (de ese Vendor)", options=comm_opts, index=0, placeholder="Select...")
else:
    commodity_sel_disp = ""

if not vendor_sel_disp or not commodity_sel_disp:
    st.info("Selecciona un **Vendor** y una **Commodity** para ver los clientes.")
    st.stop()

# Canonical key for selected commodity
commodity_norm = (
    sdf_f.loc[sdf_f["commodity_disp"] == commodity_sel_disp, "commodity_c"].dropna().iloc[0]
    if (sdf_f["commodity_disp"] == commodity_sel_disp).any()
    else _normalize_txt(commodity_sel_disp)
)

# Slice: rows for that (vendor, commodity)
vp = sdf_f[(sdf_f["vendor_c"] == vendor_norm) & (sdf_f["commodity_c"] == commodity_norm)].copy()
if vp.empty:
    st.warning("Ese vendor no vendió esa commodity en el rango seleccionado (o los nombres no coinciden).")
    st.stop()

# ------------------------
# METRICS by CUSTOMER for (Vendor, Commodity)
# ------------------------
wk = vp["date"].dt.isocalendar()
vp["week_key"] = np.where(
    vp["date"].notna(),
    wk["year"].astype(int).astype(str) + "-W" + wk["week"].astype(int).astype(str).str.zfill(2),
    ""
)

# Orders: group by invoice if available
if vp["order_id"].notna().any():
    orders = vp.groupby(["customer_c", "order_id"], dropna=False).agg(
        order_rev=("total_revenue", "sum")
    ).reset_index()
    orders_per_cust = orders.groupby("customer_c").agg(
        n_orders=("order_id", "nunique"),
        aov=("order_rev", "mean"),
    )
else:
    orders_per_cust = vp.groupby("customer_c").agg(
        n_orders=("date", "count"),
        aov=("total_revenue", "mean"),
    )

agg = vp.groupby("customer_c").agg(
    customer_disp=("customer_disp", lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""),
    total_revenue=("total_revenue", "sum"),
    total_qty=("quantity", "sum"),
    first_sale=("date", "min"),
    last_sale=("date", "max"),
).join(orders_per_cust, how="left").fillna({"n_orders": 0, "aov": 0})

# Recency
agg["recency_days"] = (as_of_date - agg["last_sale"].dt.normalize()).dt.days

# Rankings (z-scores) — optional spice
agg["z_rev"] = _winsorized_z(agg["total_revenue"])  # 60%
agg["z_freq"] = _winsorized_z(agg["n_orders"])      # 40%
agg["vp_rank_score"] = 0.6 * agg["z_rev"] + 0.4 * agg["z_freq"]

cust_tbl = (
    agg.reset_index()
       .sort_values(["vp_rank_score", "total_revenue"], ascending=[False, False])
)

# Display formatting
cust_tbl = _fmt_dates(cust_tbl, ["first_sale", "last_sale"]).copy()

cdisp_cols = [
    "customer_disp", "n_orders", "total_qty", "aov", "total_revenue",
    "first_sale", "last_sale", "recency_days",
]

cdisp = cust_tbl[cdisp_cols].rename(columns={
    "customer_disp": "Customer",
    "n_orders": "Orders",
    "total_qty": "Qty",
    "aov": "AOV",
    "total_revenue": "Revenue",
    "first_sale": "First Sale",
    "last_sale": "Last Sale",
    "recency_days": "Recency (days)",
})

styles = {
    "Orders": "{:,.0f}",
    "Qty": "{:,.0f}",
    "AOV": "${:,.2f}",
    "Revenue": "${:,.0f}",
    "Recency (days)": "{:,.0f}",
}

st.subheader("👥 Customers that bought it")
st.dataframe(_styled_table(cdisp, styles), use_container_width=True, hide_index=True)

# Quick KPIs
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Customers", value=int(cust_tbl.shape[0]))
with k2:
    st.metric("Total Revenue", value=f"${cust_tbl['total_revenue'].sum():,.0f}")
with k3:
    st.metric("Units", value=f"{cust_tbl['total_qty'].sum():,.0f}")
with k4:
    st.metric("Orders", value=f"{int(cust_tbl['n_orders'].sum())}")

# Chart (top 25 by revenue)
if ALTAIR_OK and len(cust_tbl) > 0:
    chart = alt.Chart(cust_tbl.head(25)).mark_bar().encode(
        x=alt.X("total_revenue:Q", title="Revenue"),
        y=alt.Y("customer_disp:N", sort="-x", title="Customer"),
        tooltip=["customer_disp", "n_orders", "total_qty", "aov", "total_revenue"],
    ).properties(height=480)
    st.altair_chart(chart, use_container_width=True)

# ------------------------
# DETAIL: line items (for audits & exports)
# ------------------------
st.subheader("📄 Line Items (detalle)")
cols_keep = [
    "date", "invoice_num", "sales_order", "customer_disp", "price_per_unit", "quantity", "total_revenue",
]
ld = vp[cols_keep].copy()
ld = _fmt_dates(ld, ["date"]).rename(columns={
    "date": "Date",
    "invoice_num": "Invoice",
    "sales_order": "Sales Order",
    "customer_disp": "Customer",
    "price_per_unit": "Unit Price",
    "quantity": "Qty",
    "total_revenue": "Revenue",
})

st.dataframe(
    _styled_table(ld, {"Unit Price": "${:,.2f}", "Qty": "{:,.0f}", "Revenue": "${:,.2f}"}),
    use_container_width=True,
    hide_index=True,
)

# Export buttons (commodity-aware names)
csv1 = cdisp.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download customers CSV", data=csv1, file_name="customers_vendor_commodity.csv", mime="text/csv")

csv2 = ld.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download line items CSV", data=csv2, file_name="line_items_vendor_commodity.csv", mime="text/csv")

# ------------------------
# NOTES
# ------------------------
st.markdown(
    f"""
**Notas**
- El rango de fechas es **inicio inclusivo** y **fin exclusivo** (+1 día internamente).
- *Recency* medido **al {as_of_date.date()}**.
- Los nombres se normalizan internamente (lower-case) para los joins/agrupaciones; se muestran "bonitos" para humanos.
- Si no hay `invoice_num`, el conteo de órdenes usa filas como proxy (con cariño y transparencia).
"""
)
