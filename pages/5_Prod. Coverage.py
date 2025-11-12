import re
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
st.set_page_config(page_title="Filters → (Vendors or Customers)", page_icon="🧾", layout="wide")
st.title("🧾 Commodity • Organic • Location → Vendors or Customers")
st.caption(
    "Filter by *commodity*, *organic* flag, *location*, and date range. Then choose whether to analyze **Vendors** or **Customers**. Pick one to see purchased products with metrics (Invoices, Units, Revenue)."
)

# ------------------------
# HELPERS
# ------------------------

def _normalize_txt(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00A0", " ")  # NBSP → space
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


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
# DATA LOADER
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
        "customer": ["customer", "customer_name", "client", "cliente"],
        "product": [
            "product", "item", "item_name", "product_name", "desc", "description",
            "sku", "upc", "gtin"
        ],
        "commodity": ["commodity", "comodity", "commoditie", "cooditie", "category", "family", "product_group"],
        "lot_location": ["lot_location", "location", "lot", "loc"],
        # Optional organic flag (we will also infer from text)
        "is_organic": ["is_organic", "organic", "og", "bio", "is_og", "organico", "org"],
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

    # Canonicals
    sdf["vendor_c"] = sdf["vendor"].astype(str).map(_normalize_txt)
    sdf["product_c"] = sdf["product"].astype(str).map(_normalize_txt)
    sdf["customer_c"] = sdf["customer"].astype(str).map(_normalize_txt)
    sdf["commodity_c"] = sdf["commodity"].astype(str).map(_normalize_txt)
    sdf["location_c"] = sdf["lot_location"].astype(str).map(_normalize_txt)

    # Display labels
    sdf["vendor_disp"] = sdf["vendor"].astype(str).str.title()
    sdf["product_disp"] = sdf["product"].astype(str)
    sdf["customer_disp"] = sdf["customer"].astype(str).str.title()
    sdf["commodity_disp"] = sdf["commodity"].astype(str).str.title()
    sdf["location_disp"] = sdf["lot_location"].astype(str).str.title()

    # Unified date & order id
    sdf["date"] = sdf["received_date"].fillna(sdf["reqs_date"]).fillna(sdf["created_at"])  # fallback chain
    sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")
    sdf["order_id"] = sdf["invoice_num"].astype(str)
    sdf.loc[sdf["order_id"].isin(["nan", "none", "", "NaT"]), "order_id"] = np.nan

    # Organic boolean (column or inferred from text)
    def infer_organic(row):
        v = str(row.get("is_organic", "")).strip().lower()
        if v in {"true", "t", "1", "yes", "y", "si", "sí"}:
            return True
        if v in {"false", "f", "0", "no", "n"}:
            return False
        txt = " ".join([str(row.get("product", "")), str(row.get("commodity", ""))]).lower()
        return any(k in txt for k in ["organic", "org.", "og ", " og", "bio", "org-", "org/"])

    sdf["is_organic_bool"] = sdf.apply(infer_organic, axis=1)

    return sdf

# ------------------------
# SIDEBAR FILTERS
# ------------------------
with st.sidebar:
    st.subheader("Filters")

    # Date range
    today_bo = pd.Timestamp.now(tz="America/Bogota").normalize()
    default_start = (today_bo - pd.Timedelta(days=30)).tz_localize(None)
    default_end = today_bo.tz_localize(None)

    date_range = st.date_input("Date range", value=(default_start, default_end))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)  # end exclusive
    else:
        start_date, end_date = default_start, default_end + pd.Timedelta(days=1)

# ------------------------
# DATA
# ------------------------
sdf = load_sales()
if sdf.empty:
    st.stop()

mask = (sdf["date"] >= start_date) & (sdf["date"] < end_date)
sdf_f = sdf[mask].copy()

# Dropdowns based on filtered window
with st.sidebar:
    # Scope toggle: analyze Vendors or Customers
    scope = st.radio("Analyze", options=["Vendors", "Customers"], index=0, horizontal=True)

    # Commodity filter with "All"
    comm_opts = ["All"] + sorted(sdf_f["commodity_disp"].dropna().unique().tolist())
    commodity_sel = st.selectbox("Commodity", options=comm_opts, index=0)

    # Organic selector
    org_opts = ["All", "Organic", "Conventional"]
    org_sel = st.selectbox("Organic?", options=org_opts, index=0)

    # Location filter (allow All)
    loc_opts = ["All"] + sorted(sdf_f["location_disp"].dropna().unique().tolist())
    loc_sel = st.selectbox("Location", options=loc_opts, index=0)

    # Optional pre-filter by specific Vendor or Customer (both with All)
    if scope == "Vendors":
        base_opts = ["All"] + sorted(sdf_f["vendor_disp"].dropna().unique().tolist())
        base_sel = st.selectbox("Vendor (filter)", options=base_opts, index=0)
    else:
        base_opts = ["All"] + sorted(sdf_f["customer_disp"].dropna().unique().tolist())
        base_sel = st.selectbox("Customer (filter)", options=base_opts, index=0)

# Apply filters
if commodity_sel != "All":
    cm_norm = _normalize_txt(commodity_sel)
    sdf_f = sdf_f[sdf_f["commodity_c"] == cm_norm]

if org_sel != "All":
    want_org = (org_sel == "Organic")
    sdf_f = sdf_f[sdf_f["is_organic_bool"] == want_org]

if loc_sel != "All":
    loc_norm = _normalize_txt(loc_sel)
    sdf_f = sdf_f[sdf_f["location_c"] == loc_norm]

if base_sel != "All":
    base_norm = _normalize_txt(base_sel)
    key = "vendor_c" if scope == "Vendors" else "customer_c"
    sdf_f = sdf_f[sdf_f[key] == base_norm]

# KPI line
st.caption(
    f"Rows: **{len(sdf_f)}** | Vendors: **{sdf_f['vendor_c'].nunique()}** | Customers: **{sdf_f['customer_c'].nunique()}** | Products: **{sdf_f['product_c'].nunique()}**"
)

if sdf_f.empty:
    st.warning("No data for those filters / date range.")
    st.stop()

# ------------------------
# ENTITY LIST + SELECTION (Vendors or Customers)
# ------------------------
if scope == "Vendors":
    group_key = "vendor_c"
    label_col = "vendor_disp"
    label_name = "Vendor"
else:
    group_key = "customer_c"
    label_col = "customer_disp"
    label_name = "Customer"

if sdf_f["order_id"].notna().any():
    ent_summary = (
        sdf_f.groupby(group_key).agg(
            Entity=(label_col, lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""),
            Invoices=("order_id", "nunique"),
            Products=("product_c", "nunique"),
        ).reset_index(drop=True).sort_values(["Invoices", "Products"], ascending=[False, False])
    )
else:
    ent_summary = (
        sdf_f.groupby(group_key).agg(
            Entity=(label_col, lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""),
            Invoices=("product_c", "count"),  # row proxy
            Products=("product_c", "nunique"),
        ).reset_index(drop=True).sort_values(["Invoices", "Products"], ascending=[False, False])
    )

left, right = st.columns([1, 1])
with left:
    st.subheader(f"🏷️ {label_name}s (by filters)")
    st.dataframe(
        _styled_table(ent_summary, {"Invoices": "{:,.0f}", "Products": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

with right:
    ent_opts = [""] + ent_summary["Entity"].tolist()
    ent_sel = st.selectbox(f"Pick a {label_name.lower()} to detail products", options=ent_opts, index=0, placeholder="Select...")

if not ent_sel:
    st.info(f"Select a {label_name.lower()} to see products and invoices.")
    st.stop()

ent_norm = _normalize_txt(ent_sel)
key = "vendor_c" if scope == "Vendors" else "customer_c"
vslice = sdf_f[sdf_f[key] == ent_norm].copy()

# ------------------------
# PRODUCT → COUNT(Invoice) for selected entity
# ------------------------
st.subheader(f"🧺 Products sold to {ent_sel if scope == 'Customers' else 'from ' + ent_sel}")

if vslice["order_id"].notna().any():
    prod_pvt = (
        vslice.groupby("product_c").agg(
            Product=("product_disp", lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""),
            Invoices=("order_id", "nunique"),
            Units=("quantity", "sum"),
            Revenue=("total_revenue", "sum"),
        ).reset_index(drop=True)
    )
else:
    prod_pvt = (
        vslice.groupby("product_c").agg(
            Product=("product_disp", lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""),
            Invoices=("product_c", "count"),  # row proxy
            Units=("quantity", "sum"),
            Revenue=("total_revenue", "sum"),
        ).reset_index(drop=True)
    )

prod_pvt = prod_pvt.sort_values(["Invoices", "Revenue"], ascending=[False, False])

st.dataframe(
    _styled_table(prod_pvt, {"Invoices": "{:,.0f}", "Units": "{:,.0f}", "Revenue": "${:,.0f}"}),
    use_container_width=True,
    hide_index=True,
)

# Optional chart
if ALTAIR_OK and len(prod_pvt) > 0:
    chart = alt.Chart(prod_pvt.head(25)).mark_bar().encode(
        x=alt.X("Invoices:Q", title="Count of Invoice #"),
        y=alt.Y("Product:N", sort="-x"),
        tooltip=["Product", "Invoices", "Units", "Revenue"],
    ).properties(height=420)
    st.altair_chart(chart, use_container_width=True)

# ------------------------
# EXPORTS
# ------------------------
summary_csv = ent_summary.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Summary (entities)", data=summary_csv, file_name=f"{scope.lower()}_by_filters.csv", mime="text/csv")

detail_csv = prod_pvt.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Products by entity", data=detail_csv, file_name=f"products_by_{scope.lower()}_{ent_norm or 'all'}.csv", mime="text/csv")
