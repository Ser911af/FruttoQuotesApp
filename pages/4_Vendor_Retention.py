# pages/4_Vendor_Retention.py
# Vendor Retention Rankings — by Vendor and by Buyer Assigned

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
st.set_page_config(page_title="Vendor Retention Rankings", page_icon="🏭", layout="wide")
st.title("🏭 Vendor Retention Rankings & Buyers Assigned")

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

# ------------------------
# SALES LOADER
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
        "most_recent_invoice_paid_date": ["most_recent_invoice_paid_date"],
        "received_date": ["received_date"],
        "created_at": ["created_at"],
        # IDs
        "invoice_num": ["invoice_num", "invoice #", "invoice"],
        "sales_order": ["sales_order"],
        # Dimensions
        "vendor": ["vendor", "shipper", "supplier"],
        "buyer_assigned": ["buyer_assigned", "buyer_asigned", "buyer"],
        "customer": ["customer"],
        "sales_rep": ["sales_rep", "cus_sales_rep"],
        "lot_location": ["lot_location"],
        # Metrics
        "quantity": ["quantity", "qty"],
        "total_revenue": ["total_revenue"],
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
    for c in ["reqs_date", "most_recent_invoice_paid_date", "received_date", "created_at"]:
        sdf[c] = pd.to_datetime(sdf[c], errors="coerce")
    for c in ["quantity", "total_revenue", "price_per_unit"]:
        sdf[c] = pd.to_numeric(sdf[c], errors="coerce")

    # Canonical normalized keys
    sdf["vendor_c"] = sdf["vendor"].astype(str).map(_normalize_txt)
    sdf["buyer_c"]  = sdf["buyer_assigned"].astype(str).map(_normalize_txt)

    # Display labels
    sdf["vendor_disp"] = sdf["vendor"].astype(str).str.title()
    sdf["buyer_disp"]  = sdf["buyer_assigned"].astype(str).str.title()

    # ---------- PATCH: Date basis (default = Requested Date / Silo) ----------
    sdf["date_requested"] = sdf["reqs_date"]
    sdf["date_paid"]      = sdf["most_recent_invoice_paid_date"]
    sdf["date_received"]  = sdf["received_date"]
    sdf["date_created"]   = sdf["created_at"]
    sdf["date"]           = sdf["date_requested"]  # default
    # ------------------------------------------------------------------------

    # ---------- PATCH: order_id prefer sales_order (fallback invoice_num) ----
    order_id = np.where(sdf["sales_order"].notna(), sdf["sales_order"].astype(str),
               np.where(sdf["invoice_num"].notna(), sdf["invoice_num"].astype(str), np.nan))
    order_id = pd.Series(order_id)
    order_id = order_id.mask(order_id.str.lower().isin(["nan", "none", ""]))
    sdf["order_id"] = order_id
    # ------------------------------------------------------------------------

    return sdf

# ------------------------
# METRICS CORE
# ------------------------
def _week_key(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    iso = ts.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"

def _winsorized_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0)
    # soft winsorization at 2–98%
    lo, hi = x.quantile(0.02), x.quantile(0.98)
    xw = x.clip(lower=lo, upper=hi)
    mu, sd = xw.mean(), xw.std(ddof=0)
    return (x - mu) / (sd + 1e-9)

def make_vendor_metrics(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Compute per-vendor retention metrics inside [start, end).
    Recency is measured vs `as_of` (adaptive).
    """
    if df.empty:
        return pd.DataFrame()

    d = df[(df["date"] >= start) & (df["date"] < end)].copy()
    if d.empty:
        return pd.DataFrame()

    d["week_key"] = d["date"].apply(_week_key)

    # Orders: group by order_id if present; else by row count.
    if d["order_id"].notna().any():
        orders = d.groupby(["vendor_c", "order_id"], dropna=False).agg(
            order_revenue=("total_revenue", "sum")
        ).reset_index()

        # ---------- PATCH: excluir pedidos $0 (cancelados/ajustes) ----------
        orders = orders[orders["order_revenue"].fillna(0) != 0]
        # --------------------------------------------------------------------

        orders_per_vendor = orders.groupby("vendor_c").agg(
            n_orders=("order_id", "nunique"),
            aov=("order_revenue", "mean"),
        )
    else:
        orders_per_vendor = d.groupby("vendor_c").agg(
            n_orders=("date", "count"),
            aov=("total_revenue", "mean"),
        )

    agg = d.groupby("vendor_c").agg(
        total_revenue=("total_revenue", "sum"),
        total_qty=("quantity", "sum"),
        last_sale=("date", "max"),
        first_sale=("date", "min"),
        active_weeks=("week_key", "nunique"),
    )

    # span weeks (>=1)
    span_days = (agg["last_sale"].dt.normalize() - agg["first_sale"].dt.normalize()).dt.days.clip(lower=0).fillna(0)
    agg["weeks_span"] = (span_days // 7) + 1
    agg["regularity_ratio"] = (agg["active_weeks"] / agg["weeks_span"]).clip(upper=1.0)

    # Adaptive recency
    agg["recency_days"] = (as_of.normalize() - agg["last_sale"].dt.normalize()).dt.days

    agg = agg.join(orders_per_vendor, how="left").fillna({"aov": 0, "n_orders": 0})

    # z-scores with winsorization
    agg["z_rev"] = _winsorized_z(agg["total_revenue"])  # 35%
    agg["z_freq"] = _winsorized_z(agg["n_orders"])      # 25%
    agg["z_rec"] = _winsorized_z(-agg["recency_days"].fillna(365))  # invert recency, 25%
    agg["z_reg"] = _winsorized_z(agg["regularity_ratio"])          # 15%

    agg["retention_score"] = (
        0.35 * agg["z_rev"] + 0.25 * agg["z_freq"] + 0.25 * agg["z_rec"] + 0.15 * agg["z_reg"]
    )

    out = (
        agg.reset_index()
        .rename(columns={"vendor_c": "vendor"})
        .sort_values(["retention_score", "total_revenue"], ascending=[False, False])
    )

    out = out[[  # expected output order
        "vendor", "total_revenue", "n_orders", "aov", "total_qty",
        "last_sale", "recency_days", "active_weeks", "weeks_span", "regularity_ratio", "retention_score"
    ]]
    return out

def make_buyer_rankings(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Rank buyers using (buyer, vendor) pairs within [start, end).
    Composite Rank Score = 50% Avg Retention (vendor-based), 30% revenue percentile, 20% active vendors percentile.
    """
    if df.empty:
        return pd.DataFrame()

    d = df[(df["date"] >= start) & (df["date"] < end)].copy()
    if d.empty:
        return pd.DataFrame()

    # Totals per (buyer, vendor)
    bv = (
        d.groupby(["buyer_c", "vendor_c"], dropna=False)
        .agg(rev=("total_revenue", "sum"), qty=("quantity", "sum"))
        .reset_index()
    )
    bv = bv[bv["buyer_c"].notna() & (bv["buyer_c"] != "")]

    # Pair spans
    d2 = d.copy()
    d2["week_key"] = d2["date"].apply(_week_key)
    pair_span = (
        d2.groupby(["buyer_c", "vendor_c"], dropna=False)
        .agg(
            last_sale=("date", "max"),
            first_sale=("date", "min"),
            active_weeks=("week_key", "nunique"),
        )
        .reset_index()
    )
    span_days = (pair_span["last_sale"].dt.normalize() - pair_span["first_sale"].dt.normalize()).dt.days.clip(lower=0)
    pair_span["weeks_span"] = (span_days // 7) + 1
    pair_span["regularity_pair"] = (pair_span["active_weeks"] / pair_span["weeks_span"]).clip(upper=1.0)
    pair_span["recency_days_pair"] = (as_of.normalize() - pair_span["last_sale"].dt.normalize()).dt.days

    bv2 = bv.merge(pair_span, on=["buyer_c", "vendor_c"], how="left")
    bv2["w"] = bv2["rev"].clip(lower=0) + 1e-9

    # Vendor retention scores within the range
    vend_all = make_vendor_metrics(d, start, end, as_of).rename(columns={"vendor": "vendor_c"})
    bv3 = bv2.merge(vend_all[["vendor_c", "retention_score"]], on="vendor_c", how="left")

    # Aggregate to buyer level
    by_buyer = (
        bv3.groupby("buyer_c", dropna=False)
        .apply(lambda g: pd.Series({
            "active_vendors": g["vendor_c"].nunique(),
            "total_revenue": g["rev"].sum(),
            "avg_retention_score": np.average(g["retention_score"].fillna(0), weights=g["w"]),
            "avg_regularity": np.average(g["regularity_pair"].fillna(0), weights=g["w"]),
            "avg_recency_days": np.average(
                g["recency_days_pair"].fillna(
                    g["recency_days_pair"].median() if np.isfinite(np.nanmedian(g["recency_days_pair"])) else 0
                ),
                weights=g["w"]
            ),
        }))
        .reset_index()
    )

    if by_buyer.empty:
        return by_buyer

    by_buyer["rank_score"] = (
        0.5 * by_buyer["avg_retention_score"].fillna(0) +
        0.3 * by_buyer["total_revenue"].rank(pct=True, method="average") +
        0.2 * by_buyer["active_vendors"].rank(pct=True, method="average")
    )

    by_buyer = by_buyer.sort_values(["rank_score", "total_revenue"], ascending=[False, False]).reset_index(drop=True)
    return by_buyer

def vendors_by_buyer(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, as_of: pd.Timestamp, buyer: str) -> pd.DataFrame:
    if not buyer:
        return pd.DataFrame()
    bnorm = _normalize_txt(buyer)
    d = df[(df["date"] >= start) & (df["date"] < end) & (df["buyer_c"] == bnorm)].copy()
    if d.empty:
        return pd.DataFrame()
    out = make_vendor_metrics(d, start, end, as_of)
    return out.sort_values(["retention_score", "total_revenue"], ascending=[False, False]).reset_index(drop=True)

# ------------------------
# SIDEBAR FILTERS (date range + date basis)
# ------------------------
with st.sidebar:
    st.subheader("Filters")

    # NEW: Date basis selector
    basis = st.radio(
        "Date basis",
        options=["Requested date (Silo)", "Paid date", "Received date", "Created date"],
        index=0
    )

    today_bo = pd.Timestamp.now(tz="America/Bogota").normalize()
    default_start = (today_bo - pd.Timedelta(days=90)).tz_localize(None)
    default_end = today_bo.tz_localize(None)

    date_range = st.date_input("Date range", value=(default_start, default_end))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)  # end exclusive
    else:
        start_date, end_date = default_start, default_end + pd.Timedelta(days=1)

# ✅ Adaptive as_of_date
today_norm = pd.Timestamp.now(tz="America/Bogota").normalize().tz_localize(None)
end_minus_1 = (end_date - pd.Timedelta(days=1)).normalize()
as_of_date = min(today_norm, end_minus_1)
st.caption(f"Recency as-of: **{as_of_date.date()}**")

# ------------------------
# DATA
# ------------------------
sdf = load_sales()

# Apply chosen date basis
if basis == "Paid date":
    sdf["date"] = sdf["date_paid"]
elif basis == "Received date":
    sdf["date"] = sdf["date_received"]
elif basis == "Created date":
    sdf["date"] = sdf["date_created"]
else:
    sdf["date"] = sdf["date_requested"]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sales records", value=len(sdf))
with col2:
    st.metric("Unique vendors", value=int(sdf["vendor_c"].nunique()) if not sdf.empty else 0)
with col3:
    st.metric("Unique buyers assigned", value=int(sdf["buyer_c"].nunique()) if not sdf.empty else 0)

if sdf.empty:
    st.stop()

mask = (sdf["date"] >= start_date) & (sdf["date"] < end_date)
sdf_f = sdf[mask].copy()
st.caption(f"Filtered rows: {len(sdf_f)} in selected range.")
if sdf_f.empty:
    st.warning("No data in the selected range.")
    st.stop()

# ------------------------
# EXPLANATIONS (anonymous)
# ------------------------
with st.expander("🏭 Vendor Retention Rankings — definición de métricas", expanded=False):
    st.markdown(
        """
**Retention Score**  
Combina 4 factores en un puntaje único (mismo esquema que *Customers*):  
- Revenue (35%) → `total_revenue` (z-score con winsorización 2–98%).  
- Frequency (25%) → `n_orders` (z-score).  
- Recency (25%) → días desde `last_sale` (z-score invertido).  
- Regularity (15%) → `active_weeks / weeks_span` (densidad semanal, acotada a 1.0).

**Notas**  
- El rango de fechas es **inicio inclusivo** y **fin exclusivo** (+1 día internamente).  
- La recencia se mide respecto a la **fecha de referencia** (*as-of date*).  
- Los nombres permanecen anónimos (solo etiquetas de Vendor/Buyer).  
- Ordenamiento numérico correcto preservando tipos mediante `Styler.format`.
        """
    )

with st.expander("🧭 Columnas y su interpretación (Buyer)", expanded=False):
    st.markdown(
        """
**Buyer Assigned**  
Etiqueta normalizada (Title Case para mostrar).  

**Active Vendors**  
Número de *vendors* distintos que vendieron en el rango. 

**Total Revenue**  
Suma de `total_revenue` del buyer en el rango. 

**Avg Retention Score**  
Promedio ponderado por revenue del *Retention Score* de cada vendor atendido. 

**Avg Regularity**  
Promedio ponderado del `regularity_pair` (semanas activas / semanas en el período) a nivel buyer–vendor. 

**Avg Recency Days**  
Promedio ponderado de días desde la última venta (menor es mejor). 

**Rank Score**  
Composición para ordenar buyers: 50% retención, 30% percentil de revenue, 20% percentil de vendors activos.
        """
    )

# ------------------------
# SECTION 1 — VENDOR RANKINGS
# ------------------------
st.subheader("🏭 Vendor Retention Rankings")
vendor_rank = make_vendor_metrics(sdf_f, start_date, end_date, as_of_date)
if vendor_rank.empty:
    st.info("Not enough data to compute vendor retention metrics.")
else:
    vshow = vendor_rank.copy()
    # map display labels
    disp_map = sdf_f[["vendor_c", "vendor_disp"]].drop_duplicates().rename(columns={"vendor_c": "vendor"})
    vshow = vshow.merge(disp_map, on="vendor", how="left")
    vshow["vendor"] = vshow["vendor_disp"].fillna(vshow["vendor"].str.title())
    vshow = vshow.drop(columns=["vendor_disp"], errors="ignore")

    vshow = _fmt_dates(vshow, ["last_sale"])  # first_sale no requerido en columnas de salida
    vshow = _round_cols(vshow, {"retention_score": 3})

    v_cols = [
        "vendor", "total_revenue", "n_orders", "aov", "total_qty",
        "last_sale", "recency_days", "active_weeks", "weeks_span", "regularity_ratio", "retention_score",
    ]
    vshow = vshow[v_cols]
    v_disp = _title_cols(vshow)

    v_styles = {
        "Total Revenue": "${:,.0f}",
        "N Orders": "{:,.0f}",
        "Aov": "${:,.2f}",
        "Total Qty": "{:,.0f}",
        "Recency Days": "{:,.0f}",
        "Active Weeks": "{:,.0f}",
        "Weeks Span": "{:,.0f}",
        "Regularity Ratio": "{:.2%}",
        "Retention Score": "{:.3f}",
    }
    st.dataframe(_styled_table(v_disp, v_styles), use_container_width=True, hide_index=True)

    if ALTAIR_OK and len(vshow) > 0:
        chart = alt.Chart(vshow.head(25)).mark_bar().encode(
            x=alt.X("retention_score:Q", title="Retention Score"),
            y=alt.Y("vendor:N", sort="-x", title="Vendor"),
            tooltip=["vendor", "total_revenue", "n_orders", "recency_days", "regularity_ratio", "retention_score"],
        ).properties(height=500)
        st.altair_chart(chart, use_container_width=True)

# ------------------------
# SECTION 2 — BUYER RANKINGS
# ------------------------
st.subheader("🧑‍💼 Buyer Retention Rankings")
buyer_rank = make_buyer_rankings(sdf_f, start_date, end_date, as_of_date)
if buyer_rank.empty:
    st.info("Not enough data to compute buyer rankings.")
else:
    bshow = buyer_rank.copy()
    # display labels
    disp_map_b = sdf_f[["buyer_c", "buyer_disp"]].drop_duplicates().rename(columns={"buyer_c": "buyer"})
    bshow = bshow.merge(disp_map_b, left_on="buyer_c", right_on="buyer", how="left")
    bshow["buyer_c"] = bshow["buyer_disp"].fillna(bshow["buyer_c"].str.title())
    bshow = bshow.drop(columns=["buyer" ,"buyer_disp"], errors="ignore").rename(columns={"buyer_c": "buyer_assigned"})

    bshow = _round_cols(bshow, {
        "avg_retention_score": 4,  # will display as %
        "avg_recency_days": 1,
        "rank_score": 6,
    })

    b_cols = ["buyer_assigned", "active_vendors", "total_revenue", "avg_retention_score", "avg_regularity", "avg_recency_days", "rank_score"]
    bshow = bshow[b_cols]
    b_disp = _title_cols(bshow)

    b_styles = {
        "Active Vendors": "{:,.0f}",
        "Total Revenue": "${:,.0f}",
        "Avg Retention Score": "{:.2%}",
        "Avg Regularity": "{:.2%}",
        "Avg Recency Days": "{:,.1f}",
        "Rank Score": "{:.6f}",
    }
    st.dataframe(_styled_table(b_disp, b_styles), use_container_width=True, hide_index=True)

    if ALTAIR_OK and len(bshow) > 0:
        chart2 = alt.Chart(bshow.head(20)).mark_bar().encode(
            x=alt.X("rank_score:Q", title="Rank Score"),
            y=alt.Y("buyer_assigned:N", sort="-x", title="Buyer Assigned"),
            tooltip=["buyer_assigned", "active_vendors", "total_revenue", "avg_retention_score", "avg_regularity", "avg_recency_days", "rank_score"],
        ).properties(height=420)
        st.altair_chart(chart2, use_container_width=True)

# ------------------------
# SECTION 3 — VENDORS BY BUYER
# ------------------------
st.subheader("🔎 Vendors by Buyer")
by_map = sdf_f[["buyer_c", "buyer_disp"]].drop_duplicates()
by_map = by_map[by_map["buyer_c"].notna() & (by_map["buyer_c"] != "")]
by_options = [""] + sorted(by_map["buyer_disp"].dropna().unique().tolist())
by_sel_disp = st.selectbox("Choose a Buyer Assigned", options=by_options, index=0, placeholder="Select...")

if by_sel_disp:
    buyer_norm = by_map.loc[by_map["buyer_disp"] == by_sel_disp, "buyer_c"].iloc[0]
    vb = vendors_by_buyer(sdf_f, start_date, end_date, as_of_date, buyer_norm)
    if vb.empty:
        st.info("This Buyer has no vendors in the selected range.")
    else:
        disp_map_v = sdf_f[["vendor_c", "vendor_disp"]].drop_duplicates().rename(columns={"vendor_c": "vendor"})
        vb = vb.merge(disp_map_v, on="vendor", how="left")
        vb["vendor"] = vb["vendor_disp"].fillna(vb["vendor"].str.title())
        vb = vb.drop(columns=["vendor_disp"], errors="ignore")

        vb = _fmt_dates(vb, ["last_sale"])  # keep only last_sale for display
        vb = _round_cols(vb, {"retention_score": 3})

        vb_cols = [
            "vendor", "total_revenue", "n_orders", "aov", "last_sale", "recency_days",
            "active_weeks", "weeks_span", "regularity_ratio", "retention_score",
        ]
        vb = vb[vb_cols]
        vb_disp = _title_cols(vb)

        vb_styles = {
            "Total Revenue": "${:,.0f}",
            "N Orders": "{:,.0f}",
            "Aov": "${:,.2f}",
            "Recency Days": "{:,.0f}",
            "Active Weeks": "{:,.0f}",
            "Weeks Span": "{:,.0f}",
            "Regularity Ratio": "{:.2%}",
            "Retention Score": "{:.3f}",
        }
        st.dataframe(_styled_table(vb_disp, vb_styles), use_container_width=True, hide_index=True)

# ------------------------
# NOTES
# ------------------------
st.markdown(f"""
**Notes**
- `Recency` measured **as of {as_of_date.date()}** (adaptive: today if the range reaches today; otherwise range end - 1 day).
- Percentages are **display formats**; underlying numeric values are kept for correct sorting.
- Regularity/Frequency are computed strictly **within the selected range**.
- **Date basis**: por defecto *Requested date (Silo)*. Puedes alternar a *Paid*, *Received* o *Created* desde el panel lateral.
- **Aggregation**: KPIs usan **sales_order** como identificador de pedido (fallback a invoice) y **excluyen órdenes de $0**.
""")
