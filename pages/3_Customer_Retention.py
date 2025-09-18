# pages/3_Customer_Retention.py
# Customer Retention Rankings — by Company and by Sales Rep
# Keeps: Login + sales DB access (Supabase/Azure) with schema similar to `ventas_frutto`

import re
import math
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ✅ Login (required before loading data)
from simple_auth import ensure_login, logout_button

user = ensure_login()
with st.sidebar:
    logout_button()
st.caption(f"Session: {user}")

# Optional: Altair for charts
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

# Optional: Supabase
try:
    from supabase import create_client  # pip install supabase
except Exception:
    create_client = None

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Customer Retention Rankings", page_icon="📈", layout="wide")
st.title("📈 Customer Retention Rankings")

with st.expander("What is the Retention Score?", expanded=True):
    st.markdown("""
- **Revenue (35%)**: `total_revenue` (z-score, light winsorization).
- **Frequency (25%)**: `n_orders` (z-score).
- **Recency (25%)**: days since `last_sale` (z-score, inverted).
- **Regularity (15%)**: `active_weeks / weeks_span` (weekly purchase density).

Date range is **start inclusive, end exclusive** (we add +1 day to the end date).  
Location filter uses **Lot Location**.
""")

# ------------------------
# HELPERS (logic)
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
    sec = st.secrets.get(secret_key, None)
    if not sec or not create_client:
        return None
    url = sec.get("url")
    key = sec.get("anon_key")
    if not url or not key:
        return None
    return create_client(url, key)

# ---------- DISPLAY HELPERS (UI only; do not affect calculations) ----------
def _title_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to Title Case for display."""
    df = df.copy()
    mapping = {c: c.replace("_", " ").title() for c in df.columns}
    return df.rename(columns=mapping)

def _fmt_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert date columns to 'YYYY-MM-DD' for display (empty if NaT)."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            out = s.dt.strftime("%Y-%m-%d")
            df[c] = out.fillna("")
    return df

def _round_cols(df: pd.DataFrame, spec: dict[str, int]) -> pd.DataFrame:
    """Round numeric columns according to {col: decimals}."""
    df = df.copy()
    for c, n in spec.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(n)
    return df

def _styled_table(df_disp: pd.DataFrame, styles: dict) -> "pd.io.formats.style.Styler":
    """
    Return a pandas Styler with per-column formats.
    Column keys must match the **display** names (after Title Case).
    """
    return df_disp.style.format(styles, na_rep="")

# ------------------------
# SALES LOADER (ventas_frutto)
# ------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_sales() -> pd.DataFrame:
    """
    Read sales from Supabase (preferred) or MSSQL (hooks only).
    Normalizes to snake_case columns aligned with `ventas_frutto`.
    """
    # 1) Supabase
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
            offset += got  # advance by what we actually received
        df = pd.DataFrame(rows)
    else:
        # 2) MSSQL (optional)
        ms = st.secrets.get("mssql_sales", None)
        if ms:
            st.warning("MSSQL loader not implemented here. Use sqlalchemy/pyodbc and return a DataFrame.")
            return pd.DataFrame()
        else:
            st.error("No sales source found (st.secrets['supabase_sales'] or ['mssql_sales']).")
            return pd.DataFrame()

    if df.empty:
        return df

    # Alias → standard
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
        "sale_location": ["sale_location"],
        "sales_rep": ["sales_rep", "cus_sales_rep"],
        "customer": ["customer"],
        "lot": ["lot"],
        "vendor": ["vendor", "shipper", "supplier"],
        "source": ["source"],
        "lot_location": ["lot_location"],
        "oficina": ["oficina"],  # kept for completeness
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

    # Casts
    for c in ["reqs_date","most_recent_invoice_paid_date","received_date","use_by_date","pack_date","created_at"]:
        sdf[c] = pd.to_datetime(sdf[c], errors="coerce")
    for c in ["quantity","cost_per_unit","price_per_unit","total_cost",
              "total_sold_lot_expenses","total_revenue","total_profit_usd","total_profit_pct"]:
        sdf[c] = pd.to_numeric(sdf[c], errors="coerce")

    # Normalized keys for logic
    sdf["customer_c"]       = sdf["customer"].astype(str).map(_normalize_txt)
    sdf["sales_rep_c"]      = sdf["sales_rep"].astype(str).map(_normalize_txt)
    sdf["sale_location_c"]  = sdf["sale_location"].astype(str).map(_normalize_txt)
    sdf["lot_location_c"]   = sdf["lot_location"].astype(str).map(_normalize_txt)
    sdf["is_organic"]       = sdf["organic"].map(_coerce_bool)

    # Display names (title case) for UI only
    sdf["customer_disp"]  = sdf["customer"].astype(str).str.title()
    sdf["sales_rep_disp"] = sdf["sales_rep"].astype(str).str.title()

    # Operational date (prefer received_date > reqs_date)
    sdf["date"] = sdf["received_date"].fillna(sdf["reqs_date"])
    sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")

    # Order identifier (if any)
    sdf["order_id"] = sdf["invoice_num"].astype(str)
    sdf.loc[sdf["order_id"].isin(["nan","none",""]), "order_id"] = np.nan

    return sdf

# ------------------------
# RETENTION METRICS
# ------------------------
def _week_key(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    iso = ts.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"

def make_retention_metrics(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    RFM-like + regularity per customer in [start, end).
    Regularity = active_weeks / weeks_span between first and last sale in range.
    """
    if df.empty:
        return pd.DataFrame()

    d = df[(df["date"] >= start) & (df["date"] < end)].copy()
    if d.empty:
        return pd.DataFrame()

    d["week_key"] = d["date"].apply(_week_key)

    # Orders by invoice_num if available, else use rows as proxy
    if d["order_id"].notna().any():
        orders = d.groupby(["customer_c","order_id"], dropna=False).agg(
            order_revenue=("total_revenue","sum")
        ).reset_index()
        orders_per_customer = orders.groupby("customer_c").agg(
            n_orders=("order_id","nunique"),
            aov=("order_revenue","mean")
        )
    else:
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

    # Weeks span (>=1)
    span_weeks = []
    for _, row in agg.iterrows():
        f = row["first_sale"]
        l = row["last_sale"]
        if pd.isna(f) or pd.isna(l):
            span_weeks.append(1)
        else:
            days = max(0, (l.normalize() - f.normalize()).days)
            span_weeks.append(max(1, (days // 7) + 1))
    agg["weeks_span"] = span_weeks
    agg["regularity_ratio"] = (agg["active_weeks"] / agg["weeks_span"]).clip(upper=1.0)

    # Recency (days since last sale) normalized to date (no time-of-day drift)
    today_norm = pd.Timestamp.now(tz="America/Bogota").normalize().tz_localize(None)
    agg["recency_days"] = (today_norm - agg["last_sale"].dt.normalize()).dt.days

    # Join orders/AOV
    agg = agg.join(orders_per_customer, how="left")
    agg["aov"] = agg["aov"].fillna(0)
    agg["n_orders"] = agg["n_orders"].fillna(0)

    # Z-scores with light winsorization
    def z(x):
        x_ = x.clip(lower=x.quantile(0.02), upper=x.quantile(0.98))
        mu, sd = x_.mean(), x_.std(ddof=0)
        return (x - mu) / (sd + 1e-9)

    agg["z_rev"]      = z(agg["total_revenue"].fillna(0))
    agg["z_freq"]     = z(agg["n_orders"].fillna(0))
    agg["z_recency"]  = z((-agg["recency_days"].fillna(365)))  # smaller recency_days is better
    agg["z_regular"]  = z(agg["regularity_ratio"].fillna(0))

    # Composite score (tunable)
    agg["retention_score"] = (
        0.35*agg["z_rev"] +
        0.25*agg["z_freq"] +
        0.25*agg["z_recency"] +
        0.15*agg["z_regular"]
    )

    agg = agg.reset_index().rename(columns={"customer_c":"customer"})
    cols = [
        "customer","total_revenue","n_orders","aov","total_qty",
        "first_sale","last_sale","recency_days","active_weeks","weeks_span","regularity_ratio",
        "retention_score"
    ]
    agg = agg[cols].sort_values(["retention_score","total_revenue"], ascending=[False, False])

    return agg

# ---- Pair-level metrics for (rep, customer) in rep ranking ----
def make_salesrep_rankings(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Sales Rep ranking using ALL customers they touched in the range.
    Uses pair-level (rep, customer) metrics for recency/regularity and
    revenue-weighted averages across pairs.
    """
    if df.empty:
        return pd.DataFrame()

    d = df[(df["date"] >= start) & (df["date"] < end)].copy()
    if d.empty:
        return pd.DataFrame()

    # Base revenue/qty per pair in range
    rep_cust = (d.groupby(["sales_rep_c", "customer_c"], dropna=False)
                  .agg(rev=("total_revenue","sum"),
                       qty=("quantity","sum"))
                  .reset_index())

    # Exclude blanks
    rep_cust = rep_cust[rep_cust["sales_rep_c"].notna() & (rep_cust["sales_rep_c"] != "")]

    # Pair-level recency & regularity
    d2 = d.copy()
    d2["week_key"] = d2["date"].apply(_week_key)

    pair_span = (d2.groupby(["sales_rep_c","customer_c"], dropna=False)
                   .agg(
                       last_sale=("date","max"),
                       first_sale=("date","min"),
                       active_weeks=("week_key","nunique"),
                   )
                   .reset_index())

    # weeks_span (>=1)
    span_days = (pair_span["last_sale"].dt.normalize() - pair_span["first_sale"].dt.normalize()).dt.days.clip(lower=0)
    weeks_span = (span_days // 7) + 1
    pair_span["weeks_span"] = weeks_span
    pair_span["regularity_pair"] = (pair_span["active_weeks"] / pair_span["weeks_span"]).clip(upper=1.0)

    today_norm = pd.Timestamp.now(tz="America/Bogota").normalize().tz_localize(None)
    pair_span["recency_days_pair"] = (today_norm - pair_span["last_sale"].dt.normalize()).dt.days

    # Merge pair metrics into revenue matrix
    rc = rep_cust.merge(pair_span, on=["sales_rep_c","customer_c"], how="left")
    rc = rc.rename(columns={"sales_rep_c": "sales_rep"})
    rc["w"] = rc["rev"].clip(lower=0) + 1e-9

    # Revenue-weighted averages per rep
    grp = (rc.groupby("sales_rep", dropna=False)
             .apply(lambda g: pd.Series({
                 "avg_regularity":   np.average(g["regularity_pair"].fillna(0), weights=g["w"]),
                 "avg_recency_days": np.average(
                     g["recency_days_pair"].fillna(
                         g["recency_days_pair"].median() if np.isfinite(np.nanmedian(g["recency_days_pair"])) else 0
                     ),
                     weights=g["w"]
                 )
             }))
             .reset_index())

    # Totals per rep  ✅ (usar sales_rep_c aquí y renombrar)
    rep_totals = (rep_cust.groupby("sales_rep_c", dropna=False)
                    .agg(total_revenue=("rev","sum"),
                         total_qty=("qty","sum"),
                         active_customers=("customer_c","nunique"))
                    .reset_index()
                    .rename(columns={"sales_rep_c": "sales_rep"}))

    # avg_retention_score as revenue-weighted average of customer scores (range-level)
    cust_all = make_retention_metrics(d, start, end).rename(columns={"customer":"customer_c"})
    rc2 = rc.merge(cust_all[["customer_c","retention_score"]], on="customer_c", how="left")
    by_rep_ret = (rc2.groupby("sales_rep", dropna=False)
                    .apply(lambda g: pd.Series({
                        "avg_retention_score": np.average(g["retention_score"].fillna(0), weights=g["w"])
                    }))
                    .reset_index())

    rep = (rep_totals
           .merge(grp, on="sales_rep", how="left")
           .merge(by_rep_ret, on="sales_rep", how="left")
           .fillna({"avg_regularity":0, "avg_recency_days":0, "avg_retention_score":0}))

    # Composite rank (tweak as needed)
    rep["rank_score"] = (
        0.5 * rep["avg_retention_score"] +
        0.3 * rep["total_revenue"].rank(pct=True, method="average") +
        0.2 * rep["active_customers"].rank(pct=True, method="average")
    )

    rep = rep.sort_values(["rank_score","total_revenue"], ascending=[False, False]).reset_index(drop=True)
    return rep

# ---- Customers by Sales Rep: only that rep's own sales (pair-consistent) ----
def customers_by_rep(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, rep: str) -> pd.DataFrame:
    """
    Customer ranking for a given Sales Rep computed ONLY from that rep's own sales in [start, end).
    """
    if not rep:
        return pd.DataFrame()

    rep_norm = _normalize_txt(rep)
    d = df[(df["date"] >= start) & (df["date"] < end) & (df["sales_rep_c"] == rep_norm)].copy()
    if d.empty:
        return pd.DataFrame()

    out = make_retention_metrics(d, start, end)
    return out.sort_values(["retention_score","total_revenue"], ascending=[False, False]).reset_index(drop=True)

# ------------------------
# SIDEBAR FILTERS
# ------------------------
with st.sidebar:
    st.subheader("Filters")

    # Date range (default: last 90 days, Bogota timezone)
    today_bo = pd.Timestamp.now(tz="America/Bogota").normalize()
    default_start = (today_bo - pd.Timedelta(days=90)).tz_localize(None)
    default_end = today_bo.tz_localize(None)  # default end is "today"; +1 is applied after user picks

    date_range = st.date_input("Date range", value=(default_start, default_end))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)  # end exclusive
    else:
        start_date, end_date = default_start, default_end

    if start_date >= end_date:
        start_date, end_date = end_date - pd.Timedelta(days=1), end_date

    # Location filter uses LOT LOCATION as requested
    lot_location_filter = st.text_input("Lot Location (contains):", "")
    organic_only = st.checkbox("Organic only (OG)", value=False)

# ------------------------
# DATA
# ------------------------
sdf = load_sales()

# TOP METRICS
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sales records", value=len(sdf))
with col2:
    st.metric("Unique customers", value=int(sdf["customer_c"].nunique()) if not sdf.empty else 0)
with col3:
    st.metric("Sales records (filtered)", value=0)  # will update after mask

if sdf.empty:
    st.stop()

# Apply filters
mask = (sdf["date"] >= start_date) & (sdf["date"] < end_date)
if lot_location_filter:
    mask &= sdf["lot_location_c"].str.contains(_normalize_txt(lot_location_filter))
if organic_only:
    mask &= (sdf["is_organic"] == True)

sdf_f = sdf[mask].copy()
col3.metric("Sales records (filtered)", value=len(sdf_f))
st.caption(f"Filtered rows: {len(sdf_f)} in selected range.")

if sdf_f.empty:
    st.warning("No data in the selected range/filters.")
    st.stop()

# ------------------------
# CUSTOMER RANKING (numeric + Styler for formatting)
# ------------------------
st.subheader("🏢 Customer Retention Rankings")
cust_rank = make_retention_metrics(sdf_f, start_date, end_date)
if cust_rank.empty:
    st.info("Not enough data to compute customer retention metrics.")
else:
    cust_show = cust_rank.copy()

    # Attach display names (Title Case)
    disp_map = sdf_f[["customer_c","customer_disp"]].drop_duplicates().rename(columns={"customer_c":"customer"})
    cust_show = cust_show.merge(disp_map, on="customer", how="left")
    cust_show["customer"] = cust_show["customer_disp"].fillna(cust_show["customer"].str.title())
    cust_show = cust_show.drop(columns=["customer_disp"], errors="ignore")

    # Dates (strings) + keep numeric columns numeric for sorting
    cust_show = _fmt_dates(cust_show, ["first_sale", "last_sale"])
    cust_show = _round_cols(cust_show, {"retention_score": 3})

    # Order and sweet headers
    cust_cols = ["customer","total_revenue","n_orders","aov","total_qty",
                 "last_sale","recency_days","active_weeks","weeks_span","regularity_ratio","retention_score"]
    cust_show = cust_show[cust_cols]
    cust_show_disp = _title_cols(cust_show)

    cust_styles = {
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
    st.dataframe(_styled_table(cust_show_disp, cust_styles), use_container_width=True, hide_index=True)

    if ALTAIR_OK and len(cust_show) > 0:
        chart = alt.Chart(cust_show.head(25)).mark_bar().encode(
            x=alt.X("retention_score:Q", title="Retention Score"),
            y=alt.Y("customer:N", sort="-x", title="Customer"),
            tooltip=["customer","total_revenue","n_orders","recency_days","regularity_ratio","retention_score"]
        ).properties(height=500)
        st.altair_chart(chart, use_container_width=True)

# ------------------------
# SALES REP RANKING (numeric + Styler)
# ------------------------
st.subheader("🧑‍💼 Sales Rep Retention Rankings")
rep_rank = make_salesrep_rankings(sdf_f, start_date, end_date)
if rep_rank.empty:
    st.info("Not enough data to compute sales rep ranking.")
else:
    rep_show = rep_rank.copy()
    rep_show["sales_rep"] = rep_show["sales_rep"].astype(str).str.title()

    # Rounding (keep numeric)
    rep_show = _round_cols(rep_show, {
        "avg_retention_score": 3,
        "avg_recency_days": 1,
        "rank_score": 6
    })

    rep_cols = ["sales_rep","active_customers","total_revenue",
                "avg_retention_score","avg_regularity","avg_recency_days","rank_score"]
    rep_show = rep_show[rep_cols]
    rep_show_disp = _title_cols(rep_show)

    rep_styles = {
        "Active Customers": "{:,.0f}",
        "Total Revenue": "${:,.0f}",
        "Avg Retention Score": "{:.3f}",
        "Avg Regularity": "{:.2%}",
        "Avg Recency Days": "{:,.1f}",
        "Rank Score": "{:.6f}",
    }
    st.dataframe(_styled_table(rep_show_disp, rep_styles), use_container_width=True, hide_index=True)

    if ALTAIR_OK and len(rep_show) > 0:
        chart2 = alt.Chart(rep_show.head(20)).mark_bar().encode(
            x=alt.X("rank_score:Q", title="Rank Score"),
            y=alt.Y("sales_rep:N", sort="-x", title="Sales Rep"),
            tooltip=["sales_rep","active_customers","total_revenue","avg_retention_score","avg_regularity","avg_recency_days","rank_score"]
        ).properties(height=420)
        st.altair_chart(chart2, use_container_width=True)

# ------------------------
# CUSTOMERS BY SALES REP (pair-consistent, numeric + Styler)
# ------------------------
st.subheader("🔎 Customers by Sales Rep")

# Build mapping display -> normalized (Title Case options in the UI)
rep_map_df = sdf_f[["sales_rep_c","sales_rep_disp"]].drop_duplicates()
rep_map_df = rep_map_df[rep_map_df["sales_rep_c"].notna() & (rep_map_df["sales_rep_c"] != "")]
rep_options = [""] + sorted(rep_map_df["sales_rep_disp"].dropna().unique().tolist())
rep_sel_disp = st.selectbox("Choose a Sales Rep", options=rep_options, index=0, placeholder="Select...")

if rep_sel_disp:
    # Map display back to normalized key
    rep_norm = rep_map_df.loc[rep_map_df["sales_rep_disp"] == rep_sel_disp, "sales_rep_c"].iloc[0]

    cr = customers_by_rep(sdf_f, start_date, end_date, rep_norm)
    if cr.empty:
        st.info("This Sales Rep has no customers in the selected range.")
    else:
        # Attach display names
        disp_map = sdf_f[["customer_c","customer_disp"]].drop_duplicates().rename(columns={"customer_c":"customer"})
        cr = cr.merge(disp_map, on="customer", how="left")
        cr["customer"] = cr["customer_disp"].fillna(cr["customer"].str.title())
        cr = cr.drop(columns=["customer_disp"], errors="ignore")

        # Dates and rounding
        cr = _fmt_dates(cr, ["first_sale", "last_sale"])
        cr = _round_cols(cr, {"retention_score": 3})

        cr_cols = ["customer","total_revenue","n_orders","aov","last_sale","recency_days",
                   "active_weeks","weeks_span","regularity_ratio","retention_score"]
        cr = cr[cr_cols]
        cr_disp = _title_cols(cr)

        cr_styles = {
            "Total Revenue": "${:,.0f}",
            "N Orders": "{:,.0f}",
            "Aov": "${:,.2f}",
            "Recency Days": "{:,.0f}",
            "Active Weeks": "{:,.0f}",
            "Weeks Span": "{:,.0f}",
            "Regularity Ratio": "{:.2%}",
            "Retention Score": "{:.3f}",
        }
        st.dataframe(_styled_table(cr_disp, cr_styles), use_container_width=True, hide_index=True)

# ------------------------
# NOTES
# ------------------------
st.markdown("""
**Notes:**
- `Retention Score` combines revenue (0.35), frequency (0.25), recency (0.25), and weekly regularity (0.15).
- Date filter is inclusive on start and exclusive on end (we add +1 day to the end date).
- Location filter uses **Lot Location** as requested.
- Title Case is applied only for display; normalized lowercase keys drive the logic.
""")
