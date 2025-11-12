import re
from typing import Optional, Tuple

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
st.set_page_config(page_title="Revenue YoY • Months & Days", page_icon="📊", layout="wide")
st.title("📊 Revenue YoY — by Month and by Day")
st.caption(
    "Compare total revenue Year-over-Year. Switch between **Annual (by Month)** and **Monthly (by Day)** views. KPI cards show Profit %, #Orders (count of `source` rows when available), and Total Revenue (short format) for current vs prior period."
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


def _abbr(n: float) -> str:
    if n is None or pd.isna(n):
        return "$0"
    n = float(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1_000_000_000:
        val = f"{n/1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        val = f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        val = f"{n/1_000:.2f}K"
    else:
        val = f"{n:.0f}"
    return f"{sign}${val}"


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
        # Dimensions
        "customer": ["customer", "customer_name", "client", "cliente"],
        "product": ["product", "item", "item_name", "product_name", "desc", "description"],
        # Metrics
        "quantity": ["quantity", "qty"],
        "total_revenue": ["total_revenue", "total", "line_total", "revenue", "t_revenue", "trevenue"],
        # Optional profit & cost
        "profit": ["profit", "gross_profit", "gp", "margin_amount"],
        "total_cost": ["total_cost", "cost", "cogs"],
        # Orders counter source (for #orders as requested)
        "source": ["source", "po", "purchase_order", "sales_order", "order_ref"],
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
    for c in ["quantity", "total_revenue", "profit", "total_cost"]:
        if c in sdf.columns:
            sdf[c] = pd.to_numeric(sdf[c], errors="coerce")

    # Unified date
    sdf["date"] = sdf["received_date"].fillna(sdf["reqs_date"]).fillna(sdf["created_at"])  # fallback chain
    sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")

    # Derive profit if missing but cost present
    if "profit" not in sdf.columns or sdf["profit"].isna().all():
        if "total_cost" in sdf.columns and not sdf["total_cost"].isna().all():
            sdf["profit"] = (sdf["total_revenue"] - sdf["total_cost"])  # may be NaN for rows
        else:
            sdf["profit"] = np.nan

    # Profit % (row level) — safe division
    sdf["profit_pct"] = np.where(
        sdf["total_revenue"].abs() > 1e-9,
        sdf["profit"] / sdf["total_revenue"],
        np.nan,
    )

    # Orders proxy if `source` missing → fall back to row count
    if "source" not in sdf.columns:
        sdf["source"] = np.nan

    return sdf


# ------------------------
# SIDEBAR
# ------------------------
with st.sidebar:
    st.subheader("Controls")

    # View mode
    view_mode = st.radio("View", options=["Annual (by Month)", "Monthly (by Day)"], index=0)

    # Year & Month pickers
    today = pd.Timestamp.today().normalize()
    years = sorted({d.year for d in pd.to_datetime(load_sales()["date"], errors="coerce").dropna()})
    if not years:
        years = [today.year]
    current_year_default = max(years)

    year_sel = st.selectbox("Year", options=years, index=years.index(current_year_default))
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    month_map = {i+1: m for i, m in enumerate(month_names)}

    if view_mode == "Monthly (by Day)":
        month_sel_name = st.selectbox("Month", options=month_names, index=today.month - 1)
        month_sel = month_names.index(month_sel_name) + 1
    else:
        month_sel = None
        month_sel_name = None

# ------------------------
# DATA (filtered by period)
# ------------------------
sdf = load_sales()
if sdf.empty:
    st.stop()

sdf = sdf.dropna(subset=["date"])  # require valid date
sdf["year"] = sdf["date"].dt.year
sdf["month"] = sdf["date"].dt.month
sdf["day"] = sdf["date"].dt.day

# Helper: aggregate KPI (profit %, #orders, total revenue) for a frame

def kpis(df: pd.DataFrame) -> Tuple[float, int, float]:
    # Profit % — weighted by revenue
    rev = pd.to_numeric(df["total_revenue"], errors="coerce").fillna(0)
    prof = pd.to_numeric(df["profit"], errors="coerce").fillna(0)
    profit_pct = float((prof.sum() / rev.sum()) if rev.sum() else np.nan)

    # #Orders — count of `source` records when available else rows
    if "source" in df.columns and not df["source"].isna().all():
        orders = int(df["source"].notna().sum())
    else:
        orders = int(len(df))

    total_rev = float(rev.sum())
    return profit_pct, orders, total_rev


# ------------------------
# ANNUAL (BY MONTH) VIEW
# ------------------------
if view_mode == "Annual (by Month)":
    this_year = year_sel
    prev_year = year_sel - 1

    cur = sdf[sdf["year"] == this_year]
    prv = sdf[sdf["year"] == prev_year]

    # month aggregates
    def agg_month(df):
        return (
            df.groupby("month").agg(
                revenue=("total_revenue", "sum")
            ).reset_index()
        )

    cur_m = agg_month(cur)
    prv_m = agg_month(prv)

    # KPIs
    cur_k = kpis(cur)
    prv_k = kpis(prv)

    # KPI cards (two rows)
    c1, c2, c3, s1, s2, s3 = st.columns([1,1,1,0.2,1,1])
    with c1:
        st.metric("%  {0}".format(this_year), value=f"{(cur_k[0]*100):.1f}%" if not pd.isna(cur_k[0]) else "–")
    with c2:
        st.metric("#PO's  {0}".format(this_year), value=f"{cur_k[1]:,}")
    with c3:
        st.metric("Revenue  {0}".format(this_year), value=_abbr(cur_k[2]))
    with s2:
        st.metric("%  {0}".format(prev_year), value=f"{(prv_k[0]*100):.1f}%" if not pd.isna(prv_k[0]) else "–")
    with s3:
        st.metric("#PO's  {0}".format(prev_year), value=f"{prv_k[1]:,}")
    s4 = st.columns([1])[0]
    with s4:
        st.metric("Revenue  {0}".format(prev_year), value=_abbr(prv_k[2]))

    # Build dataset for chart (side-by-side bars with labels)
    cur_m["Year"], prv_m["Year"] = str(this_year), str(prev_year)
    allm = pd.concat([cur_m, prv_m], ignore_index=True)
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    allm["Month"] = allm["month"].map({i: n for i, n in enumerate(month_names, start=1)})

    if ALTAIR_OK:
        mean_line = allm[allm["Year"] == str(this_year)]["revenue"].mean()
        rule = alt.Chart(pd.DataFrame({"y": [mean_line]})).mark_rule(strokeDash=[6,4], color="#000").encode(y="y:Q")

        base = alt.Chart(allm).transform_calculate(
            xoff="datum.Year === '"+str(this_year)+"' ? -0.15 : 0.15"
        ).mark_bar().encode(
            x=alt.X("Month:N", sort=list(month_names)),
            xOffset=alt.XOffset("Year:N"),
            y=alt.Y("revenue:Q", title="Sum of Total Revenue"),
            color=alt.Color("Year:N", scale=alt.Scale(domain=[str(prev_year), str(this_year)], range=["#1f77b4", "#ff7f0e"])),
            tooltip=["Year", "Month", alt.Tooltip("revenue:Q", title="Revenue", format="$,.0f")],
        ).properties(height=440)

        labels = alt.Chart(allm).mark_text(dy=-6, size=11).encode(
            x=alt.X("Month:N", sort=list(month_names)),
            xOffset=alt.XOffset("Year:N"),
            y=alt.Y("revenue:Q"),
            detail="Year:N",
            text=alt.Text("revenue:Q", format="$,.2s"),
            color=alt.value("black")
        )
        st.altair_chart(base + labels + rule, use_container_width=True)

    # Detail table
    st.subheader("Monthly totals")
    disp = allm.pivot_table(index="Month", columns="Year", values="revenue", aggfunc="sum").reindex(month_names)
    st.dataframe(_styled_table(disp.reset_index().rename(columns={"index": "Month"}), {str(this_year): "${:,.0f}", str(prev_year): "${:,.0f}"}), use_container_width=True, hide_index=True)


# ------------------------
# MONTHLY (BY DAY) VIEW
# ------------------------
else:
    assert month_sel is not None
    this_year = year_sel
    prev_year = year_sel - 1

    cur = sdf[(sdf["year"] == this_year) & (sdf["month"] == month_sel)]
    prv = sdf[(sdf["year"] == prev_year) & (sdf["month"] == month_sel)]

    # day aggregates
    def agg_day(df):
        return (
            df.groupby("day").agg(
                revenue=("total_revenue", "sum")
            ).reset_index()
        )

    cur_d = agg_day(cur)
    prv_d = agg_day(prv)

    # KPIs (restrict to the chosen month)
    cur_k = kpis(cur)
    prv_k = kpis(prv)

    # KPI cards
    c1, c2, c3, s1, s2, s3 = st.columns([1,1,1,0.2,1,1])
    with c1:
        st.metric(f"% {month_map[month_sel]} {this_year}", value=f"{(cur_k[0]*100):.1f}%" if not pd.isna(cur_k[0]) else "–")
    with c2:
        st.metric(f"#PO's {month_map[month_sel]} {this_year}", value=f"{cur_k[1]:,}")
    with c3:
        st.metric(f"Revenue {month_map[month_sel]} {this_year}", value=_abbr(cur_k[2]))
    with s2:
        st.metric(f"% {month_map[month_sel]} {prev_year}", value=f"{(prv_k[0]*100):.1f}%" if not pd.isna(prv_k[0]) else "–")
    with s3:
        st.metric(f"#PO's {month_map[month_sel]} {prev_year}", value=f"{prv_k[1]:,}")
    s4 = st.columns([1])[0]
    with s4:
        st.metric(f"Revenue {month_map[month_sel]} {prev_year}", value=_abbr(prv_k[2]))

    # Build dataset for chart (side-by-side bars with labels)
    cur_d["Year"], prv_d["Year"] = str(this_year), str(prev_year)
    alld = pd.concat([cur_d, prv_d], ignore_index=True)

    if ALTAIR_OK:
        mean_line = alld[alld["Year"] == str(this_year)]["revenue"].mean()
        rule = alt.Chart(pd.DataFrame({"y": [mean_line]})).mark_rule(strokeDash=[6,4], color="#000").encode(y="y:Q")

        base = alt.Chart(alld).mark_bar().encode(
            x=alt.X("day:O", title="Day"),
            xOffset=alt.XOffset("Year:N"),
            y=alt.Y("revenue:Q", title="Sum of Total Revenue"),
            color=alt.Color("Year:N", scale=alt.Scale(domain=[str(prev_year), str(this_year)], range=["#1f77b4", "#ff7f0e"])),
            tooltip=["Year", "day", alt.Tooltip("revenue:Q", title="Revenue", format="$,.0f")],
        ).properties(height=440)

        labels = alt.Chart(alld).mark_text(dy=-6, size=11).encode(
            x=alt.X("day:O", title="Day"),
            xOffset=alt.XOffset("Year:N"),
            y=alt.Y("revenue:Q"),
            detail="Year:N",
            text=alt.Text("revenue:Q", format="$,.2s"),
            color=alt.value("black")
        )
        st.altair_chart(base + labels + rule, use_container_width=True)

    # Detail table
    st.subheader("Daily totals")
    disp = alld.pivot_table(index="day", columns="Year", values="revenue", aggfunc="sum").sort_index()
    st.dataframe(_styled_table(disp.reset_index().rename(columns={"day": "Day"}), {str(this_year): "${:,.0f}", str(prev_year): "${:,.0f}"}), use_container_width=True, hide_index=True)


# ------------------------
# NOTES
# ------------------------
st.caption(
    "Notes: Profit% uses available `profit` (or `total_revenue - total_cost` when `total_cost` exists). #PO's counts non-null `source` rows; if `source` is missing it falls back to total rows in the period."
)
