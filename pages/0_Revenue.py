# revenue_yoy_by_month_and_day.py 
# Streamlit page: Revenue YoY — by Month and by Day (uses ONLY reqs_date)
# - Bars: side-by-side (this year vs last year)
# - Labels: $ compact ($4.0k, $1.2M)
# - Green line: goal (monthly/daily)
# - Black line: average revenue reference (raw average, not %)
# - KPIs for both periods; goal progress cards
# - DATE SOURCE: STRICTLY reqs_date

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

# ========================
# TARGETS (Goals)
# ========================
MONTHLY_GOAL = 3_180_000   # $3.18M per month
DAILY_GOAL   = 138_260     # $138.26K per day

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Revenue YoY • Months & Days", page_icon="📊", layout="wide")
st.title("📊 Revenue YoY — by Month and by Day")
st.caption(
    "Compare total revenue Year-over-Year. Switch between **Annual (by Month)** and **Monthly (by Day)** views. "
    "KPI cards show Profit %, #Orders (count of `source` rows when available), and Total Revenue (short format). "
    "Green line = goal. Black line = average revenue reference. **Date source: reqs_date only.**"
)

# ------------------------
# HELPERS
# ------------------------
def _normalize_txt(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00A0", " ")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _canonical_po_series(s: pd.Series) -> pd.Series:
    """
    Normaliza los PO en `source` para poder contar únicos.
    - Acepta variantes como 'PO #4926', 'po4926', 'Po   #  4926'
    - Devuelve 'po-<id>' si detecta patrón, sino el texto limpio.
    """
    def canon(x):
        if pd.isna(x):
            return np.nan
        t = _normalize_txt(x)
        # extrae lo que sigue a 'po' (números/letras/guiones)
        m = re.search(r"\bpo\s*#?\s*([a-z0-9\-]+)", t)
        if m:
            return f"po-{m.group(1)}"
        return t if t else np.nan
    return s.apply(canon)

def _abbr(n: float) -> str:
    """
    Formato corto para KPIs:
    - 678987 => $679k
    - 1_554_800 => $1.6M
    - 3_100_000_000 => $3B
    Reglas:
      - 'k' sin decimales (compacto)
      - 'M' y 'B' con 1 decimal, quitando .0 si aplica
    """
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "$0"
    n = float(n)
    sign = "-" if n < 0 else ""
    n = abs(n)

    def trim_decimal(x: float, digits: int = 1) -> str:
        s = f"{x:.{digits}f}"
        return s.rstrip("0").rstrip(".")

    if n >= 1_000_000_000:
        val = trim_decimal(n / 1_000_000_000, 1) + "B"
    elif n >= 1_000_000:
        val = trim_decimal(n / 1_000_000, 1) + "M"
    elif n >= 1_000:
        val = f"{n/1_000:.0f}k"
    else:
        val = f"{n:.0f}"
    return f"{sign}${val}"

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
# DATA LOADER  (reqs_date only)
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
        # Dates (we will use reqs_date strictly)
        "reqs_date": ["reqs_date"],
        # IDs
        "invoice_num": ["invoice_num", "invoice #", "invoice"],
        # Dimensions (minimal)
        "customer": ["customer", "customer_name", "client", "cliente"],
        "product": ["product", "item", "item_name", "product_name", "desc", "description"],
        # Metrics
        "quantity": ["quantity", "qty"],
        "total_revenue": ["total_revenue", "total", "line_total", "revenue", "t_revenue", "trevenue"],
        # Optional profit & cost
        "profit": ["profit", "gross_profit", "gp", "margin_amount"],
        "total_cost": ["total_cost", "cost", "cogs"],
        # Orders counter source (for #orders)
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
    sdf["reqs_date"] = pd.to_datetime(sdf["reqs_date"], errors="coerce")  # <-- ONLY this date is used
    for c in ["quantity", "total_revenue", "profit", "total_cost"]:
        if c in sdf.columns:
            sdf[c] = pd.to_numeric(sdf[c], errors="coerce")

    # Require valid reqs_date
    sdf = sdf.dropna(subset=["reqs_date"]).copy()
    sdf["date"] = sdf["reqs_date"]  # alias for downstream code
    sdf["year"] = sdf["date"].dt.year
    sdf["month"] = sdf["date"].dt.month
    sdf["day"] = sdf["date"].dt.day

    # Derive profit if missing but cost present
    if "profit" not in sdf.columns or sdf["profit"].isna().all():
        if "total_cost" in sdf.columns and not sdf["total_cost"].isna().all():
            sdf["profit"] = (sdf["total_revenue"] - sdf["total_cost"])
        else:
            sdf["profit"] = np.nan

    # Profit % (row level) — safe division
    sdf["profit_pct"] = np.where(
        sdf["total_revenue"].abs() > 1e-9, sdf["profit"] / sdf["total_revenue"], np.nan
    )

    # Orders proxy if `source` missing → fall back to row count later
    if "source" not in sdf.columns:
        sdf["source"] = np.nan

    return sdf

# ------------------------
# SIDEBAR (controls)
# ------------------------
sdf = load_sales()
if sdf.empty:
    st.warning("No data with a valid `reqs_date` found.")
    st.stop()

with st.sidebar:
    st.subheader("Controls")
    view_mode = st.radio("View", options=["Annual (by Month)", "Monthly (by Day)"], index=0)

    years_available = sorted(sdf["year"].unique().tolist())
    current_year_default = max(years_available)
    year_sel = st.selectbox("Year", options=years_available, index=years_available.index(current_year_default))

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    month_map = {i+1: m for i, m in enumerate(month_names)}

    if view_mode == "Monthly (by Day)":
        # Default to current month if present; otherwise first available in that year
        months_this_year = sorted(sdf.loc[sdf["year"] == year_sel, "month"].unique().tolist())
        default_month_idx = (months_this_year[-1] - 1) if months_this_year else 0
        month_sel_name = st.selectbox("Month", options=month_names, index=default_month_idx)
        month_sel = month_names.index(month_sel_name) + 1
    else:
        month_sel = None

# ------------------------
# KPI helper
# ------------------------
def kpis(df: pd.DataFrame) -> Tuple[float, int, float]:
    rev = pd.to_numeric(df["total_revenue"], errors="coerce").fillna(0.0)
    prof = pd.to_numeric(df["profit"], errors="coerce").fillna(0.0)
    profit_pct = float((prof.sum() / rev.sum()) if rev.sum() else np.nan)

    # Contar POs únicos (normalizados); si no hay `source`, fallback a #invoice únicos o filas
    if "source" in df.columns and df["source"].notna().any():
        uniq_pos = _canonical_po_series(df["source"]).dropna().unique()
        orders = int(len(uniq_pos))
    elif "invoice_num" in df.columns and df["invoice_num"].notna().any():
        orders = int(pd.Series(df["invoice_num"]).nunique())
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

    def agg_month(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("month").agg(revenue=("total_revenue", "sum")).reset_index()

    cur_m = agg_month(cur)
    prv_m = agg_month(prv)

    # KPIs
    cur_k = kpis(cur)
    prv_k = kpis(prv)

    # Goal progress (annual view uses monthly goal context)
    avg_month_cur = float(cur_m["revenue"].mean()) if not cur_m.empty else 0.0
    pct_goal_avg_month = (avg_month_cur / MONTHLY_GOAL * 100.0) if MONTHLY_GOAL else np.nan

    c1, c2, c3, c4, s1, s2, s3 = st.columns([1, 1, 1, 1, 0.2, 1, 1])
    with c1:
        st.metric(f"%  {this_year}", value=f"{(cur_k[0]*100):.1f}%" if not pd.isna(cur_k[0]) else "–")
    with c2:
        st.metric(f"#PO's  {this_year}", value=f"{cur_k[1]:,}")
    with c3:
        st.metric(f"Revenue  {this_year}", value=_abbr(cur_k[2]))
    with c4:
        st.metric("Avg Month vs Goal", value=f"{pct_goal_avg_month:.1f}%")
    with s2:
        st.metric(f"%  {prev_year}", value=f"{(prv_k[0]*100):.1f}%" if not pd.isna(prv_k[0]) else "–")
    with s3:
        st.metric(f"#PO's  {prev_year}", value=f"{prv_k[1]:,}")
    s4 = st.columns([1])[0]
    with s4:
        st.metric(f"Revenue  {prev_year}", value=_abbr(prv_k[2]))

    # Build dataset for chart (side-by-side bars with labels)
    cur_m["Year"], prv_m["Year"] = str(this_year), str(prev_year)
    allm = pd.concat([cur_m, prv_m], ignore_index=True)
    allm["Month"] = allm["month"].map({i: n for i, n in enumerate(month_names, start=1)})

    if ALTAIR_OK:
        mean_line = allm[allm["Year"] == str(this_year)]["revenue"].mean()  # black line (average)
        mean_rule = alt.Chart(pd.DataFrame({"y": [mean_line]})).mark_rule(strokeDash=[6, 4], color="#000").encode(y="y:Q")
        goal_rule = alt.Chart(pd.DataFrame({"y": [MONTHLY_GOAL]})).mark_rule(strokeDash=[4, 4], color="green").encode(y="y:Q")

        base = alt.Chart(allm).mark_bar().encode(
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
            color=alt.value("black"),
        )
        st.altair_chart(base + labels + mean_rule + goal_rule, use_container_width=True)

    # Detail table
    st.subheader("Monthly totals")
    disp = allm.pivot_table(index="Month", columns="Year", values="revenue", aggfunc="sum").reindex(month_names)
    st.dataframe(
        _styled_table(
            disp.reset_index().rename(columns={"index": "Month"}),
            {str(this_year): "${:,.0f}", str(prev_year): "${:,.0f}"},
        ),
        use_container_width=True,
        hide_index=True,
    )

# ------------------------
# MONTHLY (BY DAY) VIEW
# ------------------------
else:
    assert month_sel is not None
    this_year = year_sel
    prev_year = year_sel - 1

    cur = sdf[(sdf["year"] == this_year) & (sdf["month"] == month_sel)]
    prv = sdf[(sdf["year"] == prev_year) & (sdf["month"] == month_sel)]

    def agg_day(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("day").agg(revenue=("total_revenue", "sum")).reset_index()

    cur_d = agg_day(cur)
    prv_d = agg_day(prv)

    # KPIs (restrict to the chosen month)
    cur_k = kpis(cur)
    prv_k = kpis(prv)

    # Goal progress within month
    month_total_cur = float(cur_d["revenue"].sum() if not cur_d.empty else 0.0)
    pct_goal_month = (month_total_cur / MONTHLY_GOAL * 100.0) if MONTHLY_GOAL else np.nan
    avg_day_cur = float(cur_d["revenue"].mean() if not cur_d.empty else 0.0)
    pct_goal_day = (avg_day_cur / DAILY_GOAL * 100.0) if DAILY_GOAL else np.nan

    # KPI cards
    c1, c2, c3, c4, c5, s1, s2, s3 = st.columns([1, 1, 1, 1, 1, 0.2, 1, 1])
    with c1:
        st.metric(f"% {month_map[month_sel]} {this_year}", value=f"{(cur_k[0]*100):.1f}%" if not pd.isna(cur_k[0]) else "–")
    with c2:
        st.metric(f"#PO's {month_map[month_sel]} {this_year}", value=f"{cur_k[1]:,}")
    with c3:
        st.metric(f"Revenue {month_map[month_sel]} {this_year}", value=_abbr(cur_k[2]))
    with c4:
        st.metric("MTD vs Monthly Goal", value=f"{pct_goal_month:.1f}%")
    with c5:
        st.metric("Avg Day vs Daily Goal", value=f"{pct_goal_day:.1f}%")

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
        mean_line = alld[alld["Year"] == str(this_year)]["revenue"].mean()  # black line (average)
        mean_rule = alt.Chart(pd.DataFrame({"y": [mean_line]})).mark_rule(strokeDash=[6, 4], color="#000").encode(y="y:Q")
        goal_rule = alt.Chart(pd.DataFrame({"y": [DAILY_GOAL]})).mark_rule(strokeDash=[4, 4], color="green").encode(y="y:Q")

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
            color=alt.value("black"),
        )
        st.altair_chart(base + labels + mean_rule + goal_rule, use_container_width=True)

    # Detail table
    st.subheader("Daily totals")
    disp = alld.pivot_table(index="day", columns="Year", values="revenue", aggfunc="sum").sort_index()
    st.dataframe(
        _styled_table(
            disp.reset_index().rename(columns={"day": "Day"}),
            {str(this_year): "${:,.0f}", str(prev_year): "${:,.0f}"},
        ),
        use_container_width=True,
        hide_index=True,
    )

# ------------------------
# NOTES
# ------------------------
st.caption(
    "Notes: The page uses **reqs_date exclusively**. Rows without a valid reqs_date are dropped. "
    "Profit% uses available `profit` (or `total_revenue - total_cost` when `total_cost` exists). "
    "#PO's counts unique normalized `source` values; if `source` is missing it falls back to unique invoices or row count. "
    "Black line = average revenue reference; Green line = goal."
)
