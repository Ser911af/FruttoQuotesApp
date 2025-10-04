# pages/3_Customer_Retention.py
# Customer Retention Rankings — by Company and by Sales Rep

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
st.set_page_config(page_title="Customer Retention Rankings", page_icon="📈", layout="wide")
st.title("📈 Customer Retention Rankings")

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

    # Map aliases
    alias_map = {
        "reqs_date": ["reqs_date"],
        "most_recent_invoice_paid_date": ["most_recent_invoice_paid_date"],
        "received_date": ["received_date"],
        "sales_order": ["sales_order"],
        "invoice_num": ["invoice_num"],
        "sales_rep": ["sales_rep", "cus_sales_rep"],
        "customer": ["customer"],
        "total_revenue": ["total_revenue"],
        "total_profit_usd": ["total_profit_usd"],
        "quantity": ["quantity"],
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

    for c in ["reqs_date","most_recent_invoice_paid_date","received_date"]:
        sdf[c] = pd.to_datetime(sdf[c], errors="coerce")
    for c in ["quantity","total_revenue","total_profit_usd"]:
        sdf[c] = pd.to_numeric(sdf[c], errors="coerce")

    sdf["customer_c"]  = sdf["customer"].astype(str).map(_normalize_txt)
    sdf["sales_rep_c"] = sdf["sales_rep"].astype(str).map(_normalize_txt)

    # --- PATCH: date basis ---
    sdf["date_requested"] = sdf["reqs_date"]
    sdf["date_paid"]      = sdf["most_recent_invoice_paid_date"]
    sdf["date_received"]  = sdf["received_date"]
    sdf["date"]           = sdf["date_requested"]

    # --- PATCH: order_id from sales_order ---
    order_id = np.where(sdf["sales_order"].notna(), sdf["sales_order"].astype(str),
               np.where(sdf["invoice_num"].notna(), sdf["invoice_num"].astype(str), np.nan))
    order_id = pd.Series(order_id).mask(lambda x: x.str.lower().isin(["nan","none",""]))
    sdf["order_id"] = order_id

    return sdf

# ------------------------
# METRICS
# ------------------------
def _week_key(ts: pd.Timestamp) -> str:
    if pd.isna(ts): return ""
    iso = ts.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"

def make_retention_metrics(df, start, end, as_of):
    if df.empty: return pd.DataFrame()
    d = df[(df["date"] >= start) & (df["date"] < end)].copy()
    if d.empty: return pd.DataFrame()

    d["week_key"] = d["date"].apply(_week_key)

    if d["order_id"].notna().any():
        orders = d.groupby(["customer_c","order_id"]).agg(
            order_revenue=("total_revenue","sum")
        ).reset_index()
        orders = orders[orders["order_revenue"].fillna(0) != 0]  # excluye $0
        orders_per_customer = orders.groupby("customer_c").agg(
            n_orders=("order_id","nunique"), aov=("order_revenue","mean")
        )
    else:
        orders_per_customer = d.groupby("customer_c").agg(
            n_orders=("date","count"), aov=("total_revenue","mean")
        )

    agg = d.groupby("customer_c").agg(
        total_revenue=("total_revenue","sum"),
        total_qty=("quantity","sum"),
        last_sale=("date","max"),
        first_sale=("date","min"),
        active_weeks=("week_key","nunique")
    )

    span_weeks = []
    for _, row in agg.iterrows():
        f, l = row["first_sale"], row["last_sale"]
        if pd.isna(f) or pd.isna(l): span_weeks.append(1)
        else:
            days = max(0,(l.normalize()-f.normalize()).days)
            span_weeks.append(max(1,(days//7)+1))
    agg["weeks_span"] = span_weeks
    agg["regularity_ratio"] = (agg["active_weeks"]/agg["weeks_span"]).clip(upper=1.0)
    agg["recency_days"] = (as_of.normalize()-agg["last_sale"].dt.normalize()).dt.days

    agg = agg.join(orders_per_customer, how="left").fillna({"aov":0,"n_orders":0})

    def z(x):
        x_=x.clip(lower=x.quantile(0.02),upper=x.quantile(0.98))
        mu,sd=x_.mean(),x_.std(ddof=0)
        return (x-mu)/(sd+1e-9)

    agg["z_rev"]=z(agg["total_revenue"].fillna(0))
    agg["z_freq"]=z(agg["n_orders"].fillna(0))
    agg["z_recency"]=z((-agg["recency_days"].fillna(365)))
    agg["z_regular"]=z(agg["regularity_ratio"].fillna(0))

    agg["retention_score"]=(
        0.35*agg["z_rev"]+0.25*agg["z_freq"]+0.25*agg["z_recency"]+0.15*agg["z_regular"]
    )

    agg=agg.reset_index().rename(columns={"customer_c":"customer"})
    return agg

def make_salesrep_rankings(df,start,end,as_of):
    if df.empty: return pd.DataFrame()
    d=df[(df["date"]>=start)&(df["date"]<end)].copy()
    if d.empty: return pd.DataFrame()

    rep_cust=(d.groupby(["sales_rep_c","customer_c"])
                .agg(rev=("total_revenue","sum"),qty=("quantity","sum"))
                .reset_index())
    rep_cust=rep_cust[rep_cust["sales_rep_c"].notna() & (rep_cust["sales_rep_c"]!="")]

    # --- PATCH: agrupar por sales_rep_c ---
    rep_totals=(rep_cust.groupby("sales_rep_c")
                  .agg(total_revenue=("rev","sum"),
                       total_qty=("qty","sum"),
                       active_customers=("customer_c","nunique"))
                  .reset_index()
                  .rename(columns={"sales_rep_c":"sales_rep"}))

    # pair-level spans
    d2=d.copy(); d2["week_key"]=d2["date"].apply(_week_key)
    pair_span=(d2.groupby(["sales_rep_c","customer_c"])
                 .agg(last_sale=("date","max"),
                      first_sale=("date","min"),
                      active_weeks=("week_key","nunique"))
                 .reset_index())
    span_days=(pair_span["last_sale"].dt.normalize()-pair_span["first_sale"].dt.normalize()).dt.days.clip(lower=0)
    pair_span["weeks_span"]=(span_days//7)+1
    pair_span["regularity_pair"]=(pair_span["active_weeks"]/pair_span["weeks_span"]).clip(upper=1.0)
    pair_span["recency_days_pair"]=(as_of.normalize()-pair_span["last_sale"].dt.normalize()).dt.days

    rc=rep_cust.merge(pair_span,on=["sales_rep_c","customer_c"],how="left")
    rc=rc.rename(columns={"sales_rep_c":"sales_rep"})
    rc["w"]=rc["rev"].clip(lower=0)+1e-9

    # group by rep
    grp=(rc.groupby("sales_rep")
           .apply(lambda g: pd.Series({
               "avg_regularity":np.average(g["regularity_pair"].fillna(0),weights=g["w"]),
               "avg_recency_days":np.average(g["recency_days_pair"].fillna(0),weights=g["w"])
           }))
           .reset_index())

    cust_all=make_retention_metrics(d,start,end,as_of).rename(columns={"customer":"customer_c"})
    rc2=rc.merge(cust_all[["customer_c","retention_score"]],on="customer_c",how="left")
    by_rep_ret=(rc2.groupby("sales_rep")
                  .apply(lambda g: pd.Series({
                      "avg_retention_score":np.average(g["retention_score"].fillna(0),weights=g["w"])
                  }))
                  .reset_index())

    rep=(rep_totals
         .merge(grp,on="sales_rep",how="left")
         .merge(by_rep_ret,on="sales_rep",how="left")
         .fillna({"avg_regularity":0,"avg_recency_days":0,"avg_retention_score":0}))

    rep["rank_score"]=(
        0.5*rep["avg_retention_score"]+
        0.3*rep["total_revenue"].rank(pct=True)+
        0.2*rep["active_customers"].rank(pct=True)
    )
    return rep.sort_values(["rank_score","total_revenue"],ascending=[False,False]).reset_index(drop=True)
