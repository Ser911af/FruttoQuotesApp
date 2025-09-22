# Vendor × Product → Customers — v2 (UX mejorada)
# Propósito: reducir fricción y convertir la pregunta "¿quién compró este producto a este vendor?" en acciones.
# Incluye: propósito/decisiones, plantillas rápidas, flujos alternativos, deep-links, defaults,
# Buyer Assigned y vistas bonus (Dormidos / Participación / Comparar vendors)

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

try:
    from rapidfuzz import process, fuzz  # fuzzy search UX opcional
    _FUZZ = True
except Exception:
    _FUZZ = False

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Vendor × Product → Customers", page_icon="🧾", layout="wide")
st.title("🧾 Vendor × Product → Customers")
st.caption("Flujos rápidos y vistas accionables para compras, ventas y vendor management.")

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
# SALES LOADER (con buyer y product)
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
        "product": ["product", "item", "item_name", "product_name", "desc", "description", "sku", "upc", "gtin"],
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
    sdf["buyer_c"] = sdf["buyer_assigned"].astype(str).map(_normalize_txt)

    # Display labels
    sdf["vendor_disp"] = sdf["vendor"].astype(str).str.title()
    sdf["product_disp"] = sdf["product"].astype(str)
    sdf["customer_disp"] = sdf["customer"].astype(str).str.title()
    sdf["buyer_disp"] = sdf["buyer_assigned"].astype(str).str.title()

    # Unified date
    sdf["date"] = sdf["received_date"].fillna(sdf["reqs_date"]).fillna(sdf["created_at"])  # fallback chain
    sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")

    # Order identifier
    sdf["order_id"] = sdf["invoice_num"].astype(str)
    sdf.loc[sdf["order_id"].isin(["nan", "none", "", "NaT"]), "order_id"] = np.nan

    return sdf


# ------------------------
# SIDEBAR: fechas + propósito
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

    with st.expander("¿Para qué sirve (decisiones)?", expanded=True):
        st.markdown(
            """
- **Comercial**: priorizar llamadas (recency + AOV), reactivar dormidos.
- **Compras**: ajustar run-rate por SKU y reducir desperdicio.
- **Vendor mgmt**: negociar términos con cartera real por producto.
- **Riesgo**: detectar concentración por cliente.
            """
        )

# Adaptive as_of_date
today_norm = pd.Timestamp.now(tz="America/Bogota").normalize().tz_localize(None)
end_minus_1 = (end_date - pd.Timedelta(days=1)).normalize()
as_of_date = min(today_norm, end_minus_1)
st.caption(f"Recency as-of: **{as_of_date.date()}**")

# ------------------------
# DATA + métricas base
# ------------------------
sdf = load_sales()

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Sales records", value=len(sdf))
with m2:
    st.metric("Vendors", value=int(sdf["vendor_c"].nunique()) if not sdf.empty else 0)
with m3:
    st.metric("Productos", value=int(sdf["product_c"].nunique()) if not sdf.empty else 0)
with m4:
    st.metric("Clientes", value=int(sdf["customer_c"].nunique()) if not sdf.empty else 0)
with m5:
    st.metric("Buyers", value=int(sdf["buyer_c"].nunique()) if not sdf.empty else 0)

if sdf.empty:
    st.stop()

mask = (sdf["date"] >= start_date) & (sdf["date"] < end_date)
sdf_f = sdf[mask].copy()
st.caption(f"Filtered rows: {len(sdf_f)} in selected range.")
if sdf_f.empty:
    st.warning("No data in the selected range.")
    st.stop()

# ------------------------
# Active vendors/products (para listas cortas + defaults)
# ------------------------
vend_stats = (sdf_f.groupby("vendor_c", dropna=False)
                .agg(rev=("total_revenue","sum"), n_orders=("order_id","nunique"))
                .reset_index()
                .sort_values(["rev","n_orders"], ascending=[False,False]))
vend_map = (sdf_f[["vendor_c","vendor_disp"]].drop_duplicates().set_index("vendor_c")["vendor_disp"])
active_vendors = vend_stats[vend_stats["rev"].fillna(0) > 0]

prod_stats_all = (sdf_f.groupby(["product_c","product_disp"], dropna=False)
                    .agg(rev=("total_revenue","sum"), n_orders=("order_id","nunique"))
                    .reset_index()
                    .sort_values(["rev","n_orders"], ascending=[False,False]))

# ------------------------
# Plantillas de preguntas (entry points)
# ------------------------
st.subheader("⚡ Plantillas rápidas")
col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("Top 20 vendors del período"):
        st.session_state["vendor_norm_list"] = active_vendors["vendor_c"].head(20).tolist()
        st.session_state["entry_mode"] = "top_vendors"
with col_b:
    if st.button("Top productos de este vendor"):
        st.session_state["entry_mode"] = "top_products_vendor"
with col_c:
    if st.button("Clientes dormidos (≥30 días)"):
        st.session_state["entry_mode"] = "dormidos"

# ------------------------
# Deep-links: leer query params si vienen
# ------------------------
qp = st.query_params
pre_v = qp.get("vendor", None)
pre_p = qp.get("product", None)
pre_c = qp.get("customer", None)

# ------------------------
# TABS: Flujos alternativos
# ------------------------
tab_vp, tab_pv, tab_cv, tab_bonus = st.tabs([
    "Vendor → Producto",
    "Producto → Vendor",
    "Cliente → (Vendor, Producto)",
    "Vistas bonus"
])

# ------------------------
# TAB 1 — Vendor → Producto (principal)
# ------------------------
with tab_vp:
    left, right = st.columns([1, 1])
    vend_opts = [""] + [vend_map.get(v, v.title()) for v in active_vendors["vendor_c"]]

    default_vendor_index = 0
    if pre_v is not None:
        vcanon = pre_v if isinstance(pre_v, str) else pre_v[0]
        vname = vend_map.get(vcanon, None)
        if vname and vname in vend_opts:
            default_vendor_index = vend_opts.index(vname)

    with left:
        vendor_sel_disp = st.selectbox("Vendor (activos en el rango)", vend_opts, index=default_vendor_index, placeholder="Top first")

    if vendor_sel_disp:
        vendor_norm = (
            sdf_f.loc[sdf_f["vendor_disp"] == vendor_sel_disp, "vendor_c"].dropna().iloc[0]
            if (sdf_f["vendor_disp"] == vendor_sel_disp).any()
            else _normalize_txt(vendor_sel_disp)
        )
        vmask = sdf_f["vendor_c"] == vendor_norm
        prod_stats = (sdf_f[vmask].groupby(["product_c","product_disp"], dropna=False)
                      .agg(rev=("total_revenue","sum"), n_orders=("order_id","nunique"))
                      .reset_index()
                      .sort_values(["rev","n_orders"], ascending=[False,False]))

        q_prod = st.text_input("Buscar producto (typos-friendly)", "")
        pdisp_list = prod_stats["product_disp"].astype(str).tolist()
        if q_prod and _FUZZ:
            top = process.extract(q_prod, pdisp_list, scorer=fuzz.WRatio, limit=30)
            pdisp_list = [pdisp_list[idx] for _, _, idx in top]

        topN = st.slider("Top N productos por revenue", 10, 150, 30)
        prod_opts = [""] + pdisp_list[:topN]

        default_product_index = 0
        if pre_p is not None and isinstance(pre_p, str) and pre_p in prod_opts:
            default_product_index = prod_opts.index(pre_p)

        with right:
            product_sel_disp = st.selectbox("Producto (de ese Vendor)", prod_opts, index=default_product_index, placeholder="Select…")

        if vendor_sel_disp and product_sel_disp:
            product_norm = (
                sdf_f.loc[sdf_f["product_disp"] == product_sel_disp, "product_c"].dropna().iloc[0]
                if (sdf_f["product_disp"] == product_sel_disp).any()
                else _normalize_txt(product_sel_disp)
            )
            st.query_params.update(vendor=vendor_norm, product=product_sel_disp)

            vp = sdf_f[(sdf_f["vendor_c"] == vendor_norm) & (sdf_f["product_c"] == product_norm)].copy()
            if vp.empty:
                st.warning("Ese vendor no vendió ese producto en el rango seleccionado (o los nombres no coinciden).")
            else:
                if vp["order_id"].notna().any():
                    orders = vp.groupby(["customer_c", "order_id"], dropna=False).agg(order_rev=("total_revenue", "sum")).reset_index()
                    orders_per_cust = orders.groupby("customer_c").agg(n_orders=("order_id", "nunique"), aov=("order_rev", "mean"))
                else:
                    orders_per_cust = vp.groupby("customer_c").agg(n_orders=("date", "count"), aov=("total_revenue", "mean"))

                agg = vp.groupby("customer_c").agg(
                    customer_disp=("customer_disp", lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""),
                    buyer_disp=("buyer_disp", lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""),
                    total_revenue=("total_revenue", "sum"),
                    total_qty=("quantity", "sum"),
                    first_sale=("date", "min"),
                    last_sale=("date", "max"),
                ).join(orders_per_cust, how="left").fillna({"n_orders": 0, "aov": 0})

                agg["recency_days"] = (as_of_date - agg["last_sale"].dt.normalize()).dt.days
                agg["z_rev"], agg["z_freq"] = _winsorized_z(agg["total_revenue"]), _winsorized_z(agg["n_orders"])
                agg["vp_rank_score"] = 0.6 * agg["z_rev"] + 0.4 * agg["z_freq"]

                cust_tbl = agg.reset_index().sort_values(["vp_rank_score", "total_revenue"], ascending=[False, False])
                cust_tbl = _fmt_dates(cust_tbl, ["first_sale", "last_sale"])  

                cdisp = cust_tbl[[
                    "customer_disp", "buyer_disp", "n_orders", "total_qty", "aov", "total_revenue",
                    "first_sale", "last_sale", "recency_days",
                ]].rename(columns={
                    "customer_disp": "Customer",
                    "buyer_disp": "Buyer Assigned",
                    "n_orders": "Orders",
                    "total_qty": "Qty",
                    "aov": "AOV",
                    "total_revenue": "Revenue",
                    "first_sale": "First Sale",
                    "last_sale": "Last Sale",
                    "recency_days": "Recency (days)",
                })

                styles = {"Orders": "{:,.0f}", "Qty": "{:,.0f}", "AOV": "${:,.2f}", "Revenue": "${:,.0f}", "Recency (days)": "{:,.0f}"}
                st.subheader("👥 Customers that bought it")
                st.dataframe(_styled_table(cdisp, styles), use_container_width=True, hide_index=True)

                k1, k2, k3, k4 = st.columns(4)
                with k1: st.metric("Customers", value=int(cust_tbl.shape[0]))
                with k2: st.metric("Total Revenue", value=f"${cust_tbl['total_revenue'].sum():,.0f}")
                with k3: st.metric("Units", value=f"{cust_tbl['total_qty'].sum():,.0f}")
                with k4: st.metric("Orders", value=f"{int(cust_tbl['n_orders'].sum())}")

                if ALTAIR_OK and len(cust_tbl) > 0:
                    chart = alt.Chart(cust_tbl.head(25)).mark_bar().encode(
                        x=alt.X("total_revenue:Q", title="Revenue"),
                        y=alt.Y("customer_disp:N", sort="-x", title="Customer"),
                        tooltip=["customer_disp", "buyer_disp", "n_orders", "total_qty", "aov", "total_revenue"],
                    ).properties(height=420)
                    st.altair_chart(chart, use_container_width=True)

                st.subheader("📄 Line Items (detalle)")
                ld = vp[["date", "invoice_num", "sales_order", "customer_disp", "buyer_disp", "price_per_unit", "quantity", "total_revenue"]].copy()
                ld = _fmt_dates(ld, ["date"]).rename(columns={
                    "date": "Date", "invoice_num": "Invoice", "sales_order": "Sales Order",
                    "customer_disp": "Customer", "buyer_disp": "Buyer Assigned",
                    "price_per_unit": "Unit Price", "quantity": "Qty", "total_revenue": "Revenue",
                })
                st.dataframe(_styled_table(ld, {"Unit Price": "${:,.2f}", "Qty": "{:,.0f}", "Revenue": "${:,.2f}"}), use_container_width=True, hide_index=True)

                st.download_button("⬇️ Download customers CSV", data=cdisp.to_csv(index=False).encode("utf-8"), file_name="customers_vendor_product.csv", mime="text/csv")
                st.download_button("⬇️ Download line items CSV", data=ld.to_csv(index=False).encode("utf-8"), file_name="line_items_vendor_product.csv", mime="text/csv")

                call_cols = ["Customer", "Buyer Assigned", "Orders", "Qty", "Revenue", "Recency (days)"]
                st.download_button("📞 Crear lista de llamadas (CSV)", data=cdisp[call_cols].to_csv(index=False).encode("utf-8"), file_name="call_list_vendor_product.csv", mime="text/csv")

# ------------------------
# TAB 2 — Producto → Vendor
# ------------------------
with tab_pv:
    st.markdown("Cuando sabes el **producto**, pero no el vendor.")
    q_prod2 = st.text_input("Buscar producto (fuzzy)", value=pre_p or "")
    pstats = prod_stats_all.copy()
    pdisp_list2 = pstats["product_disp"].astype(str).tolist()
    if q_prod2 and _FUZZ:
        top = process.extract(q_prod2, pdisp_list2, scorer=fuzz.WRatio, limit=50)
        pdisp_list2 = [pdisp_list2[idx] for _, _, idx in top]
    prod_sel2 = st.selectbox("Producto", [""] + pdisp_list2[:50], index=0)

    if prod_sel2:
        pc2 = sdf_f.loc[sdf_f["product_disp"] == prod_sel2, "product_c"].dropna()
        if pc2.empty:
            st.warning("Producto no encontrado en el rango.")
        else:
            pc2 = pc2.iloc[0]
            pv = sdf_f[sdf_f["product_c"] == pc2]
            vend_rank = (pv.groupby(["vendor_c","vendor_disp"], dropna=False)
                           .agg(rev=("total_revenue","sum"), orders=("order_id","nunique"))
                           .reset_index()
                           .sort_values(["rev","orders"], ascending=[False,False]))
            st.dataframe(_styled_table(vend_rank.rename(columns={"rev":"Revenue","orders":"Orders"}), {"Revenue":"${:,.0f}","Orders":"{:,.0f}"}), use_container_width=True, hide_index=True)
            st.caption("Tip: copia el vendor y vuelve a la pestaña principal para ver los clientes de ese SKU.")

# ------------------------
# TAB 3 — Cliente → (Vendor, Producto)
# ------------------------
with tab_cv:
    st.markdown("Cuando partes del **cliente** y quieres ver historial por vendor/producto y detectar gaps.")
    cust_opts = [""] + sorted(sdf_f["customer_disp"].dropna().unique().tolist())
    default_c_idx = 0
    if pre_c and pre_c in cust_opts:
        default_c_idx = cust_opts.index(pre_c)
    cust_sel = st.selectbox("Cliente", cust_opts, index=default_c_idx)

    if cust_sel:
        cc = sdf_f.loc[sdf_f["customer_disp"] == cust_sel, "customer_c"].dropna().iloc[0]
        cl = sdf_f[sdf_f["customer_c"] == cc]
        grid = (cl.groupby(["vendor_disp","product_disp"], dropna=False)
                  .agg(rev=("total_revenue","sum"), qty=("quantity","sum"), last_sale=("date","max"))
                  .reset_index()
                  .sort_values(["rev","qty"], ascending=[False,False]))
        grid = _fmt_dates(grid, ["last_sale"]).rename(columns={"rev":"Revenue","qty":"Qty"})
        st.dataframe(_styled_table(grid, {"Revenue":"${:,.0f}","Qty":"{:,.0f}"}), use_container_width=True, hide_index=True)

        thr = st.slider("Dormido si recency ≥ días", 15, 120, 45)
        rec = (cl.groupby(["vendor_disp","product_disp"], dropna=False)
                 .agg(last_sale=("date","max"))
                 .reset_index())
        rec["recency_days"] = (as_of_date - rec["last_sale"].dt.normalize()).dt.days
        dorm = rec[rec["recency_days"] >= thr].sort_values("recency_days", ascending=False)
        dorm = _fmt_dates(dorm, ["last_sale"]).rename(columns={"last_sale":"Last Sale"})
        st.markdown("**Productos dormidos para este cliente**")
        st.dataframe(_styled_table(dorm, {"recency_days":"{:,.0f}"}), use_container_width=True, hide_index=True)

# ------------------------
# TAB 4 — Vistas bonus (Dormidos / Participación / Comparar vendors)
# ------------------------
with tab_bonus:
    st.subheader("🛌 Clientes dormidos para un vendor+producto (global)")
    thr2 = st.slider("Dormido si recency ≥ días (global)", 15, 120, 30)
    rec2 = (sdf_f.groupby(["vendor_disp","product_disp","customer_disp"], dropna=False)
              .agg(last_sale=("date","max"), rev=("total_revenue","sum"))
              .reset_index())
    rec2["recency_days"] = (as_of_date - rec2["last_sale"].dt.normalize()).dt.days
    dorm2 = rec2[rec2["recency_days"] >= thr2].sort_values(["vendor_disp","product_disp","recency_days"], ascending=[True,True,False])
    dorm2 = _fmt_dates(dorm2, ["last_sale"]).rename(columns={"last_sale":"Last Sale","rev":"Revenue"})
    st.dataframe(_styled_table(dorm2, {"Revenue":"${:,.0f}","recency_days":"{:,.0f}"}), use_container_width=True, hide_index=True)

    st.subheader("🥧 Participación por cliente (share del SKU del vendor)")
    st.caption("Selecciona vendor y producto en la pestaña principal para reflejarse aquí (deep-link).")
    if "vendor" in st.query_params and "product" in st.query_params:
        vcanon = st.query_params["vendor"]
        pdisp = st.query_params["product"]
        pc = sdf_f.loc[sdf_f["product_disp"] == pdisp, "product_c"].dropna()
        if not pc.empty:
            pc = pc.iloc[0]
            vp2 = sdf_f[(sdf_f["vendor_c"] == vcanon) & (sdf_f["product_c"] == pc)]
            part = vp2.groupby("customer_disp").agg(rev=("total_revenue","sum")).reset_index()
            total = part["rev"].sum()
            if total > 0:
                part["share"] = part["rev"] / total
                part = part.sort_values("share", ascending=False)
                st.dataframe(_styled_table(part.rename(columns={"rev":"Revenue","share":"Share"}), {"Revenue":"${:,.0f}","Share":"{:.1%}"}), use_container_width=True, hide_index=True)
                if ALTAIR_OK and len(part) > 0:
                    chartp = alt.Chart(part.head(20)).mark_bar().encode(
                        x=alt.X("share:Q", title="Share"),
                        y=alt.Y("customer_disp:N", sort="-x", title="Customer"),
                        tooltip=["customer_disp","rev","share"],
                    ).properties(height=420)
                    st.altair_chart(chartp, use_container_width=True)

    st.subheader("🆚 Comparar vendors para el mismo producto")
    q_cmp = st.text_input("Producto para comparar vendors", value=pre_p or "")
    if q_cmp:
        pc3 = sdf_f.loc[sdf_f["product_disp"].str.lower() == q_cmp.lower(), "product_c"].dropna()
        if not pc3.empty:
            pc3 = pc3.iloc[0]
            cmpdf = (sdf_f[sdf_f["product_c"] == pc3]
                        .groupby(["vendor_disp"], dropna=False)
                        .agg(avg_price=("price_per_unit","mean"), orders=("order_id","nunique"), rev=("total_revenue","sum"))
                        .reset_index()
                        .sort_values(["rev","orders"], ascending=[False,False]))
            st.dataframe(_styled_table(_round_cols(cmpdf, {"avg_price":2}).rename(columns={"avg_price":"Avg Price","orders":"Orders","rev":"Revenue"}), {"Avg Price":"${:,.2f}","Orders":"{:,.0f}","Revenue":"${:,.0f}"}), use_container_width=True, hide_index=True)

# ------------------------
# Notas finales
# ------------------------
st.markdown(
    f"""
**Notas**
- Rango de fechas: **inicio inclusivo** y **fin exclusivo** (+1 día internamente).
- *Recency* medido **al {as_of_date.date()}**.
- Listas reducidas: solo **vendors activos** y ordenados por revenue.
- **Deep-links**: la URL lleva `vendor`, `product`, `customer` para vistas compartibles.
- **Buyer Assigned** visible y exportable para listas de llamadas.
"""
)
