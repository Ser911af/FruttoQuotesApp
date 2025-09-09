# pages/0_Explorer.py
import os
import pandas as pd
import streamlit as st

# Auth simple (asegúrate de tener simple_auth.py en la raíz del proyecto)
from simple_auth import ensure_login, logout_button

# Altair opcional (no rompe si no está instalado)
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

# ------------------------
# FruttoFoods Explorer (Supabase)
# ------------------------

st.set_page_config(page_title="FruttoFoods Explorer", layout="wide")

# ✅ Exigir login antes de cargar nada pesado
user = ensure_login()   # detiene la página si no hay sesión
# (Opcional) botón de salir aquí mismo
with st.sidebar:
    logout_button()

LOGO_PATH   = "data/Asset 7@4x.png"
BRAND_GREEN = "#8DC63F"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_quotations_from_supabase():
    """Trae quotations paginado; ante errores devuelve DF vacío (no rompe UI)."""
    try:
        from supabase import create_client
    except Exception as e:
        st.error(f"Falta 'supabase' en requirements: {e}")
        return pd.DataFrame()

    try:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
    except Exception:
        st.error("No encontré SUPABASE_URL / SUPABASE_ANON_KEY en secrets.")
        return pd.DataFrame()

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    frames, page_size = [], 1000
    for i in range(1000):
        start, end = i*page_size, i*page_size + page_size - 1
        try:
            resp = (
                sb.table("quotations")
                  .select(
                      "cotization_date,organic,product,price,location,"
                      "volume_num,volume_unit,volume_standard,vendorclean"
                  )
                  .range(start, end)
                  .execute()
            )
        except Exception as e:
            st.error(f"Error consultando Supabase: {e}")
            return pd.DataFrame()

        rows = resp.data or []
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < page_size:
            break

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df

    # Validación básica
    needed = [
        "cotization_date","organic","product","price","location",
        "volume_num","volume_unit","volume_standard","vendorclean"
    ]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        st.error(f"Faltan columnas en Supabase: {miss}")
        return pd.DataFrame()

    # Normalización
    df["cotization_date"] = pd.to_datetime(df["cotization_date"], errors="coerce")
    df["Organic"] = pd.to_numeric(df["organic"], errors="coerce").astype("Int64")
    df["Price"]   = pd.to_numeric(df["price"], errors="coerce")

    # Filtra filas sin precio numérico
    df = df.dropna(subset=["Price"])

    vol_std = pd.to_numeric(df["volume_standard"], errors="coerce")
    vol_num = pd.to_numeric(df["volume_num"], errors="coerce")
    df["volume_standard"] = vol_std.fillna(vol_num).fillna(1)
    df["volume_unit"]     = df["volume_unit"].astype(str).fillna("unit")
    df["price_per_unit"]  = df["Price"] / df["volume_standard"]

    df = df.rename(columns={
        "product":"Product",
        "location":"Location",
        "vendorclean":"VendorClean"
    })

    # ✅ FIX: usar .str.upper() (antes era .upper())
    df = df[~df["Price"].astype(str).str.upper().str.contains("PAS", na=False)]

    df = df.sort_values("cotization_date", ascending=False)
    return df

# ---------- UI ----------
df = fetch_all_quotations_from_supabase()

col1, col2 = st.columns([3, 1])
with col1:
    st.title("FruttoFoods Quotation Tool")
    st.caption(f"Sesión: {user}")
with col2:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=80)
    else:
        st.warning("Logo no encontrado en data/Asset 7@4x.png")

st.sidebar.header("Quotation Filters")

if df.empty:
    st.sidebar.info("Sin datos disponibles desde Supabase.")
    st.info("Sin datos para mostrar.")
    st.stop()

# 1) Products
all_products = sorted(df['Product'].dropna().unique().tolist())
products = st.sidebar.multiselect("Products", options=all_products, default=all_products)

# 2) Location
locs_for_product = df[df['Product'].isin(products)]['Location'].dropna().unique()
locs_for_product = sorted(locs_for_product.tolist())
locations = st.sidebar.multiselect("Location", options=locs_for_product, default=locs_for_product)

# 3) Organic
sub = df[df['Product'].isin(products)]
if locations:
    sub = sub[sub['Location'].isin(locations)]
org_vals = sorted([v for v in sub['Organic'].dropna().unique().tolist() if v in (0,1)])
org_map  = {0:'Conventional', 1:'Organic'}
org_options = ['All'] + [org_map[o] for o in org_vals]
organic = st.sidebar.selectbox("Organic Status", org_options)

# 4) Volume Unit
sub2 = sub.copy()
if organic != 'All':
    sub2 = sub2[sub2['Organic'] == (0 if organic=='Conventional' else 1)]
vu_opts = sorted([v for v in sub2['volume_unit'].dropna().unique().tolist() if v])
volume_unit = st.sidebar.selectbox("Volume Unit", ['All'] + vu_opts)

# Filtros
g = df[df['Product'].isin(products)] if products else df.copy()
if locations:
    g = g[g['Location'].isin(locations)]
if organic != 'All':
    g = g[g['Organic'] == (0 if organic=='Conventional' else 1)]
if volume_unit != 'All':
    g = g[g['volume_unit'] == volume_unit]

# Tabla
if g.empty:
    st.warning("No hay datos para los filtros seleccionados.")
else:
    display = g.rename(columns={
        'cotization_date': 'Date',
        'volume_unit': 'Volume Unit',
        'price_per_unit': 'Price per Unit',
        'VendorClean': 'Vendor'
    })[['Date','Product','Location','Volume Unit','Price per Unit','Vendor']]

    display['Date'] = pd.to_datetime(display['Date'], errors='coerce')
    display = display.sort_values(by=['Date'], ascending=[False])
    display['Date'] = display['Date'].dt.strftime("%m/%d/%Y")
    display['Price per Unit'] = display['Price per Unit'].map(lambda x: f"${x:.2f}")

    st.subheader("Filtered Quotations")
    st.dataframe(display, use_container_width=True)

    # Métricas
    st.subheader("Key Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Min Price/Unit", f"${g['price_per_unit'].min():.2f}")
    c2.metric("Max Price/Unit", f"${g['price_per_unit'].max():.2f}")
    c3.metric("Avg Price/Unit", f"${g['price_per_unit'].mean():.2f}")

    # Chart
    st.subheader("Average Price/Unit by Vendor")
    if not g['VendorClean'].dropna().empty:
        avg_vendor = g.groupby('VendorClean', dropna=True)['price_per_unit'].mean().reset_index()
        if not avg_vendor.empty and ALTAIR_OK:
            chart = alt.Chart(avg_vendor).mark_bar(color=BRAND_GREEN).encode(
                x=alt.X('VendorClean:N', title='Vendor', sort='-y'),
                y=alt.Y('price_per_unit:Q', title='Avg Price/Unit')
            )
            st.altair_chart(chart, use_container_width=True)
            best = avg_vendor.loc[avg_vendor['price_per_unit'].idxmin()]
            st.success(f"**Vendor recomendado:** {best['VendorClean']} a ${best['price_per_unit']:.2f} por unidad")
        elif not ALTAIR_OK:
            st.info("Altair no está instalado; omitiendo gráfico.")
        else:
            st.info("No hay datos agregables por Vendor con los filtros actuales.")
    else:
        st.info("No hay Vendors en el conjunto filtrado.")
