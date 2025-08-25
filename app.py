import streamlit as st
import pandas as pd
import os
import altair as alt

# ------------------------
# FruttoFoods Quotation Tool (Supabase Edition)
# ------------------------

st.set_page_config(page_title="FruttoFoods Quotation Tool", layout="wide")

LOGO_PATH   = "data/Asset 7@4x.png"
BRAND_GREEN = "#8DC63F"

# ------------------------
# Helpers
# ------------------------
def organic_to_num(val):
    if val == 'Conventional': return 0
    if val == 'Organic':     return 1
    return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_quotations_from_supabase():
    """
    Trae toda la tabla quotations con paginado y normaliza columnas/tipos
    para que el resto de la UI siga funcionando sin cambios.
    """
    try:
        from supabase import create_client
    except Exception as e:
        st.error(f"Falta el paquete 'supabase'. Instálalo en requirements.txt. Detalle: {e}")
        st.stop()

    # 1) Cliente
    try:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
    except Exception as e:
        st.error("No encontré SUPABASE_URL y/o SUPABASE_ANON_KEY en secrets. Revísalos.")
        st.stop()

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 2) Paginado
    frames = []
    page_size = 1000
    for i in range(1000):  # tope de seguridad
        start = i * page_size
        end   = start + page_size - 1
        try:
            resp = (
                sb.table("quotations")
                  .select("cotization_date,organic,product,price,location,volume_num,volume_unit,volume_standard,vendorclean")
                  .range(start, end)
                  .execute()
            )
        except Exception as e:
            st.error(f"Error consultando Supabase: {e}")
            st.stop()

        rows = resp.data or []
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < page_size:
            break

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if df.empty:
        return df

    # 3) Validación de columnas mínimas
    expected_cols = [
        "cotization_date","organic","product","price","location",
        "volume_num","volume_unit","volume_standard","vendorclean"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en Supabase: {missing}")
        st.stop()

    # 4) Normalización de tipos y nombres para que la UI actual funcione
    # Fechas ("M/D/YYYY" como texto en la tabla) -> datetime
    df["cotization_date"] = pd.to_datetime(df["cotization_date"], errors="coerce")

    # Organic 0/1 (num) -> creamos columna 'Organic' (numérica) para la UI
    df["Organic"] = pd.to_numeric(df["organic"], errors="coerce").astype("Int64")

    # Price a num
    df["Price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["Price"])

    # volume_standard puede venir como texto; intentamos numérico, si no usamos volume_num; si no, 1
    vol_std = pd.to_numeric(df["volume_standard"], errors="coerce")
    vol_num = pd.to_numeric(df["volume_num"], errors="coerce")
    df["volume_standard"] = vol_std.fillna(vol_num).fillna(1)

    # volume_unit a str
    df["volume_unit"] = df["volume_unit"].astype(str).fillna("unit")

    # price_per_unit
    df["price_per_unit"] = df["Price"] / df["volume_standard"]

    # Renombrar a los usados por la UI
    df = df.rename(columns={
        "product": "Product",
        "location": "Location",
        "vendorclean": "VendorClean",
    })

    # Compat: por si aún hay registros con "PAS" en Price (no debería, pero prevenimos)
    df = df[~df["Price"].astype(str).str.upper().str.contains("PAS", na=False)]

    # Orden ideal por fecha descendente para primera vista
    df = df.sort_values("cotization_date", ascending=False)

    return df


# ------------------------
# Carga de datos (desde Supabase)
# ------------------------
df = fetch_all_quotations_from_supabase()
if df.empty:
    st.warning("La tabla `quotations` en Supabase no tiene datos.")
    st.stop()

# ------------------------
# Sidebar filters (dinámicos)
# ------------------------
st.sidebar.header("Quotation Filters")

# 1) Products (multiselect)
all_products = sorted(df['Product'].dropna().unique().tolist())
products = st.sidebar.multiselect(
    "Products",
    options=all_products,
    default=all_products  # por defecto, todos
)

# 2) Location dinámico (dependiente de producto)
locs_for_product = df[df['Product'].isin(products)]['Location'].dropna().unique()
locs_for_product = sorted(locs_for_product.tolist())
locations = st.sidebar.multiselect(
    "Location",
    options=locs_for_product,
    default=locs_for_product
)

# 3) Organic Status dinámico
sub = df[df['Product'].isin(products)]
if locations:
    sub = sub[sub['Location'].isin(locations)]

# org_vals viene de la columna numérica 'Organic'
org_vals = sorted([v for v in sub['Organic'].dropna().unique().tolist() if v in (0,1)])
org_map  = {0: 'Conventional', 1: 'Organic'}
org_options = ['All'] + [org_map[o] for o in org_vals]
organic = st.sidebar.selectbox("Organic Status", org_options)

# 4) Volume Unit dinámico
sub2 = sub.copy()
if organic != 'All':
    sub2 = sub2[sub2['Organic'] == organic_to_num(organic)]
vu_opts = sorted([v for v in sub2['volume_unit'].dropna().unique().tolist() if v])
volume_unit = st.sidebar.selectbox("Volume Unit", ['All'] + vu_opts)

# ------------------------
# Aplicar filtros
# ------------------------
if not products:
    g = df.copy()
else:
    g = df[df['Product'].isin(products)]

if locations:
    g = g[g['Location'].isin(locations)]

if organic != 'All':
    g = g[g['Organic'] == organic_to_num(organic)]

if volume_unit != 'All':
    g = g[g['volume_unit'] == volume_unit]

# ------------------------
# Layout y logo
# ------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("FruttoFoods Quotation Tool")
with col2:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=80)
    else:
        st.warning("Logo no encontrado. Verifica 'data/Asset 7@4x.png'.")

# ------------------------
# Mostrar resultados
# ------------------------
if g.empty:
    st.warning("No hay datos para los filtros seleccionados.")
else:
    # Preparar tabla para display
    display = g.rename(columns={
        'cotization_date': 'Date',
        'volume_unit': 'Volume Unit',
        'price_per_unit': 'Price per Unit',
        'VendorClean': 'Vendor'
    })[['Date', 'Product', 'Location', 'Volume Unit', 'Price per Unit', 'Vendor']]

    # Asegurar datetime y ordenar
    display['Date'] = pd.to_datetime(display['Date'], errors='coerce')
    display = display.sort_values(by=['Date'], ascending=[False])

    # Formatos
    display['Date'] = display['Date'].dt.strftime("%m/%d/%Y")
    display['Price per Unit'] = display['Price per Unit'].map(lambda x: f"${x:.2f}")

    st.subheader("Filtered Quotations")
    st.dataframe(display, use_container_width=True)

    # Métricas clave
    st.subheader("Key Metrics")
    min_val = g['price_per_unit'].min()
    max_val = g['price_per_unit'].max()
    avg_val = g['price_per_unit'].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Min Price/Unit", f"${min_val:.2f}")
    c2.metric("Max Price/Unit", f"${max_val:.2f}")
    c3.metric("Avg Price/Unit", f"${avg_val:.2f}")

    # Gráfico de precio medio por vendor
    st.subheader("Average Price/Unit by Vendor")
    if not g['VendorClean'].dropna().empty:
        avg_vendor = g.groupby('VendorClean', dropna=True)['price_per_unit'].mean().reset_index()
        if not avg_vendor.empty:
            chart = alt.Chart(avg_vendor).mark_bar(color=BRAND_GREEN).encode(
                x=alt.X('VendorClean:N', title='Vendor', sort='-y'),
                y=alt.Y('price_per_unit:Q', title='Avg Price/Unit')
            )
            st.altair_chart(chart, use_container_width=True)

            # Recomendación: vendor más barato
            best = avg_vendor.loc[avg_vendor['price_per_unit'].idxmin()]
            st.success(f"**Vendor recomendado:** {best['VendorClean']} a ${best['price_per_unit']:.2f} por unidad")
        else:
            st.info("No hay datos agregables por Vendor con los filtros actuales.")
    else:
        st.info("No hay Vendors en el conjunto filtrado.")
