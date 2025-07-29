import streamlit as st
import pandas as pd
import os
import altair as alt

# ------------------------
# FruttoFoods Quotation Tool
# ------------------------

LOGO_PATH   = "data/Asset 7@4x.png"
BRAND_GREEN = "#8DC63F"
DATA_PATH   = "data/market_cleaned_cleaned.xlsx"

def organic_to_num(val):
    if val == 'Conventional': return 0
    if val == 'Organic':     return 1
    return None

# Carga de datos
if os.path.exists(DATA_PATH):
    df = pd.read_excel(DATA_PATH)
else:
    st.error(f"Archivo no encontrado: {DATA_PATH}")
    st.stop()

# Parsear y limpiar fechas
df['cotization_date'] = pd.to_datetime(df['cotization_date'], errors='coerce')

# Limpieza y cálculo price_per_unit
df = df[~df['Price'].astype(str).str.upper().str.contains('PAS', na=False)]
df['Price'] = pd.to_numeric(
    df['Price'].astype(str).str.replace(r'[\$,]', '', regex=True),
    errors='coerce'
)
df = df.dropna(subset=['Price'])
df['volume_standard'] = pd.to_numeric(df['volume_standard'], errors='coerce').fillna(1)
df['volume_unit'] = df['volume_unit'].astype(str)
df['price_per_unit'] = df['Price'] / df['volume_standard']

# ------------------------
# Sidebar filters (dinámicos)
# ------------------------
st.sidebar.header("Quotation Filters")

# 1) Product
product = st.sidebar.selectbox(
    "Product",
    sorted(df['Product'].dropna().unique())
)

# 2) Location dinámico
locs_for_product = df[df['Product'] == product]['Location'].dropna().unique()
locations = st.sidebar.multiselect(
    "Location",
    options=sorted(locs_for_product),
    default=sorted(locs_for_product)
)

# 3) Organic Status dinámico
sub = df[df['Product'] == product]
if locations:
    sub = sub[sub['Location'].isin(locations)]
org_vals = sub['Organic'].dropna().unique().tolist()
org_map = {0: 'Conventional', 1: 'Organic'}
org_options = ['All'] + [org_map[o] for o in sorted(org_vals)]
organic = st.sidebar.selectbox(
    "Organic Status",
    org_options
)

# 4) Volume Unit dinámico
sub2 = sub.copy()
if organic != 'All':
    sub2 = sub2[sub2['Organic'] == organic_to_num(organic)]
vu_opts = sorted(sub2['volume_unit'].dropna().unique())
volume_unit = st.sidebar.selectbox(
    "Volume Unit",
    ['All'] + vu_opts
)

# 5) Date range robusto
# Tomar solo fechas válidas
valid_dates = df['cotization_date'].dropna().dt.date
if not valid_dates.empty:
    min_date = valid_dates.min()
    max_date = valid_dates.max()
else:
    st.error("No hay fechas válidas en los datos.")
    st.stop()

date_range = st.sidebar.date_input(
    "Rango de Fecha",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Normalizar salida de date_input en start_date, end_date
if isinstance(date_range, (tuple, list)):
    start_date, end_date = date_range[0], date_range[-1]
else:
    start_date = end_date = date_range

# ------------------------
# Aplicar filtros
# ------------------------
g = df[df['Product'] == product]

if locations:
    g = g[g['Location'].isin(locations)]

if organic != 'All':
    g = g[g['Organic'] == organic_to_num(organic)]

if volume_unit != 'All':
    g = g[g['volume_unit'] == volume_unit]

# Filtrar rango de fecha, ignorando NaT
mask_date = (
    g['cotization_date'].notna() &
    g['cotization_date'].dt.date.between(start_date, end_date)
)
g = g[mask_date]

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
    # Preparar tabla con la fecha formateada
    display = g.rename(columns={
        'cotization_date': 'Date',
        'Product': 'Product',
        'Location': 'Location',
        'volume_unit': 'Volume Unit',
        'price_per_unit': 'Price per Unit',
        'VendorClean': 'Vendor'
    })[['Date','Product','Location','Volume Unit','Price per Unit','Vendor']]

    # Formatear Date dd/mm/yy
    display['Date'] = display['Date'].dt.strftime("%d/%m/%y")
    # Formatear price
    display['Price per Unit'] = display['Price per Unit'].map(lambda x: f"${x:.2f}")

    # Ordenar de más reciente a más antiguo, luego Vendor
    display = display.sort_values(by=['Date','Vendor'], ascending=[False, True])

    st.subheader("Filtered Quotations")
    st.dataframe(display)

    # Métricas clave
    st.subheader("Key Metrics")
    min_val = g['price_per_unit'].min()
    max_val = g['price_per_unit'].max()
    avg_val = g['price_per_unit'].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Min Price/Unit", f"${min_val:.2f}")
    c2.metric("Max Price/Unit", f"${max_val:.2f}")
    c3.metric("Avg Price/Unit", f"${avg_val:.2f}")

    # Gráfico medio por vendor
    st.subheader("Average Price/Unit by Vendor")
    avg_vendor = g.groupby('VendorClean')['price_per_unit'].mean().reset_index()
    chart = alt.Chart(avg_vendor).mark_bar(color=BRAND_GREEN).encode(
        x=alt.X('VendorClean:N', title='Vendor'),
        y=alt.Y('price_per_unit:Q', title='Avg Price/Unit')
    )
    st.altair_chart(chart, use_container_width=True)

    # Recomendación: vendor más barato (min price_per_unit)
    best = avg_vendor.loc[avg_vendor['price_per_unit'].idxmin()]
    st.success(f"**Vendor recomendado:** {best['VendorClean']} a ${best['price_per_unit']:.2f} por unidad")
