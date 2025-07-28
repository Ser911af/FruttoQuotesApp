import streamlit as st
import pandas as pd
import os
import altair as alt

# ------------------------
# FruttoFoods Quotation Tool
# ------------------------

LOGO_PATH    = "data/Asset 7@4x.png"
BRAND_GREEN  = "#8DC63F"
data_path    = "data/market_cleaned_cleaned.xlsx"

# Carga de datos con verificación
if os.path.exists(data_path):
    df = pd.read_excel(data_path)
else:
    st.error(f"Archivo no encontrado: {data_path}")
    st.stop()

# Limpieza y cálculo de price_per_unit
df = df[~df['Price'].astype(str).str.upper().str.contains('PAS', na=False)]
df['Price'] = pd.to_numeric(df['Price'].astype(str)
                            .str.replace(r'[\$,]', '', regex=True),
                            errors='coerce')
df = df.dropna(subset=['Price'])
df['volume_standard'] = pd.to_numeric(df['volume_standard'],
                                      errors='coerce').fillna(1)
df['volume_unit'] = df['volume_unit'].astype(str)
df['price_per_unit'] = df['Price'] / df['volume_standard']

# ------------------------
# Sidebar filters
# ------------------------
st.sidebar.header("Quotation Filters")

# 1) Product
product = st.sidebar.selectbox(
    "Product",
    sorted(df['Product'].dropna().unique())
)

# 2) Dynamic Location based on selected Product
available_locs = sorted(
    df[df['Product'] == product]['Location']
      .dropna()
      .unique()
)
locations = st.sidebar.multiselect(
    "Location",
    options=available_locs,
    default=available_locs
)

# 3) Otros filtros
organic = st.sidebar.selectbox(
    "Organic Status",
    ['All', 'Conventional', 'Organic']
)
volume_unit = st.sidebar.selectbox(
    "Volume Unit",
    ['All'] + sorted(df['volume_unit'].unique())
)

def organic_to_num(val):
    if val == 'Conventional': return 0
    if val == 'Organic':     return 1
    return None

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
    display = g.rename(columns={
        'Product': 'Product',
        'Location': 'Location',
        'volume_unit': 'Volume Unit',
        'price_per_unit': 'Price per Unit',
        'VendorClean': 'Vendor'
    })[['Product', 'Location', 'Volume Unit', 'Price per Unit', 'Vendor']]
    display['Price per Unit'] = display['Price per Unit'].map(lambda x: f"${x:.2f}")
    st.subheader("Filtered Quotations")
    st.dataframe(display)

    st.subheader("Key Metrics")
    min_val = g['price_per_unit'].min()
    max_val = g['price_per_unit'].max()
    avg_val = g['price_per_unit'].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Min Price/Unit", f"${min_val:.2f}")
    c2.metric("Max Price/Unit", f"${max_val:.2f}")
    c3.metric("Avg Price/Unit", f"${avg_val:.2f}")

    st.subheader("Average Price/Unit by Vendor")
    avg_vendor = g.groupby('VendorClean')['price_per_unit'].mean().reset_index()
    chart = alt.Chart(avg_vendor).mark_bar(color=BRAND_GREEN).encode(
        x=alt.X('VendorClean:N', title='Vendor'),
        y=alt.Y('price_per_unit:Q', title='Avg Price/Unit')
    )
    st.altair_chart(chart, use_container_width=True)

    best = avg_vendor.loc[avg_vendor['price_per_unit'].idxmax()]
    st.success(f"**Vendor recomendado:** {best['VendorClean']} a ${best['price_per_unit']:.2f} por unidad")
