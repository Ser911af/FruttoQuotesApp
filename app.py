import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import altair as alt

# ------------------------
# FruttoFoods Quotation Tool
# ------------------------

# Brand assets and colors
LOGO_PATH = "data/Asset 7@4x.png"
BRAND_GREEN = "#8DC63F"  # Primary FruttoFoods green

# Load data
data_path = "data/market_cleaned.xlsx"

if os.path.exists(data_path):
    df = pd.read_excel(data_path)
else:
    st.error(f"Archivo no encontrado en la ruta: {data_path}")
    st.stop()

# Data cleaning
df = df[~df['Price'].astype(str).str.upper().str.contains('PAS', na=False)]
df['Price'] = pd.to_numeric(
    df['Price'].astype(str).str.replace(r'[\$,]', '', regex=True),
    errors='coerce'
)
df = df.dropna(subset=['Price'])
df['volume_standard'] = pd.to_numeric(df['volume_standard'], errors='coerce').fillna(1)
df['volume_unit'] = df['volume_unit'].astype(str)
df['price_per_unit'] = df['Price'] / df['volume_standard']

# Sidebar filters
st.sidebar.header("Quotation Filters")
product = st.sidebar.selectbox("Product", sorted(df['Product'].dropna().unique()))
location = st.sidebar.selectbox("Location", ['All'] + sorted(df['Location'].dropna().unique()))
organic = st.sidebar.selectbox("Organic Status", ['All', 'Conventional', 'Organic'])
volume_unit = st.sidebar.selectbox("Volume Unit", ['All'] + sorted(df['volume_unit'].unique()))

def organic_to_num(val):
    if val == 'Conventional': return 0
    if val == 'Organic': return 1
    return None

# Apply filters
g = df[df['Product'] == product]
if location != 'All':
    g = g[g['Location'] == location]
if organic != 'All':
    org_val = organic_to_num(organic)
    g = g[g['Organic'] == org_val]
if volume_unit != 'All':
    g = g[g['volume_unit'] == volume_unit]

# Layout header with logo
col1, col2 = st.columns([3,1])
with col1:
    st.title("FruttoFoods Quotation Tool")
with col2:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=80)
    else:
        st.warning("Logo no encontrado. Asegúrate de que 'data/Asset 7@4x.png' esté en el repositorio.")

# Display results
if g.empty:
    st.warning("No data available for the selected filters.")
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

    best = avg_vendor.loc[avg_vendor['price_per_unit'].idxmin()]
    st.success(f"**Suggested Vendor:** {best['VendorClean']} at ${best['price_per_unit']:.2f} per unit")

    st.subheader("Predictive Model: Price per Unit")
    if len(g) >= 5:
        model_df = df[['Product', 'Location', 'Organic', 'volume_unit', 'price_per_unit']].copy()
        X = model_df.drop('price_per_unit', axis=1)
        y = model_df['price_per_unit']
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Product', 'Location', 'volume_unit'])
        ], remainder='passthrough')
        model = Pipeline([
            ('pre', preprocessor),
            ('reg', LinearRegression())
        ])
        model.fit(X, y)
        inp = pd.DataFrame([{  
            'Product': product,
            'Location': location if location != 'All' else df['Location'].mode()[0],
            'Organic': organic_to_num(organic) if organic != 'All' else 0,
            'volume_unit': volume_unit if volume_unit != 'All' else df['volume_unit'].mode()[0]
        }])
        pred = model.predict(inp)[0]
        st.info(f"Estimated Price/Unit: ${pred:.2f}")
    else:
        st.info("Not enough data for reliable predictions.")
