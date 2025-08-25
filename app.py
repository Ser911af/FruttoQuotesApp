import streamlit as st
import pandas as pd
import os
import altair as alt
import re

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
        # No hacemos st.stop() aquí para que al menos se dibujen tabs/headers
        return pd.DataFrame()

    # 1) Cliente
    try:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
    except Exception:
        st.error("No encontré SUPABASE_URL y/o SUPABASE_ANON_KEY en secrets. Revísalos.")
        return pd.DataFrame()

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

    # 3) Validación de columnas mínimas
    expected_cols = [
        "cotization_date","organic","product","price","location",
        "volume_num","volume_unit","volume_standard","vendorclean"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en Supabase: {missing}")
        return pd.DataFrame()

    # 4) Normalización de tipos y nombres para que la UI actual funcione
    # Fechas ("M/D/YYYY" como texto en la tabla) -> datetime
    df["cotization_date"] = pd.to_datetime(df["cotization_date"], errors="coerce")

    # Organic 0/1 (num) -> 'Organic' (numérica) para la UI
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

# ===== Helpers para Daily Sheet =====
_size_regex = re.compile(
    r"(\d+\s?lb|\d+\s?ct|\d+\s?[xX]\s?\d+|bulk|jbo|xl|lg|med|fancy|4x4|4x5|5x5|60cs)",
    flags=re.IGNORECASE
)

def _family_from_product(p: str) -> str:
    s = (p or "").lower()
    if any(k in s for k in ["tomato", "roma", "round", "grape"]):
        return "Tomato"
    if any(k in s for k in ["squash", "zucchini", "gray"]):
        return "Soft Squash"
    if "cucumber" in s or "cuke" in s:
        return "Cucumbers"
    if any(k in s for k in ["pepper", "bell", "jalape", "habanero", "serrano"]):
        return "Bell Peppers"
    return "Others"

def _size_from_product(p: str) -> str:
    p = p or ""
    m = _size_regex.search(p)
    return m.group(1) if m else ""

def _ogcv(x) -> str:
    try:
        xi = int(x)
        return "OG" if xi == 1 else "CV" if xi == 0 else ""
    except Exception:
        s = str(x).strip().lower()
        return "OG" if s in ("organic","org","1","true","sí","si","yes","y") else "CV" if s in ("conventional","conv","0","false","no","n") else ""

def _volume_str(row) -> str:
    q = row.get("volume_num")
    u = row.get("volume_unit") or ""
    try:
        q = float(q)
        q = int(q) if q.is_integer() else q
    except Exception:
        q = ""
    return f"{q} {u}".strip()

def _format_price(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return ""

# ------------------------
# Carga de datos (desde Supabase)
# ------------------------
df = fetch_all_quotations_from_supabase()

# ------------------------
# Layout y logo (SIEMPRE se pintan)
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
# Tabs: Explorer + Daily Sheet (SIEMPRE se pintan)
# ------------------------
tab1, tab2 = st.tabs(["Explorer", "Daily Sheet"])

# ------------------------
# Sidebar filters (dinámicos) — se muestran aunque df esté vacío
# ------------------------
st.sidebar.header("Quotation Filters")

if df is None or df.empty:
    st.sidebar.info("Datos no disponibles todavía.")
else:
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
# Tab 1: Explorer (tu vista actual)
# ------------------------
with tab1:
    if df is None or df.empty:
        st.warning("No hay datos disponibles desde Supabase.")
    else:
        # Aplicar filtros
        if not products:
            g = df.copy()
        else:
            g = df[df['Product'].isin(products)]

        if locations:
            g = g[g['Location'].isin(locations)]

        if 'organic' in locals() and organic != 'All':
            g = g[g['Organic'] == organic_to_num(organic)]

        if 'volume_unit' in locals() and volume_unit != 'All':
            g = g[g['volume_unit'] == volume_unit]

        # Mostrar resultados
        if g.empty:
            st.warning("No hay datos para los filtros seleccionados.")
        else:
            display = g.rename(columns={
                'cotization_date': 'Date',
                'volume_unit': 'Volume Unit',
                'price_per_unit': 'Price per Unit',
                'VendorClean': 'Vendor'
            })[['Date', 'Product', 'Location', 'Volume Unit', 'Price per Unit', 'Vendor']]

            display['Date'] = pd.to_datetime(display['Date'], errors='coerce')
            display = display.sort_values(by=['Date'], ascending=[False])
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

# ------------------------
# Tab 2: Daily Sheet (único grid + filtros)
# ------------------------
with tab2:
    st.header("🔥💥💰 Daily Sheet (Vista del día) 💰💥🔥")

    if df is None or df.empty:
        st.info("No hay datos disponibles desde Supabase.")
    else:
        # Columna de fecha normalizada para filtros del día
        df["_date"] = pd.to_datetime(df["cotization_date"], errors="coerce").dt.date

        if df["_date"].dropna().empty:
            st.info("No se pudo interpretar ninguna fecha en 'cotization_date'. Revisa el formato (M/D/YYYY).")
        else:
            default_date = max(d for d in df["_date"] if pd.notna(d))
            sel_date = st.date_input("Fecha a mostrar", value=default_date)

            # Base del día usando la columna ya preparada
            day_df = df[df["_date"] == sel_date].copy()

            if day_df.empty:
                st.warning("No hay cotizaciones para la fecha seleccionada.")
            else:
                # Derivados para la vista
                day_df["Shipper"] = day_df["VendorClean"]
                day_df["OG/CV"]   = day_df["Organic"].apply(lambda x: "OG" if pd.notna(x) and int(x)==1 else ("CV" if pd.notna(x) and int(x)==0 else ""))
                day_df["Where"]   = day_df["Location"]
                day_df["Size"]    = day_df["Product"].apply(_size_from_product)
                day_df["Volume?"] = day_df.apply(_volume_str, axis=1)
                day_df["Price$"]  = day_df["Price"].apply(_format_price)
                day_df["Family"]  = day_df["Product"].apply(_family_from_product)

                # Filtros del grid
                cols = st.columns(4)
                with cols[0]:
                    fams = ["Tomato","Soft Squash","Cucumbers","Bell Peppers","Others"]
                    sel_fams = st.multiselect("Familias", options=fams, default=fams)
                with cols[1]:
                    locs = sorted([x for x in day_df["Where"].dropna().unique().tolist() if x != ""])
                    sel_locs = st.multiselect("Ubicaciones", options=locs, default=locs)
                with cols[2]:
                    search = st.text_input("Buscar producto (contiene)", "")
                with cols[3]:
                    sort_opt = st.selectbox("Ordenar por", ["Product", "Shipper", "Where", "Price (asc)", "Price (desc)"])

                # Aplicar filtros
                day_df = day_df[day_df["Family"].isin(sel_fams)]
                if sel_locs:
                    day_df = day_df[day_df["Where"].isin(sel_locs)]
                if search.strip():
                    s = search.strip().lower()
                    day_df = day_df[day_df["Product"].str.lower().str.contains(s, na=False)]

                # Orden
                if sort_opt == "Price (asc)":
                    day_df = day_df.sort_values("Price", ascending=True)
                elif sort_opt == "Price (desc)":
                    day_df = day_df.sort_values("Price", ascending=False)
                else:
                    day_df = day_df.sort_values(sort_opt)

                # Grid final
                show = day_df[["Shipper","Where","OG/CV","Product","Size","Volume?","Price$", "Family"]].reset_index(drop=True)
                st.dataframe(show, use_container_width=True)

                # Descarga CSV
                csv_bytes = show.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Descargar CSV (vista del día)",
                    data=csv_bytes,
                    file_name=f"daily_sheet_{sel_date}.csv",
                    mime="text/csv"
                )
