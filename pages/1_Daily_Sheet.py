import streamlit as st 
import pandas as pd
import os
import re

# ---- Altair opcional (con detección) ----
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

st.set_page_config(page_title="FruttoFoods Daily Sheet", layout="wide")

# ---- Versión visible para confirmar despliegue ----
VERSION = "Daily_Sheet v2025-09-03 - visuals ON"
st.caption(VERSION)

LOGO_PATH = "data/Asset 7@4x.png"

# ------------------------
# Helpers (parsing y formateo)
# ------------------------
_size_regex = re.compile(
    r"(\d+\s?lb|\d+\s?ct|\d+\s?[xX]\s?\d+|bulk|jbo|xl|lg|med|fancy|4x4|4x5|5x5|60cs)",
    flags=re.IGNORECASE
)

def _size_from_product(p: str) -> str:
    if not isinstance(p, str):
        return ""
    m = _size_regex.search(p)
    return m.group(1) if m else ""

def _ogcv(x) -> str:
    try:
        xi = int(x)
        return "OG" if xi == 1 else "CV" if xi == 0 else ""
    except Exception:
        s = str(x).strip().lower()
        return "OG" if s in ("organic","org","1","true","sí","si","yes","y") else \
               "CV" if s in ("conventional","conv","0","false","no","n") else ""

def _volume_str(row) -> str:
    q = row.get("volume_num")
    u = (row.get("volume_unit") or "").strip()
    try:
        q = float(q)
        q = int(q) if float(q).is_integer() else q
    except Exception:
        q = ""
    return f"{q} {u}".strip()

def _format_price(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return ""

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

# ------------------------
# Data fetch (Supabase)
# ------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_quotations_from_supabase():
    """Trae quotations paginado; ante errores devuelve DF vacío (no rompe UI)."""
    try:
        from supabase import create_client
    except Exception as e:
        st.error(f"Falta 'supabase' en requirements.txt: {e}")
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
        start, end = i * page_size, i * page_size + page_size - 1
        try:
            resp = (
                sb.table("quotations")
                  .select("id,cotization_date,organic,product,price,location,volume_num,volume_unit,volume_standard,vendorclean")
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

    # Normalización mínima
    df["cotization_date"] = pd.to_datetime(df["cotization_date"], errors="coerce")
    df["Organic"] = pd.to_numeric(df["organic"], errors="coerce").astype("Int64")
    df["Price"]   = pd.to_numeric(df["price"], errors="coerce")
    df["volume_unit"] = df["volume_unit"].astype(str).fillna("unit")
    df = df.rename(columns={"product":"Product","location":"Location","vendorclean":"VendorClean"})
    return df

# ------------------------
# UI
# ------------------------
st.title("Daily Sheet")

# Logo centrado + utilidades
colA, colB, colC = st.columns([1, 2, 1])
with colB:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.info("Logo no encontrado. Verifica 'data/Asset 7@4x.png'.")

cc1, cc2 = st.columns(2)
with cc1:
    if st.button("🧹 Limpiar caché de datos"):
        st.cache_data.clear()
        st.success("Caché limpiada. Vuelve a cargar la página o usa 'Forzar recarga'.")
with cc2:
    if st.button("🔄 Forzar recarga (rerun)"):
        st.rerun()

df = fetch_all_quotations_from_supabase()

if df.empty:
    st.info("Sin datos disponibles desde Supabase por ahora.")
    st.caption("Página en construcción — pronto agregamos la vista del día.")
    st.stop()

# Columna fecha normalizada
df["_date"] = pd.to_datetime(df["cotization_date"], errors="coerce").dt.date
valid_dates = df["_date"].dropna()
if valid_dates.empty:
    st.warning("No se pudo interpretar ninguna fecha en 'cotization_date'.")
    st.stop()

# Selector de fecha en formato mm/dd/yyyy
default_date = max(valid_dates)
sel_date = st.date_input("Fecha a mostrar", value=default_date, format="MM/DD/YYYY")

# Subset del día
day_df = df[df["_date"] == sel_date].copy()
if day_df.empty:
    st.warning("No hay cotizaciones para la fecha seleccionada.")
    st.stop()

# Campos derivados
day_df["Shipper"] = day_df["VendorClean"]
day_df["OG/CV"]   = day_df["Organic"].apply(_ogcv)
day_df["Where"]   = day_df["Location"]
day_df["Size"]    = day_df["Product"].apply(_size_from_product)
day_df["Volume?"] = day_df.apply(_volume_str, axis=1)
day_df["Price$"]  = day_df["Price"].apply(_format_price)
day_df["Family"]  = day_df["Product"].apply(_family_from_product)
day_df["Date"]    = pd.to_datetime(day_df["cotization_date"], errors="coerce").dt.strftime("%m/%d/%Y")

# ---------- Filtros de la vista del día ----------
cols = st.columns(4)

# 1) Productos disponibles (reemplaza "Familias")
with cols[0]:
    product_options = sorted([x for x in day_df["Product"].dropna().unique().tolist() if str(x).strip() != ""])
    sel_products = st.multiselect("Productos (disponibles)", options=product_options, default=product_options)

# 2) Ubicaciones
with cols[1]:
    locs = sorted([x for x in day_df["Where"].dropna().unique().tolist() if str(x).strip() != ""])
    sel_locs = st.multiselect("Ubicaciones", options=locs, default=locs)

# 3) Búsqueda por texto en Product
with cols[2]:
    search = st.text_input("Buscar producto (contiene)", "")

# 4) Orden
with cols[3]:
    sort_opt = st.selectbox("Ordenar por", ["Product", "Shipper", "Where", "Price (asc)", "Price (desc)"])

# ---- Aplicar filtros ----
if sel_products:
    day_df = day_df[day_df["Product"].isin(sel_products)]
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

# ---------- Modo edición (todas las variables excepto la fecha) ----------
st.divider()
edit_mode = st.toggle(
    "✏️ Modo edición (todo excepto fecha)",
    value=False,
    help="Edita Shipper, Where, Product, OG/CV, Price, Volume Qty/Unit. La fecha permanece bloqueada."
)

if edit_mode:
    # TODAS las columnas reales excepto fecha
    editable_cols = ["VendorClean", "Location", "Product", "organic", "Price", "volume_num", "volume_unit"]

    edit_df = day_df[["id", "cotization_date"] + editable_cols].copy()
    # Renombrar para que el editor sea legible (y luego revertimos antes de guardar)
    edit_df = edit_df.rename(columns={
        "VendorClean": "Shipper",
        "Location": "Where",
        "Price": "price"  # editor trabaja con numérico simple
    })

    col_config = {
        "id": st.column_config.TextColumn("ID", disabled=True),
        "cotization_date": st.column_config.DatetimeColumn("Date", format="MM/DD/YYYY", disabled=True),
        "Shipper": st.column_config.TextColumn("Shipper"),
        "Where": st.column_config.TextColumn("Where"),
        "Product": st.column_config.TextColumn("Product"),
        "organic": st.column_config.NumberColumn("OG/CV (1=OG,0=CV)", min_value=0, max_value=1, step=1),
        "price": st.column_config.NumberColumn("Price", min_value=0.0, step=0.01),
        "volume_num": st.column_config.NumberColumn("Volume Qty", min_value=0.0, step=0.01),
        "volume_unit": st.column_config.TextColumn("Volume Unit"),
    }

    st.caption("Edita los campos y presiona **Guardar cambios**.")
    edited_df = st.data_editor(
        edit_df,
        key="editor_all",
        num_rows="fixed",
        use_container_width=True,
        column_config=col_config,
        column_order=["id","cotization_date","Shipper","Where","Product","organic","price","volume_num","volume_unit"]
    )

    if st.button("💾 Guardar cambios", type="primary", use_container_width=True):
        orig = edit_df.set_index("id")[["Shipper","Where","Product","organic","price","volume_num","volume_unit"]]
        new  = edited_df.set_index("id")[["Shipper","Where","Product","organic","price","volume_num","volume_unit"]]

        changed_mask = (orig != new) & ~(orig.isna() & new.isna())
        dirty_ids = new.index[changed_mask.any(axis=1)].tolist()

        if not dirty_ids:
            st.success("No hay cambios por guardar.")
        else:
            payload = []
            for _id in dirty_ids:
                row = new.loc[_id].to_dict()

                # Conversión mínima segura
                for k in ["price", "volume_num"]:
                    try:
                        if row.get(k) not in (None, ""):
                            row[k] = float(row[k])
                    except Exception:
                        pass
                try:
                    if row.get("organic") not in (None, ""):
                        row["organic"] = int(row["organic"])
                except Exception:
                    pass

                # Revertir nombres a columnas reales de la tabla
                row["VendorClean"] = row.pop("Shipper", None)
                row["Location"]    = row.pop("Where", None)
                row["Price"]       = row.pop("price", None)

                # Quitar Nones para no sobreescribir con nulls
                clean = {k: v for k, v in row.items() if v is not None}
                payload.append({"id": _id, **clean})

            try:
                from supabase import create_client
                SUPABASE_URL = st.secrets["SUPABASE_URL"]
                SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
                sb = create_client(SUPABASE_URL, SUPABASE_KEY)

                for item in payload:
                    _id = item.pop("id")
                    sb.table("quotations").update(item).eq("id", _id).execute()

                st.success(f"Se actualizaron {len(payload)} registro(s). 🎉")
                st.balloons()

                # Refrescar en memoria
                upd = new.loc[dirty_ids].reset_index()
                upd = upd.rename(columns={"Shipper":"VendorClean","Where":"Location","price":"Price"})
                for _, r in upd.iterrows():
                    row_mask = day_df["id"] == r["id"]
                    for col in ["VendorClean","Location","Product","organic","Price","volume_num","volume_unit"]:
                        if col in r and pd.notna(r[col]):
                            day_df.loc[row_mask, col] = r[col]
                day_df["Shipper"] = day_df["VendorClean"]
                day_df["Where"]   = day_df["Location"]
                day_df["Price$"]  = day_df["Price"].apply(_format_price)

            except Exception as e:
                st.error(f"Error al guardar cambios: {e}")

# ---------- Vista de la tabla (solo lectura bonita) ----------
show = day_df[["Date","Shipper","Where","OG/CV","Product","Size","Volume?","Price$", "Family"]].reset_index(drop=True)
st.dataframe(show, use_container_width=True)

# Descarga CSV con fecha formateada
csv_bytes = show.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar CSV (vista del día)",
    data=csv_bytes,
    file_name=f"daily_sheet_{sel_date.strftime('%m-%d-%Y')}.csv",
    mime="text/csv"
)

# =========================
# Visualizaciones (filtros independientes)
# =========================
st.markdown("## 📊 Visualizaciones")
st.info("Marcador: entramos a la sección de visualizaciones.")

if not ALTAIR_OK:
    st.warning("Altair no está instalado. Agrega `altair>=5` a requirements.txt y reinicia la app.")
else:
    # Base para visualizaciones: TODO el histórico (no solo el día)
    viz_df = df.copy()

    # Normalizaciones necesarias
    viz_df["date_only"] = pd.to_datetime(viz_df["cotization_date"], errors="coerce").dt.date
    viz_df["price_num"] = pd.to_numeric(viz_df["Price"], errors="coerce")
    viz_df["volume_num"] = pd.to_numeric(viz_df["volume_num"], errors="coerce")

    # -------- Filtros independientes --------
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1.2])

    with c1:
        prod_opts = sorted([x for x in viz_df["Product"].dropna().unique().tolist() if str(x).strip() != ""])
        sel_prod = st.selectbox("Producto", options=["(selecciona uno)"] + prod_opts, index=0)

    with c2:
        # Rango de fechas
        min_d, max_d = viz_df["date_only"].min(), viz_df["date_only"].max()
        start_d, end_d = st.date_input(
            "Rango de fechas",
            value=(min_d or default_date, max_d or default_date),
            format="MM/DD/YYYY"
        )

    with c3:
        loc_opts = sorted([x for x in viz_df["Location"].dropna().unique().tolist() if str(x).strip() != ""])
        sel_locs_v = st.multiselect("Ubicaciones (viz)", options=loc_opts, default=loc_opts)

    with c4:
        ship_opts = sorted([x for x in viz_df["VendorClean"].dropna().unique().tolist() if str(x).strip() != ""])
        sel_ships_v = st.multiselect("Shippers (viz)", options=ship_opts, default=ship_opts)

    # Aplicar filtros
    fdf = viz_df.copy()
    if sel_prod != "(selecciona uno)":
        fdf = fdf[fdf["Product"] == sel_prod]
    if sel_locs_v:
        fdf = fdf[fdf["Location"].isin(sel_locs_v)]
    if sel_ships_v:
        fdf = fdf[fdf["VendorClean"].isin(sel_ships_v)]
    if isinstance(start_d, tuple):  # por si el widget devuelve 2 fechas en una tupla
        start_d, end_d = start_d
    fdf = fdf[(fdf["date_only"] >= start_d) & (fdf["date_only"] <= end_d)]

    # ---- Smoke test para confirmar render ----
    st.markdown("#### 🔍 Test de render")
    st.write(f"Filas totales en df para viz: {len(viz_df)} | Filas tras filtros: {len(fdf)}")
    if len(viz_df) > 0:
        test_df = viz_df.head(50).copy()
        chart_test = alt.Chart(test_df).mark_point().encode(
            x=alt.X("date_only:T", title="Fecha"),
            y=alt.Y("price_num:Q", title="Precio"),
            tooltip=["Product","Location","VendorClean","price_num"]
        ).properties(title="(Test) puntos precio vs fecha", height=200)
        st.altair_chart(chart_test, use_container_width=True)
    else:
        st.write("Sin datos para test.")

    # ---- Graficas solo si hay producto elegido ----
    if sel_prod == "(selecciona uno)":
        st.info("Selecciona un **Producto** para habilitar las visualizaciones.")
    else:
        if fdf.empty:
            st.warning("Sin datos para los filtros seleccionados.")
        else:
            # -------- 1) Precio promedio diario por ubicación (línea) --------
            g1 = (fdf
                  .groupby(["date_only","Location"], as_index=False)
                  .agg(avg_price=("price_num","mean")))

            chart1 = alt.Chart(g1).mark_line(point=True).encode(
                x=alt.X("date_only:T", title="Fecha"),
                y=alt.Y("avg_price:Q", title="Precio promedio"),
                color=alt.Color("Location:N", title="Ubicación"),
                tooltip=[alt.Tooltip("date_only:T","Fecha"), "Location:N", alt.Tooltip("avg_price:Q", format=".2f")]
            ).properties(title=f"Precio promedio diario — {sel_prod}", height=300)

            st.altair_chart(chart1, use_container_width=True)

            # -------- 2) Distribución de precios por ubicación (caja o barras) --------
            if fdf["Location"].nunique() > 1 and len(fdf) >= 10:
                chart2 = alt.Chart(fdf).mark_boxplot().encode(
                    x=alt.X("Location:N", title="Ubicación"),
                    y=alt.Y("price_num:Q", title="Precio"),
                    color=alt.Color("Location:N", legend=None)
                ).properties(title="Distribución de precios por ubicación", height=320)
            else:
                g2 = fdf.groupby("Location", as_index=False).agg(avg_price=("price_num","mean"))
                chart2 = alt.Chart(g2).mark_bar().encode(
                    x=alt.X("Location:N", title="Ubicación"),
                    y=alt.Y("avg_price:Q", title="Precio promedio"),
                    tooltip=["Location:N", alt.Tooltip("avg_price:Q", format=".2f")]
                ).properties(title="Precio promedio por ubicación", height=320)

            st.altair_chart(chart2, use_container_width=True)

            # -------- 3) Participación por shipper (volumen) --------
            g3 = (fdf.groupby("VendorClean", as_index=False)
                     .agg(total_volume=("volume_num","sum"))
                     .sort_values("total_volume", ascending=False)
                     .head(15))

            chart3 = alt.Chart(g3).mark_bar().encode(
                y=alt.Y("VendorClean:N", sort="-x", title="Shipper"),
                x=alt.X("total_volume:Q", title="Volumen total"),
                tooltip=["VendorClean:N", alt.Tooltip("total_volume:Q", format=",.0f")]
            ).properties(title="Top shippers por volumen (máx. 15)", height=350)

            st.altair_chart(chart3, use_container_width=True)

            # -------- 4) Volumen histórico del producto (línea) --------
            g4 = (fdf.groupby("date_only", as_index=False)
                     .agg(total_volume=("volume_num","sum")))

            chart4 = alt.Chart(g4).mark_line(point=True).encode(
                x=alt.X("date_only:T", title="Fecha"),
                y=alt.Y("total_volume:Q", title="Volumen total"),
                tooltip=[alt.Tooltip("date_only:T","Fecha"), alt.Tooltip("total_volume:Q", format=",.0f")]
            ).properties(title="Volumen total por día", height=300)

            st.altair_chart(chart4, use_container_width=True)
