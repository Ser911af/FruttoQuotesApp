import streamlit as st 
import pandas as pd
import os
import re

st.set_page_config(page_title="FruttoFoods Daily Sheet", layout="wide")

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

# Logo centrado arriba
colA, colB, colC = st.columns([1, 2, 1])
with colB:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.info("Logo no encontrado. Verifica 'data/Asset 7@4x.png'.")

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

# ---------- Modo edición de precios ----------
st.divider()
edit_mode = st.toggle("✏️ Modo edición de precios", value=False, help="Habilita edición inline SOLO de la columna Price.")

if edit_mode:
    # Editor con columnas reales; SOLO price es editable
    editable_cols = ["price"]

    edit_df = day_df[["id", "cotization_date", "VendorClean", "Location", "Product", "Price"]].copy()
    edit_df = edit_df.rename(columns={
        "VendorClean": "Shipper",
        "Location": "Where",
        "Price": "price"  # editor trabaja con 'price' numérica
    })

    col_config = {
        "id": st.column_config.TextColumn("ID", help="Clave del registro", disabled=True),
        "cotization_date": st.column_config.DatetimeColumn("Date", format="MM/DD/YYYY", disabled=True),
        "Shipper": st.column_config.TextColumn("Shipper", disabled=True),
        "Where": st.column_config.TextColumn("Where", disabled=True),
        "Product": st.column_config.TextColumn("Product", disabled=True),
        "price": st.column_config.NumberColumn("Price (numérico)", min_value=0.0, step=0.01, help="Ingresa solo números (sin $)"),
    }

    st.caption("Edita el **Price** y luego presiona **Guardar cambios**.")
    edited_df = st.data_editor(
        edit_df,
        key="editor_prices",
        num_rows="fixed",
        use_container_width=True,
        column_config=col_config,
        column_order=["id", "cotization_date", "Shipper", "Where", "Product", "price"]
    )

    if st.button("💾 Guardar cambios", type="primary", use_container_width=True):
        # Detectar cambios en 'price' por fila
        orig = edit_df.set_index("id")[editable_cols]
        new  = edited_df.set_index("id")[editable_cols]

        changed_mask = (orig != new) & ~(orig.isna() & new.isna())
        dirty_ids = new.index[changed_mask.any(axis=1)].tolist()

        if not dirty_ids:
            st.success("No hay cambios por guardar.")
        else:
            payload = []
            for _id in dirty_ids:
                new_price = new.loc[_id, "price"]
                try:
                    new_price = float(new_price)
                except Exception:
                    continue
                payload.append({"id": _id, "price": new_price})

            if not payload:
                st.info("No se detectaron precios válidos para actualizar.")
            else:
                try:
                    from supabase import create_client
                    SUPABASE_URL = st.secrets["SUPABASE_URL"]
                    SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
                    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

                    # update por id (fila a fila)
                    for item in payload:
                        _id = item["id"]
                        sb.table("quotations").update({"price": item["price"]}).eq("id", _id).execute()

                    st.success(f"Se actualizaron {len(payload)} precio(s). 🎉")
                    st.balloons()

                    # Refrescar en memoria para que la vista muestre el nuevo Price$
                    id_to_price = {p["id"]: p["price"] for p in payload}
                    mask = day_df["id"].isin(id_to_price.keys())
                    day_df.loc[mask, "Price"] = day_df.loc[mask, "id"].map(id_to_price)
                    day_df["Price$"] = day_df["Price"].apply(_format_price)

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
