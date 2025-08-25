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
        st.image(LOGO_PATH, use_container_width=True)  # evita warning
    else:
        st.info("Logo no encontrado. Verifica 'data/Asset 7@4x.png'.")

df = fetch_all_quotations_from_supabase()

# Si no hay datos, dejamos la página “suave”
if df.empty:
    st.info("Sin datos disponibles desde Supabase por ahora.")
    st.caption("Página en construcción — pronto agregamos la vista del día.")
    st.stop()

# Columna fecha normalizada para filtros del día
df["_date"] = pd.to_datetime(df["cotization_date"], errors="coerce").dt.date
valid_dates = df["_date"].dropna()
if valid_dates.empty:
    st.warning("No se pudo interpretar ninguna fecha en 'cotization_date'. Revisa el formato fuente.")
    st.stop()

# Selector de fecha (default: la más reciente)
default_date = max(valid_dates)
sel_date = st.date_input("Fecha a mostrar", value=default_date)

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

# Filtros de la vista del día
cols = st.columns(4)
with cols[0]:
    fams = ["Tomato", "Soft Squash", "Cucumbers", "Bell Peppers", "Others"]
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
