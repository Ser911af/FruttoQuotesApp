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
VERSION = "Daily_Sheet v2025-09-09 - credenciales por secciones (supabase_quotes) + Volume"
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

def _choose_size(row) -> str:
    # 1) Prioriza size_text (talla/grade original)
    stxt = row.get("size_text")
    if isinstance(stxt, str) and stxt.strip():
        return stxt.strip()
    # 2) Fallback legacy: volume_standard
    vs = row.get("volume_standard")
    if isinstance(vs, str) and vs.strip():
        return vs.strip()
    # 3) Último recurso: inferir desde Product
    return _size_from_product(row.get("Product", ""))

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

def _norm_name(x: str) -> str:
    if not isinstance(x, str):
        return ""
    s = x.strip()
    return s[:1].upper() + s[1:].lower() if s else s

# ------------------------
# Supabase helpers (por secciones) — con fallbacks
# ------------------------
def _read_section(section_name: str = "supabase_quotes") -> dict:
    """
    Lee una sección de st.secrets (p.ej. 'supabase_quotes') y valida claves mínimas.
    También hace fallback a SUPABASE_URL / SUPABASE_ANON_KEY y variables de entorno.
    """
    try:
        sec = st.secrets[section_name]
    except Exception:
        sec = {}  # no existe la sección -> usar fallbacks

    url = (sec.get("url") if isinstance(sec, dict) else None) \
          or st.secrets.get("SUPABASE_URL") \
          or os.getenv("SUPABASE_URL")
    key = (sec.get("anon_key") if isinstance(sec, dict) else None) \
          or st.secrets.get("SUPABASE_ANON_KEY") \
          or st.secrets.get("SUPABASE_KEY") \
          or os.getenv("SUPABASE_ANON_KEY") \
          or os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError("No encontré credenciales de Supabase. Define [supabase_quotes] en secrets o usa SUPABASE_URL/SUPABASE_ANON_KEY.")

    table  = (sec.get("table") if isinstance(sec, dict) else None) or "quotations"
    schema = (sec.get("schema") if isinstance(sec, dict) else None) or "public"
    return {"url": url, "anon_key": key, "table": table, "schema": schema}

def _create_client(url: str, key: str):
    try:
        from supabase import create_client
    except Exception as e:
        raise ImportError(f"Falta 'supabase' en requirements.txt: {e}")
    return create_client(url, key)

def _sb_table(sb, schema: str, table: str):
    """Devuelve un handle a la tabla respetando el schema si el cliente lo soporta."""
    try:
        return sb.schema(schema).table(table)  # supabase-py v2+
    except Exception:
        return sb.table(table)                 # fallback v1

# ------------------------
# Data fetch (Supabase - usa sección supabase_quotes)
# ------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_quotations_from_supabase():
    """Trae quotations paginado desde la sección [supabase_quotes] o fallbacks."""
    try:
        cfg = _read_section("supabase_quotes")
        sb = _create_client(cfg["url"], cfg["anon_key"])
        tbl = _sb_table(sb, cfg["schema"], cfg["table"])
    except Exception as e:
        st.error(f"Config/cliente Supabase inválido: {e}")
        return pd.DataFrame()

    frames, page_size = [], 1000
    for i in range(1000):
        start, end = i * page_size, i * page_size + page_size - 1
        try:
            resp = (
                tbl.select(
                    "id,cotization_date,organic,product,price,location,"
                    "volume_num,volume_unit,volume_standard,vendorclean,"
                    "size_text"
                )
                .range(start, end)
                .execute()
            )
        except Exception as e:
            st.error(f"Error consultando Supabase: {e}")
            return pd.DataFrame()

        rows = getattr(resp, "data", None) or []
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
    if "size_text" not in df.columns:
        df["size_text"] = pd.NA
    df = df.rename(columns={
        "product":"Product",
        "location":"Location",
        "vendorclean":"VendorClean"
    })
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
day_df["Size"]    = day_df.apply(_choose_size, axis=1)
day_df["Volume"]  = day_df.apply(_volume_str, axis=1)   # ← Nombre final sin '?'
day_df["Price$"]  = day_df["Price"].apply(_format_price)
day_df["Family"]  = day_df["Product"].apply(_family_from_product)
day_df["Date"]    = pd.to_datetime(day_df["cotization_date"], errors="coerce").dt.strftime("%m/%d/%Y")

# ---------- Filtros de la vista del día ----------
cols = st.columns(4)
with cols[0]:
    product_options = sorted([x for x in day_df["Product"].dropna().unique().tolist() if str(x).strip() != ""])
    sel_products = st.multiselect("Productos (disponibles)", options=product_options, default=product_options)
with cols[1]:
    locs = sorted([x for x in day_df["Where"].dropna().unique().tolist() if str(x).strip() != ""])
    sel_locs = st.multiselect("Ubicaciones", options=locs, default=locs)
with cols[2]:
    search = st.text_input("Buscar producto (contiene)", "")
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

# ---------- Modo edición (Size edita size_text) ----------
st.divider()
edit_mode = st.toggle(
    "✏️ Modo edición (todo excepto fecha)",
    value=False,
    help="Edita Shipper, Where, Product, Size (size_text), OG/CV, Price, Volume Qty/Unit. La fecha permanece bloqueada."
)

if edit_mode:
    edit_df = day_df[[
        "id", "cotization_date", "VendorClean", "Location", "Product",
        "size_text", "Organic", "Price", "volume_num", "volume_unit"
    ]].copy()

    edit_df = edit_df.rename(columns={
        "VendorClean": "Shipper",
        "Location": "Where",
        "size_text": "Size",     # UI muestra "Size" pero se guarda en size_text
        "Organic": "organic",
        "Price": "price",
    })

    col_config = {
        "id": st.column_config.TextColumn("ID", disabled=True),
        "cotization_date": st.column_config.DatetimeColumn("Date", format="MM/DD/YYYY", disabled=True),
        "Shipper": st.column_config.TextColumn("Shipper"),
        "Where": st.column_config.TextColumn("Where"),
        "Product": st.column_config.TextColumn("Product"),
        "Size": st.column_config.TextColumn("Size (size_text)"),
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
        column_order=["id","cotization_date","Shipper","Where","Product","Size","organic","price","volume_num","volume_unit"]
    )

    if st.button("💾 Guardar cambios", type="primary", use_container_width=True):
        ORIG = edit_df.set_index("id")[["Shipper","Where","Product","Size","organic","price","volume_num","volume_unit"]]
        NEW  = edited_df.set_index("id")[["Shipper","Where","Product","Size","organic","price","volume_num","volume_unit"]]

        changed_mask = (ORIG != NEW) & ~(ORIG.isna() & NEW.isna())
        dirty_ids = NEW.index[changed_mask.any(axis=1)].tolist()

        if not dirty_ids:
            st.success("No hay cambios por guardar.")
        else:
            payload = []
            for _id in dirty_ids:
                ui_row = NEW.loc[_id].to_dict()

                # Tipos
                for k in ["price", "volume_num"]:
                    v = ui_row.get(k)
                    try:
                        ui_row[k] = float(v) if v not in (None, "") else None
                    except Exception:
                        pass
                v = ui_row.get("organic")
                try:
                    ui_row["organic"] = int(v) if v not in (None, "") else None
                except Exception:
                    pass

                # UI -> BD
                db_row = {
                    "vendorclean": ui_row.get("Shipper"),
                    "location": ui_row.get("Where"),
                    "product": ui_row.get("Product"),
                    "size_text": ui_row.get("Size"),
                    "organic": ui_row.get("organic"),
                    "price": ui_row.get("price"),
                    "volume_num": ui_row.get("volume_num"),
                    "volume_unit": ui_row.get("volume_unit"),
                }
                clean_db_row = {k: v for k, v in db_row.items() if v is not None}
                if not clean_db_row:
                    continue
                payload.append({"id": _id, **clean_db_row})

            try:
                cfg = _read_section("supabase_quotes")
                sb = _create_client(cfg["url"], cfg["anon_key"])
                tbl = _sb_table(sb, cfg["schema"], cfg["table"])

                for item in payload:
                    _id = item.pop("id")
                    tbl.update(item).eq("id", _id).execute()

                st.success(f"Se actualizaron {len(payload)} registro(s). 🎉")
                st.balloons()

                # Refrescar day_df local
                upd = NEW.loc[dirty_ids].reset_index()
                upd = upd.rename(columns={
                    "Shipper":"VendorClean",
                    "Where":"Location",
                    "price":"Price",
                    "Size":"size_text",
                })
                for _, r in upd.iterrows():
                    mask = day_df["id"] == r["id"]
                    for col in ["VendorClean","Location","Product","organic","Price","volume_num","volume_unit","size_text"]:
                        if col in r and pd.notna(r[col]):
                            day_df.loc[mask, col] = r[col]

                # Derivadas:
                day_df["Shipper"] = day_df["VendorClean"]
                day_df["Where"]   = day_df["Location"]
                day_df["Price$"]  = day_df["Price"].apply(_format_price)
                day_df["Size"]    = day_df.apply(_choose_size, axis=1)
                day_df["Volume"]  = day_df.apply(_volume_str, axis=1)

            except Exception as e:
                st.error(f"Error al guardar cambios: {e}")

# ---------- Vista de la tabla (solo lectura bonita) ----------
show = day_df[["Date","Shipper","Where","OG/CV","Product","Size","Volume","Price$", "Family"]].reset_index(drop=True)
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
# 📊 Visualizaciones (BASADAS en la tabla visible)
# =========================
st.markdown("## 📊 Visualizaciones (tabla actual)")

if not ALTAIR_OK:
    st.warning("Altair no está instalado. Agrega `altair>=5` a requirements.txt y reinicia la app.")
else:
    viz_day = day_df.copy()

    if viz_day.empty:
        st.info("No hay datos en la tabla actual para graficar.")
    else:
        viz_day["price_num"] = pd.to_numeric(viz_day["Price"], errors="coerce")
        viz_day["volume_num"] = pd.to_numeric(viz_day["volume_num"], errors="coerce")
        viz_day["Where_norm"] = viz_day["Where"].apply(_norm_name)
        viz_day["Shipper_norm"] = viz_day["Shipper"].apply(_norm_name)

        # ---- KPIs rápidos (de lo visible) ----
        c_k1, c_k2, c_k3, c_k4 = st.columns(4)
        with c_k1:
            mean_price = viz_day["price_num"].mean()
            st.metric("Precio promedio (tabla)", f"{mean_price:.2f}" if pd.notna(mean_price) else "—")
        with c_k2:
            min_price = viz_day["price_num"].min()
            st.metric("Precio mínimo (tabla)", f"{min_price:.2f}" if pd.notna(min_price) else "—")
        with c_k3:
            max_price = viz_day["price_num"].max()
            st.metric("Precio máximo (tabla)", f"{max_price:.2f}" if pd.notna(max_price) else "—")
        with c_k4:
            st.metric("Ofertas visibles", f"{len(viz_day)}")

        # ---- 1) Precio promedio por ubicación (barras) ----
        g_loc = (viz_day.groupby("Where_norm", as_index=False)
                        .agg(avg_price=("price_num","mean"),
                             offers=("Where_norm","count")))
        if not g_loc.empty:
            chart_loc = alt.Chart(g_loc).mark_bar().encode(
                x=alt.X("avg_price:Q", title="Precio promedio"),
                y=alt.Y("Where_norm:N", sort="-x", title="Ubicación"),
                tooltip=["Where_norm:N", alt.Tooltip("avg_price:Q", format=".2f"), "offers:Q"]
            ).properties(title="Precio promedio por ubicación (según tabla)", height=320)
            st.altair_chart(chart_loc, use_container_width=True)

        # ---- 2) Precio promedio por shipper (barras) ----
        g_ship = (viz_day.groupby("Shipper_norm", as_index=False)
                         .agg(avg_price=("price_num","mean"),
                              offers=("Shipper_norm","count")))
        if not g_ship.empty:
            chart_ship = alt.Chart(g_ship).mark_bar().encode(
                x=alt.X("avg_price:Q", title="Precio promedio"),
                y=alt.Y("Shipper_norm:N", sort="-x", title="Shipper"),
                tooltip=["Shipper_norm:N", alt.Tooltip("avg_price:Q", format=".2f"), "offers:Q"]
            ).properties(title="Precio promedio por shipper (según tabla)", height=350)
            st.altair_chart(chart_ship, use_container_width=True)

        # ---- 3) Volumen por shipper (barras) ----
        g_vol = (viz_day.groupby("Shipper_norm", as_index=False)
                        .agg(total_volume=("volume_num","sum")))
        g_vol = g_vol[g_vol["total_volume"].fillna(0) > 0]
        if not g_vol.empty:
            chart_vol = alt.Chart(g_vol).mark_bar().encode(
                x=alt.X("total_volume:Q", title="Volumen total"),
                y=alt.Y("Shipper_norm:N", sort="-x", title="Shipper"),
                tooltip=["Shipper_norm:N", alt.Tooltip("total_volume:Q", format=",.0f")]
            ).properties(title="Volumen por shipper (según tabla)", height=350)
            st.altair_chart(chart_vol, use_container_width=True)

        # ---- 4) Precio promedio por producto (barras) ----
        g_prod = (viz_day.groupby("Product", as_index=False)
                          .agg(avg_price=("price_num","mean"),
                               offers=("Product","count")))
        if not g_prod.empty and g_prod["Product"].nunique() > 1:
            chart_prod = alt.Chart(g_prod).mark_bar().encode(
                x=alt.X("avg_price:Q", title="Precio promedio"),
                y=alt.Y("Product:N", sort="-x", title="Producto"),
                tooltip=["Product:N", alt.Tooltip("avg_price:Q", format=".2f"), "offers:Q"]
            ).properties(title="Precio promedio por producto (según tabla)", height=350)
            st.altair_chart(chart_prod, use_container_width=True)

        # ---- 5) Scatter Precio vs Volumen (outliers) ----
        if viz_day["volume_num"].fillna(0).sum() > 0:
            scatter = alt.Chart(viz_day.dropna(subset=["price_num","volume_num"])).mark_circle(size=80).encode(
                x=alt.X("price_num:Q", title="Precio"),
                y=alt.Y("volume_num:Q", title="Volumen"),
                tooltip=["Product","Shipper_norm","Where_norm",
                         alt.Tooltip("price_num:Q", format=".2f"),
                         alt.Tooltip("volume_num:Q", format=",.0f")]
            ).properties(title="Precio vs Volumen (según tabla)", height=320)
            st.altair_chart(scatter, use_container_width=True)

        # ---- 6) Tabla de extremos ----
        with st.expander("🔎 Ver extremos de precio (tabla visible)"):
            tmp = viz_day[["Product","Shipper","Where","price_num","Volume"]].dropna(subset=["price_num"]).copy()
            tmp = tmp.sort_values("price_num")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Bottom 5 (más baratos)**")
                st.dataframe(tmp.head(5).rename(columns={"price_num":"Price"}), use_container_width=True)
            with c2:
                st.write("**Top 5 (más caros)**")
                st.dataframe(tmp.tail(5).rename(columns={"price_num":"Price"}), use_container_width=True)
