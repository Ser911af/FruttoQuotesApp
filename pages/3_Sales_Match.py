import re
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Match Tester (Quotes ↔ Sales)", layout="wide")
st.title("🔗 Match Tester — Quotes ↔ Sales")
st.caption("Conexión a dos proyectos Supabase usando blocks [supabase_quotes] y [supabase_sales] en secrets.toml")

# ----------------------------
# Helpers de conexión/diagnóstico
# ----------------------------
def _mask_url_host(url: str) -> str:
    try:
        m = re.match(r"https?://([^/]+)/?", url)
        if not m:
            return "url-malformed"
        host = m.group(1)
        parts = host.split(".")
        if len(parts) > 2:
            parts[0] = parts[0][:3] + "…"  # enmascara el subdominio
        return ".".join(parts)
    except Exception:
        return "url-unknown"

def _load_block(name: str):
    """Carga un bloque tipo [supabase_quotes] o [supabase_sales] del secrets."""
    blk = st.secrets.get(name)
    if not blk:
        return None, None, None, None
    url = blk.get("url")
    key = blk.get("anon_key")
    table = blk.get("table")
    schema = blk.get("schema", "public")
    return url, key, schema, table

@st.cache_data(ttl=180, show_spinner=False)
def _connect_and_preview(block_name: str, select_cols: list[str] | None = None, max_rows: int = 10000):
    """Conecta a un bloque de secrets, lee columnas y devuelve DF + diagnóstico."""
    try:
        from supabase import create_client
    except Exception as e:
        return None, f"Falta el paquete 'supabase': {e}. Instala con: pip install supabase"

    url, key, schema, table = _load_block(block_name)
    if not url or not key or not table:
        return None, f"No encontré url/anon_key/table en [${block_name}] de secrets.toml"

    diag = f"{block_name}: host={_mask_url_host(url)} · schema={schema} · table={table}"
    try:
        sb = create_client(url, key)
    except Exception as e:
        return None, f"{diag} → Error creando cliente Supabase: {e}"

    # Prueba de conteo (puede requerir RLS)
    try:
        test = (sb.schema(schema)
                  .table(table)
                  .select("count:id", count="exact")
                  .limit(1)
                  .execute())
        total = getattr(test, "count", None)
        diag += f" · filas≈{total if total is not None else 'desconocido'}"
    except Exception as e:
        # No abortamos aún; puede que policies no permitan ese select
        diag += f" · (sin conteo: {e})"

    # Lectura
    try:
        sel = "*" if not select_cols else ",".join(select_cols)
        # Nota: si max_rows es grande y la tabla también, podrías paginar. Para prototipo leemos tope.
        resp = (sb.schema(schema).table(table).select(sel).limit(max_rows).execute())
        rows = resp.data or []
        df = pd.DataFrame(rows)
        return df, diag
    except Exception as e:
        return None, f"{diag} → Error leyendo datos: {e}"

# ----------------------------
# Cargar ambas fuentes
# ----------------------------
with st.expander("⚙️ Diagnóstico y configuración", expanded=True):
    st.write("Este panel muestra de dónde se están leyendo los datos y un conteo aproximado (si policies lo permiten).")

    # Sugerencias de columnas típicas (puedes ajustar)
    quote_default_cols = [
        "cotization_date","product","vendorclean","location","organic","price",
        "volume_standard","volume_unit"
    ]
    sales_default_cols = [
        # Ajusta a tus nombres reales en ventas_frutto:
        "fecha","product","vendor","location","precio","cantidad","unidad"
    ]

    quotes_cols = st.text_input(
        "Columnas a leer desde quotations (coma-separadas)",
        value=",".join(quote_default_cols)
    )
    sales_cols = st.text_input(
        "Columnas a leer desde ventas_frutto (coma-separadas)",
        value=",".join(sales_default_cols)
    )

    quotes_cols_list = [c.strip() for c in quotes_cols.split(",") if c.strip()]
    sales_cols_list  = [c.strip() for c in sales_cols.split(",") if c.strip()]

    df_q, diag_q = _connect_and_preview("supabase_quotes", quotes_cols_list)
    df_s, diag_s = _connect_and_preview("supabase_sales", sales_cols_list)

    st.caption("🔌 " + (diag_q or "Quotes: sin diagnóstico"))
    st.caption("🔌 " + (diag_s or "Sales: sin diagnóstico"))

# Mostrar previas
c1, c2 = st.columns(2)
with c1:
    st.subheader("Quotes (bloque: supabase_quotes)")
    if df_q is None or df_q.empty:
        st.warning("No hay datos de quotes (o no se pudo leer). Revisa RLS/policies, columnas o secrets.")
    else:
        st.dataframe(df_q.head(20), use_container_width=True, height=350)

with c2:
    st.subheader("Sales (bloque: supabase_sales)")
    if df_s is None or df_s.empty:
        st.warning("No hay datos de sales (o no se pudo leer). Revisa RLS/policies, columnas o secrets.")
    else:
        st.dataframe(df_s.head(20), use_container_width=True, height=350)

st.divider()

# ----------------------------
# Match interactivo
# ----------------------------
st.subheader("🔎 Configurar Match")
if df_q is None or df_q.empty or df_s is None or df_s.empty:
    st.info("Carga datos de ambas fuentes para habilitar el match.")
    st.stop()

# Proponer claves por defecto
q_cols = df_q.columns.tolist()
s_cols = df_s.columns.tolist()

# Heurística de nombres comunes
def _suggest(colnames, candidates):
    lname = [c.lower() for c in colnames]
    for cand in candidates:
        if cand.lower() in lname:
            idx = lname.index(cand.lower())
            return colnames[idx]
    return None

q_key1_suggest = _suggest(q_cols, ["product"])
q_key2_suggest = _suggest(q_cols, ["vendorclean","vendor","shipper"])

s_key1_suggest = _suggest(s_cols, ["product","producto"])
s_key2_suggest = _suggest(s_cols, ["vendor","proveedor","shipper"])

q_key1 = st.selectbox("Columna de PRODUCTO en Quotes", q_cols, index=q_cols.index(q_key1_suggest) if q_key1_suggest in q_cols else 0)
q_key2 = st.selectbox("Columna de VENDOR en Quotes",   q_cols, index=q_cols.index(q_key2_suggest) if q_key2_suggest in q_cols else 0)

s_key1 = st.selectbox("Columna de PRODUCTO en Sales",  s_cols, index=s_cols.index(s_key1_suggest) if s_key1_suggest in s_cols else 0)
s_key2 = st.selectbox("Columna de VENDOR en Sales",    s_cols, index=s_cols.index(s_key2_suggest) if s_key2_suggest in s_cols else 0)

st.caption("Tip: si en ventas no tienes vendor, puedes elegir otra columna (por ejemplo, `supplier`, `cliente`, etc.) o dejar el match solo por producto.")

# Normalización simple (lower + strip)
def _norm_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

# Armar llaves
q = df_q.copy()
s = df_s.copy()
q["_match_prod"] = _norm_series(q[q_key1])
s["_match_prod"] = _norm_series(s[s_key1])

# Vendor opcional
use_vendor = True
if q_key2 and s_key2:
    q["_match_vendor"] = _norm_series(q[q_key2])
    s["_match_vendor"] = _norm_series(s[s_key2])
else:
    use_vendor = False

# Join keys
on_keys = ["_match_prod", "_match_vendor"] if use_vendor else ["_match_prod"]

# Selección de columnas a mostrar
with st.expander("🧰 Seleccionar columnas a mostrar en el resultado", expanded=False):
    left_cols  = st.multiselect("Columnas de Quotes", q_cols, default=[c for c in q_cols if c in ("cotization_date","product","vendorclean","location","price","volume_standard","volume_unit")])
    right_cols = st.multiselect("Columnas de Sales",  s_cols, default=[c for c in s_cols if c in ("fecha","product","vendor","location","precio","cantidad","unidad")])

# Match
st.subheader("🔗 Resultado del Match")
how = st.selectbox("Tipo de unión", ["inner","left","right","outer"], index=0)

try:
    merged = q.merge(s, on=on_keys, how=how, suffixes=("_q","_s"))
    # Ordenar por fecha si existen columnas posibles
    date_cols = [c for c in ["cotization_date","fecha"] if c in merged.columns]
    if date_cols:
        for c in date_cols:
            merged[c] = pd.to_datetime(merged[c], errors="coerce")
        merged = merged.sort_values(by=date_cols, ascending=False)

    # columnas visibles
    out_cols = []
    for c in left_cols:
        if c in merged.columns:
            out_cols.append(c)
    for c in right_cols:
        if c in merged.columns:
            out_cols.append(c)

    # Si no seleccionaste nada, mostramos algo razonable
    if not out_cols:
        out_cols = [c for c in merged.columns if not c.startswith("_match")]

    st.dataframe(merged[out_cols].head(200), use_container_width=True, height=420)
    st.caption(f"Filas unidas: {len(merged)}  •  on={on_keys}  •  how='{how}'")
except Exception as e:
    st.error(f"No pude hacer el merge: {e}")

st.divider()

# Diagnóstico de “no matcheados” (solo hace sentido para left/right)
st.subheader("🧭 Diagnóstico de no coincidencias")
if how in ("left","outer"):
    only_left = q.merge(s[on_keys], on=on_keys, how="left", indicator=True)
    only_left = only_left[only_left["_merge"] == "left_only"]
    st.write("Solo en Quotes (sin match en Sales):", len(only_left))
    if not only_left.empty:
        st.dataframe(only_left[[q_key1] + ([q_key2] if use_vendor else [])].head(50), use_container_width=True, height=260)

if how in ("right","outer"):
    only_right = s.merge(q[on_keys], on=on_keys, how="left", indicator=True)
    only_right = only_right[only_right["_merge"] == "left_only"]
    st.write("Solo en Sales (sin match en Quotes):", len(only_right))
    if not only_right.empty:
        st.dataframe(only_right[[s_key1] + ([s_key2] if use_vendor else [])].head(50), use_container_width=True, height=260)

st.caption("Nota: si no aparecen datos, revisa policies RLS de Supabase para permitir SELECT con la clave anon en ambas tablas.")
