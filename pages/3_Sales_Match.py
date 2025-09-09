
import streamlit as st
from simple_auth import ensure_login, current_user, logout_button

st.set_page_config(page_title="Sales Match", page_icon="🔎", layout="wide")
st.title("🔎 Better Match")

user = ensure_login()  # corta la ejecución si no hay login

st.success(f"Bienvenido, {user}")
st.write("Contenido de prueba de Explorer…")

# Botón de logout opcional aquí también
logout_button()
# pages/3_Match.py
import re
import unicodedata
import pandas as pd
import streamlit as st

# (Opcional) usa tu login simple
try:
    from simple_auth import ensure_login, logout_button
    _AUTH = True
except Exception:
    _AUTH = False

st.set_page_config(page_title="Supabase Match Tester", layout="wide")
st.title("🔗 Supabase Match Tester")
st.caption("Prueba dos proyectos Supabase y haz un 'match' por Product + Vendor")

if _AUTH:
    user = ensure_login()
    with st.sidebar: logout_button()

# ---------------------------
# Helpers de credenciales
# ---------------------------
def _load_sb_block(block_name):
    blk = st.secrets.get(block_name)
    if blk and "url" in blk and "anon_key" in blk:
        return blk["url"], blk["anon_key"], blk.get("schema", "public"), blk.get("table", None), "block"
    return None, None, "public", None, None

def _load_supabase_a():
    # Bloque nombrado
    url, key, schema, table, source = _load_sb_block("supabase_a")
    if url and key:
        return url, key, schema, table, f"{source}:supabase_a"

    # Claves planas
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_ANON_KEY")
    if url and key:
        return url, key, "public", None, "flat:A"

    return None, None, "public", None, None

def _load_supabase_b():
    # Bloque nombrado
    url, key, schema, table, source = _load_sb_block("supabase_b")
    if url and key:
        return url, key, schema, table, f"{source}:supabase_b"

    # Claves planas 2
    url = st.secrets.get("SUPABASE2_URL")
    key = st.secrets.get("SUPABASE2_ANON_KEY")
    if url and key:
        return url, key, "public", None, "flat:B"

    return None, None, "public", None, None

def _mask_host(url: str) -> str:
    try:
        m = re.match(r"https?://([^/]+)/?", url)
        host = m.group(1)
        parts = host.split(".")
        if len(parts) > 2:
            parts[0] = parts[0][:3] + "…"
        return ".".join(parts)
    except Exception:
        return "unknown-host"

# ---------------------------
# Conexión y lectura
# ---------------------------
def get_client(url, key):
    try:
        from supabase import create_client
    except Exception as e:
        st.error(f"Falta el paquete 'supabase': {e}. Instala con: pip install supabase")
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        st.error(f"No pude crear cliente Supabase: {e}")
        return None

def read_table(sb, schema, table, columns, limit=None):
    try:
        q = sb.schema(schema).table(table).select(",".join(columns))
        if limit:
            q = q.limit(limit)
        resp = q.execute()
        rows = resp.data or []
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error leyendo {schema}.{table}: {e}")
        return pd.DataFrame()

# ---------------------------
# Normalización y match
# ---------------------------
def normalize_txt(x):
    if pd.isna(x): return ""
    s = str(x).strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')  # quita acentos
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def prepare_keys(df, prod_col, vend_col, out_key="match_key"):
    df = df.copy()
    df["_prod_norm"] = df[prod_col].map(normalize_txt)
    df["_vend_norm"] = df[vend_col].map(normalize_txt)
    df[out_key] = df["_prod_norm"] + " | " + df["_vend_norm"]
    return df

# ---------------------------
# UI de conexión
# ---------------------------
urlA, keyA, schemaA, default_tableA, sourceA = _load_supabase_a()
urlB, keyB, schemaB, default_tableB, sourceB = _load_supabase_b()

colA, colB = st.columns(2)
with colA:
    st.subheader("Proyecto A")
    if not urlA or not keyA:
        st.error("No se encontraron credenciales para A. Usa [supabase_a] o SUPABASE_URL/SUPABASE_ANON_KEY.")
    else:
        st.success(f"Credenciales A detectadas · host={_mask_host(urlA)} · schema={schemaA}")
with colB:
    st.subheader("Proyecto B")
    if not urlB or not keyB:
        st.error("No se encontraron credenciales para B. Usa [supabase_b] o SUPABASE2_URL/SUPABASE2_ANON_KEY.")
    else:
        st.success(f"Credenciales B detectadas · host={_mask_host(urlB)} · schema={schemaB}")

if not (urlA and keyA and urlB and keyB):
    st.stop()

sbA = get_client(urlA, keyA)
sbB = get_client(urlB, keyB)
if not (sbA and sbB):
    st.stop()

st.divider()

# ---------------------------
# Parámetros de tablas/columnas
# ---------------------------
st.subheader("Parámetros de prueba")

with st.form("params"):
    c1, c2 = st.columns(2)
    with c1:
        tableA = st.text_input("Tabla A", value=default_tableA or "quotations")
        colA_prod = st.text_input("Columna producto A", value="product")
        colA_vend = st.text_input("Columna vendor A", value="vendorclean")
        colsA_extra = st.text_input("Columnas extra A (coma)", value="price,location,cotization_date")
        limitA = st.number_input("Límite A (opcional)", min_value=0, value=500, step=100)

    with c2:
        tableB = st.text_input("Tabla B", value=default_tableB or "sales")
        colB_prod = st.text_input("Columna producto B", value="product")
        colB_vend = st.text_input("Columna vendor B", value="vendor")
        colsB_extra = st.text_input("Columnas extra B (coma)", value="price,location,date")
        limitB = st.number_input("Límite B (opcional)", min_value=0, value=500, step=100)

    submitted = st.form_submit_button("Probar y hacer match")

if not submitted:
    st.info("Configura los parámetros y presiona **Probar y hacer match**.")
    st.stop()

# ---------------------------
# Lectura de datos
# ---------------------------
base_cols_A = [colA_prod, colA_vend]
extraA = [c.strip() for c in colsA_extra.split(",") if c.strip()]
colsA = list(dict.fromkeys(base_cols_A + extraA))

base_cols_B = [colB_prod, colB_vend]
extraB = [c.strip() for c in colsB_extra.split(",") if c.strip()]
colsB = list(dict.fromkeys(base_cols_B + extraB))

dfA = read_table(sbA, schemaA, tableA, colsA, limitA or None)
dfB = read_table(sbB, schemaB, tableB, colsB, limitB or None)

c1, c2 = st.columns(2)
with c1:
    st.caption(f"A: {schemaA}.{tableA} — {len(dfA)} filas, columnas: {list(dfA.columns)}")
    st.dataframe(dfA.head(20), use_container_width=True)
with c2:
    st.caption(f"B: {schemaB}.{tableB} — {len(dfB)} filas, columnas: {list(dfB.columns)}")
    st.dataframe(dfB.head(20), use_container_width=True)

if dfA.empty or dfB.empty:
    st.warning("Alguna de las tablas no devolvió filas. Revisa credenciales, RLS/policies y nombres de columnas/tabla.")
    st.stop()

# ---------------------------
# Preparar claves y match
# ---------------------------
if colA_prod not in dfA.columns or colA_vend not in dfA.columns:
    st.error(f"Tabla A no tiene columnas {colA_prod}/{colA_vend}.")
    st.stop()
if colB_prod not in dfB.columns or colB_vend not in dfB.columns:
    st.error(f"Tabla B no tiene columnas {colB_prod}/{colB_vend}.")
    st.stop()

dfa = prepare_keys(dfA, colA_prod, colA_vend, out_key="match_key")
dfb = prepare_keys(dfB, colB_prod, colB_vend, out_key="match_key")

# Match (inner join) por clave normalizada
joined = pd.merge(
    dfa, dfb,
    on="match_key",
    how="inner",
    suffixes=("_A", "_B")
)

# Métricas
st.subheader("Resultados del match")
c1, c2, c3 = st.columns(3)
c1.metric("Filas A", len(dfa))
c2.metric("Filas B", len(dfb))
c3.metric("Matches (inner)", len(joined))

if joined.empty:
    st.info("No hubo matches con las columnas actuales. Prueba normalizaciones distintas o revisa Product/Vendor.")
else:
    # Muestra columnas clave y algunas extra si existen
    show_cols = []
    for col in [colA_prod+"_A", colA_vend+"_A", colB_prod+"_B", colB_vend+"_B"]:
        if col in joined.columns:
            show_cols.append(col)
    # Posibles extras comunes
    for col in ["price_A", "price_B", "location_A", "location_B", "cotization_date_A", "date_B"]:
        if col in joined.columns:
            show_cols.append(col)
    show_cols = list(dict.fromkeys(["match_key"] + show_cols))  # dedup
    st.dataframe(joined[show_cols].head(200), use_container_width=True)

st.success("Prueba completada.")
