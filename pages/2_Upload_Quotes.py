import os
import re
import math
import json
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

st.set_page_config(page_title="FruttoFoods Uploader", layout="wide")

LOGO_PATH = "data/Asset 7@4x.png"

# ---------------------------
# UI Helpers
# ---------------------------
def show_logo_center():
    colA, colB, colC = st.columns([1, 2, 1])
    with colB:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.info("Logo no encontrado (data/Asset 7@4x.png)")

# ---------------------------
# Normalizadores genéricos (fallback)
# ---------------------------
REQ_COLS = [
    "cotization_date", "organic", "product", "price", "location",
    "concat", "volume_num", "volume_unit", "volume_standard",
    "vendorclean", "source_chat_id", "source_message_id"
]

HEADER_ALIASES = {
    "cotization_date": ["cotization_date", "date", "fecha", "cotizacion", "cotization"],
    "organic": ["organic", "og", "cv", "og/cv"],
    "product": ["product", "producto"],
    "price": ["price", "precio"],
    "location": ["location", "where", "ubicacion", "ubicación"],
    "concat": ["concat", "id_concat", "key", "clave", "concatenate"],
    "volume_num": ["volume_num", "volume?", "qty", "cantidad", "vol_num", "cantidad_volumen"],
    "volume_unit": ["volume_unit", "unidad", "unit", "vol_unit"],
    "volume_standard": ["volume_standard", "vol_std", "std_volume", "estandar"],
    "vendorclean": ["vendorclean", "vendor", "shipper", "proveedor"],
    "source_chat_id": ["source_chat_id", "chat_id", "telegram_chat_id"],
    "source_message_id": ["source_message_id", "message_id", "telegram_message_id"],
}

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    out_map = {}
    for tgt, aliases in HEADER_ALIASES.items():
        for a in aliases:
            if a in lower:
                out_map[lower[a]] = tgt
                break
    return df.rename(columns=out_map)

def to_num(x):
    if pd.isna(x): return None
    s = str(x).strip()
    s = s.replace("$", "").replace("€", "").replace("£", "")
    s = s.replace(" ", "")
    s = s.replace(",", ".")         # 8,50 -> 8.50
    s = re.sub(r"\.+", ".", s)     # 10..95 -> 10.95
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return None

def to_int01(x):
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    if s in ("1","og","organic","org","true","sí","si","y","yes"): return 1
    if s in ("0","cv","conv","conventional","false","no","n"): return 0
    try:
        v = int(float(s))
        return 1 if v == 1 else 0
    except Exception:
        return 0

def parse_date_any(x):
    if pd.isna(x): return None
    s = str(x).strip()
    if s.isdigit() and 20000 < int(s) < 60000:  # Excel serial
        dt = pd.to_datetime("1899-12-30") + pd.to_timedelta(int(s), unit="D")
        return f"{dt.month}/{dt.day}/{dt.year}"
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt): return None
    return f"{dt.month}/{dt.day}/{dt.year}"

def excel_serial_from_mdy(mdy: str):
    parts = str(mdy).split("/")
    if len(parts) != 3: return None
    try:
        m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return None
    base = pd.to_datetime("1899-12-30")
    dt = pd.Timestamp(year=y, month=m, day=d)
    return int((dt - base).days)

def build_concat(date_mdy, product, provided=None):
    if provided and str(provided).strip():
        return str(provided).replace(" ", "")
    serial = excel_serial_from_mdy(date_mdy)
    if serial is None: return None
    pname = str(product or "").replace(" ", "")
    return f"{serial}{pname}"

# ---------------------------
# Parser ESPECÍFICO del Excel Diario (tu formato ordenado)
# ---------------------------
DAILY_ALIASES = {
    "date": ["date", "fecha"],
    "shipper": ["shipper", "vendor", "proveedor"],
    "ogcv": ["og/cv", "ogcv", "og", "cv"],
    "product": ["product", "producto"],
    "size": ["size", "talla", "grado", "grade"],
    "volume": ["volume?", "volume", "volumen?"],
    "price": ["price", "precio"],
    "where": ["where", "ubicacion", "ubicación", "location"],
    "concat": ["concat", "clave", "key"],
    "date2": ["date2"],  # ignorado
}

def _rename_daily_headers(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    out_map = {}
    for tgt, aliases in DAILY_ALIASES.items():
        for a in aliases:
            if a in lower:
                out_map[lower[a]] = tgt
                break
    return df.rename(columns=out_map)

_VOL_STD = {
    "p": "pallet",
    "cs": "case",
    "ct": "count",
    "lb": "lb",
}

def _parse_volume(field):
    if field is None or (isinstance(field, float) and pd.isna(field)):
        return (None, None, None)
    s = str(field).strip()
    if s.upper() == "LOAD":
        return (None, None, "load")
    s2 = s.lower().replace(" ", "")
    m = re.match(r"^(\d+)([a-z]+)$", s2)            # 2500cs, 6p
    if m:
        num = float(m.group(1))
        unit = m.group(2).upper()
        std = _VOL_STD.get(m.group(2), None)
        return (num, unit, std)
    m2 = re.match(r"^(\d+)\s*([a-z]+)$", s.strip(), flags=re.IGNORECASE)  # "2500 CS"
    if m2:
        num = float(m2.group(1))
        unit = m2.group(2).upper()
        std = _VOL_STD.get(m2.group(2).lower(), None)
        return (num, unit, std)
    return (None, None, None)

def _to_title_case(s: str) -> str:
    if not s: return ""
    return re.sub(r"\b(\w)", lambda m: m.group(1).upper(), str(s).lower())

def normalize_daily_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _rename_daily_headers(df_raw.copy())

    # ¿Tiene pinta del layout diario?
    if not set(["date","product","price"]).issubset(set(df.columns)):
        return pd.DataFrame()

    # Asegurar columnas opcionales
    for c in ["shipper","ogcv","size","volume","where","concat"]:
        if c not in df.columns:
            df[c] = None

    # Fecha → M/D/YYYY
    df["cotization_date"] = df["date"].apply(parse_date_any)

    # Organic 0/1
    def _ogcv(x):
        if pd.isna(x): return 0
        s = str(x).strip().lower()
        if s in ("og","organic","org","1","true","sí","si","y","yes"): return 1
        if s in ("cv","conv","conventional","0","false","no","n"): return 0
        return 0
    df["organic"] = df["ogcv"].apply(_ogcv).astype("Int64")

    # product = Product + Size
    prod = df["product"].astype(str).str.strip()
    size = df["size"].fillna("").astype(str).str.strip()
    df["product_norm"] = (prod + " " + size).str.replace(r"\s+", " ", regex=True).str.strip()

    # price
    df["price_norm"] = df["price"].apply(to_num)

    # location
    df["location_norm"] = df["where"].fillna("").astype(str).str.strip()

    # volume tripleta
    vols = df["volume"].apply(_parse_volume)
    df["volume_num"] = vols.apply(lambda t: t[0])
    df["volume_unit"] = vols.apply(lambda t: t[1])
    df["volume_standard"] = vols.apply(lambda t: t[2])

    # vendor
    df["vendorclean"] = df["shipper"].fillna("").astype(str).apply(_to_title_case)

    # concat
    def _concat_row(r):
        if pd.notna(r.get("concat")) and str(r["concat"]).strip():
            return str(r["concat"]).replace(" ", "")
        return build_concat(r["cotization_date"], r["product_norm"], None)
    df["concat_norm"] = df.apply(_concat_row, axis=1)

    out = pd.DataFrame({
        "cotization_date": df["cotization_date"],
        "organic": df["organic"].fillna(0).astype(int),
        "product": df["product_norm"],
        "price": df["price_norm"],
        "location": df["location_norm"],
        "concat": df["concat_norm"],
        "volume_num": df["volume_num"],
        "volume_unit": df["volume_unit"],
        "volume_standard": df["volume_standard"],
        "vendorclean": df["vendorclean"],
        "source_chat_id": None,
        "source_message_id": None,
    })
    out = out.dropna(subset=["cotization_date","product","price","concat"])
    return out.reset_index(drop=True)

# ---------------------------
# Integrador: decide formato
# ---------------------------
def normalize_to_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    1) Si el archivo coincide con el layout DIARIO → usa normalize_daily_table.
    2) Sino, intenta el normalizador genérico por alias.
    """
    try:
        daily = normalize_daily_table(df_raw)
        if not daily.empty:
            return daily[REQ_COLS]
    except Exception:
        pass

    base = normalize_headers(df_raw.copy())
    for c in REQ_COLS:
        if c not in base.columns:
            base[c] = None

    base["cotization_date"] = base["cotization_date"].apply(parse_date_any)
    base["organic"] = base["organic"].apply(to_int01)
    base["product"] = base["product"].astype(str).str.strip()
    base["price"] = base["price"].apply(to_num)
    base["location"] = base["location"].astype(str).str.strip()
    base["volume_num"] = base["volume_num"].apply(to_num)
    base["volume_unit"] = base["volume_unit"].astype(str).str.strip()
    vs = base["volume_standard"].apply(to_num)
    base["volume_standard"] = vs.where(vs.notna(), base["volume_num"]).fillna(1)
    base["vendorclean"] = base["vendorclean"].astype(str).str.strip()

    base["concat"] = base.apply(
        lambda r: r["concat"] if isinstance(r["concat"], str) and r["concat"].strip()
        else build_concat(r["cotization_date"], r["product"], None),
        axis=1,
    )

    out = base[REQ_COLS].copy()
    out = out.dropna(subset=["cotization_date","product","price","concat"])
    return out.reset_index(drop=True)

# ---------------------------
# UI
# ---------------------------
st.title("Uploader — Cotizaciones del día")
show_logo_center()
st.write(
    "Sube tu **Excel/CSV** del día. Soporta el formato ordenado ("
    "`Date | Shipper | OG/CV | Product | Size | Volume? | Price | Where | Concat | Date2`)"
    " o archivos con columnas sueltas (usa el normalizador genérico)."
)

uploaded = st.file_uploader("Archivo (Excel .xlsx o CSV)", type=["xlsx","xls","csv"])

# Borrado por fecha (opcional)
with st.expander("🧹 Borrar registros por fecha (opcional)"):
    del_date = st.text_input("Fecha a borrar (M/D/YYYY)", value="")
    if st.button("Borrar día", disabled=(not del_date.strip()), type="secondary"):
        try:
            from supabase import create_client
            SUPABASE_URL = st.secrets["SUPABASE_URL"]
            SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
            sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            q = sb.table("quotations").delete().eq("cotization_date", del_date.strip()).execute()
            st.success("Borrado ejecutado.")
        except Exception as e:
            st.error(f"No pude borrar: {e}")

if not uploaded:
    st.info("⬆️ Arrastra un archivo aquí o haz click para seleccionar uno.")
    st.caption("Tip: si usas el layout diario, los encabezados pueden estar en mayúsculas/minúsculas indistinto.")
    st.stop()

# Leer archivo
try:
    if uploaded.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_excel(uploaded, engine="openpyxl")
except Exception as e:
    st.error(f"No pude leer el archivo: {e}")
    st.stop()

st.write("**Encabezados detectados:**", list(raw.columns))

# Normalizar
norm = normalize_to_schema(raw)
if norm.empty:
    st.warning("No pude reconocer el formato o no hay filas válidas. Verifica encabezados y datos.")
    st.stop()

st.subheader("Preview normalizado (listo para `quotations`)")
st.dataframe(norm.head(300), use_container_width=True)
st.info(f"Filas listas para subir: **{len(norm)}**")

# Descargas
st.download_button(
    "⬇️ Descargar CSV normalizado",
    data=norm.to_csv(index=False).encode("utf-8"),
    file_name="quotations_normalized.csv",
    mime="text/csv",
    use_container_width=True,
)

st.download_button(
    "⬇️ Descargar JSONL",
    data="\n".join(json.dumps(o, ensure_ascii=False) for o in norm.to_dict(orient="records")).encode("utf-8"),
    file_name="quotations_normalized.jsonl",
    mime="text/plain",
    use_container_width=True,
)

# Upsert a Supabase
st.subheader("Insertar en Supabase (upsert, evita duplicados)")
if st.button("🚀 Upsert ahora", type="primary", use_container_width=True):
    try:
        from supabase import create_client
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)

        on_conflict_cols = (
            "cotization_date,organic,product,price,location,concat,"
            "volume_num,volume_unit,volume_standard,vendorclean,source_chat_id,source_message_id"
        )
        BATCH = 1000
        rows = norm.to_dict(orient="records")
        total = len(rows)
        for i in range(0, total, BATCH):
            chunk = rows[i:i+BATCH]
            sb.table("quotations").upsert(chunk, on_conflict=on_conflict_cols).execute()
        st.success(f"Upsert OK. Filas procesadas: {total}")
    except Exception as e:
        st.error(f"Error insertando en Supabase: {e}")
