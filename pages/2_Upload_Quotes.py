import os
import io
import re
import datetime as dt
from typing import Dict, List

import pandas as pd
import numpy as np
import streamlit as st

try:
    from supabase import create_client
except Exception:
    create_client = None  # Supabase optional in local dev

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="Upload Quotes — Paste Mode", page_icon="📋", layout="wide")
st.title("📋 Ingesta de Cotizaciones (pegar desde portapapeles)")
st.caption("Pega tus cotizaciones tal cual salen de Excel/Email/Sheets. Valido, normalizo y subo a la base.")

# ------------------------
# Utilidades globales
# ------------------------
def _get_secret(name: str):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)

# Limpia cache al iniciar; útil si Streamlit conserva funciones viejas
st.cache_data.clear()
with st.sidebar:
    if st.button("🔄 Limpiar caché"):
        st.cache_data.clear()
        st.success("Cache limpia. Presiona Rerun.")

# ------------------------
# Helper functions
# ------------------------
EXCEL_EPOCH = dt.date(1899, 12, 30)  # Excel date origin

def _cot_date_to_mdy_text(val):
    """
    Devuelve siempre 'M/D/YYYY' como texto.
    Acepta:
      - pandas.Timestamp / datetime / date
      - 'mmddyy' (p.ej. '082525')
      - 'mm/dd/yy', 'mm/dd/yyyy', 'm/d/yyyy'
      - seriales de Excel (int/float razonables)
    """
    if val is None or (isinstance(val, float) and pd.isna(val)) or (isinstance(val, str) and val.strip() == ""):
        return None

    # 1) pandas datetime-like directo
    try:
        ts = pd.to_datetime(val, errors="raise")
        if not pd.isna(ts):
            ts = ts.to_pydatetime()
            return f"{ts.month}/{ts.day}/{ts.year}"
    except Exception:
        pass

    # 2) string 'mmddyy' exacto (6 dígitos)
    if isinstance(val, str) and re.fullmatch(r"\d{6}", val):
        mm  = int(val[0:2])
        dd  = int(val[2:4])
        yy  = int(val[4:6])
        yyyy = 2000 + yy  # mapea 00-99 a 2000-2099
        d = dt.date(yyyy, mm, dd)
        return f"{d.month}/{d.day}/{d.year}"

    # 3) serial de Excel
    if isinstance(val, (int, float)) and not pd.isna(val):
        try:
            base = EXCEL_EPOCH + dt.timedelta(days=int(val))
            return f"{base.month}/{base.day}/{base.year}"
        except Exception:
            pass

    # 4) parse genérico para otros strings
    if isinstance(val, str):
        try:
            ts = pd.to_datetime(val, errors="raise", dayfirst=False, infer_datetime_format=True)
            ts = ts.to_pydatetime()
            return f"{ts.month}/{ts.day}/{ts.year}"
        except Exception:
            pass

    return None

@st.cache_data(show_spinner=False)
def _detect_separator(sample: str) -> str:
    if "\t" in sample:
        return "\t"
    comma = sample.count(",")
    semicolon = sample.count(";")
    pipe = sample.count("|")
    if comma > semicolon and comma > pipe:
        return ","
    if semicolon >= comma and semicolon > pipe:
        return ";"
    if pipe > 0:
        return "|"
    return r"\s{2,}"

@st.cache_data(show_spinner=False)
def _read_pasted(text: str) -> pd.DataFrame:
    text = text.strip().replace("\r\n", "\n")
    if not text:
        return pd.DataFrame()
    sep = _detect_separator(text)
    buf = io.StringIO(text)
    if sep == r"\s{2,}":
        rows = [re.split(r"\s{2,}", ln.strip()) for ln in text.splitlines() if ln.strip()]
        width = max(len(r) for r in rows)
        rows = [r + [None] * (width - len(r)) for r in rows]
        df = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        df = pd.read_csv(buf, sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    return df

STANDARD_COLS = [
    "Date", "Supplier", "OG/CV", "Product", "Size", "Volume", "Price", "Where", "Concat", "Date2"
]

@st.cache_data(show_spinner=False)
def _suggest_mappings(cols: List[str]) -> Dict[str, str]:
    mapping = {}
    lowered = {c.lower(): c for c in cols}
    def find(*candidates):
        for cand in candidates:
            for k, orig in lowered.items():
                if cand in k:
                    return orig
        return None
    mapping["Date"] = find("date", "fecha")
    mapping["Supplier"] = find("supplier", "seller", "vendor", "provee")
    mapping["OG/CV"] = find("og/cv", "og", "cv")
    mapping["Product"] = find("product", "item")
    mapping["Size"] = find("size", "pack")
    mapping["Volume"] = find("volume", "qty", "quantity", "volumen")
    mapping["Price"] = find("price", "usd", "$", "precio")
    mapping["Where"] = find("where", "city", "location", "loc", "nogales", "mcallen", "pharr")
    mapping["Concat"] = lowered.get("concat")
    mapping["Date2"] = lowered.get("date2")
    return mapping

@st.cache_data(show_spinner=False)
def _normalize(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame()
    for std_col, src in colmap.items():
        if src and src in df.columns:
            out[std_col] = df[src]
        else:
            out[std_col] = None
    if out["Date"].notna().any():
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=False, infer_datetime_format=True)
    else:
        out["Date"] = pd.NaT
    out["OG/CV"] = out["OG/CV"].astype(str).str.upper().str.extract(r"(OG|CV)", expand=False)
    for col in ["Product", "Size", "Where", "Supplier"]:
        out[col] = out[col].astype(str).str.strip().replace({"None": None, "nan": None})
    out["Volume"] = (
        out["Volume"].astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
        .astype(float)
    )
    out["Price"] = (
        out["Price"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
        .astype(float)
    )
    needs_concat = out.get("Concat").isna().all()
    if needs_concat:
        excel_serial = out["Date"].dt.date.apply(lambda d: (d - EXCEL_EPOCH).days if pd.notna(d) else None)
        out["Concat"] = excel_serial.astype("Int64").astype(str) + out["Product"].fillna("") + out["Size"].fillna("")
    if out.get("Date2").isna().all():
        out["Date2"] = 1
    out = out[STANDARD_COLS]
    return out

def _validate(df: pd.DataFrame) -> List[str]:
    issues = []
    required = ["Date", "Supplier", "OG/CV", "Product", "Price", "Where"]
    for col in required:
        if df[col].isna().all():
            issues.append(f"Columna requerida vacía: {col}")
    if df["Date"].isna().any():
        issues.append("Hay filas con fecha inválida.")
    if (df["OG/CV"].isna()).any():
        issues.append("Valores OG/CV no detectados (usa 'OG' o 'CV').")
    if (df["Price"].isna()).any():
        issues.append("Precios con formato inválido.")
    return issues

# =====================
# Diagnóstico rápido de credenciales + prueba
# =====================
with st.expander("🧪 Diagnóstico de Supabase (opcional)"):
    url_probe = _get_secret("SUPABASE_URL")
    key_probe = _get_secret("SUPABASE_KEY")
    table_probe = os.getenv("SUPABASE_TABLE", st.secrets.get("SUPABASE_TABLE", "quotations")) if hasattr(st, "secrets") else os.getenv("SUPABASE_TABLE", "quotations")

    st.write("URL presente:", bool(url_probe))
    st.write("KEY presente:", bool(key_probe))
    st.write("Tabla destino:", table_probe)

    def _test_insert_row():
        if not url_probe or not key_probe:
            return False, "Faltan SUPABASE_URL o SUPABASE_KEY (en st.secrets o variables de entorno)."
        if create_client is None:
            return False, "Paquete 'supabase' no disponible. Instala 'supabase' (supabase-py)."
        client = create_client(url_probe, key_probe)
        now = pd.Timestamp.utcnow().tz_localize(None).date()
        payload = [{
            "cotization_date": _cot_date_to_mdy_text(now),  # <-- M/D/YYYY texto
            "organic": 0,
            "product": "_probe_streamlit_",
            "price": 0.01,
            "location": "diagnostic",
            "concat": f"diag-{int(pd.Timestamp.utcnow().timestamp())}",
            "volume_num": None,
            "volume_unit": None,
            "volume_standard": None,
            "vendorclean": "_probe_vendor_",
            "source_chat_id": "streamlit",
            "source_message_id": str(int(pd.Timestamp.utcnow().timestamp()))
        }]
        try:
            client.table(table_probe).upsert(
                payload,
                on_conflict=(
                    "cotization_date,organic,product,price,location,concat,"
                    "volume_num,volume_unit,volume_standard,vendorclean,"
                    "source_chat_id,source_message_id"
                )
            ).execute()
            return True, f"Inserción/Upsert OK en '{table_probe}'."
        except Exception as e:
            return False, f"Error al upsert: {e}"

    if st.button("Probar conexión e insertar fila de diagnóstico", key="probe_btn"):
        ok, msg = _test_insert_row()
        (st.success if ok else st.error)(msg)

# ------------------------
# UI — Pegar y parsear
# ------------------------
st.subheader("1) Pega tus cotizaciones aquí")
st.write("Asegúrate de incluir **encabezados** en la primera fila.")

pasted = st.text_area("Pegar datos", value="", height=220)

if pasted:
    raw_df = _read_pasted(pasted)
    if raw_df.empty:
        st.error("No pude leer la tabla pegada. Revisa delimitadores o encabezados.")
        st.stop()
    st.success(f"Detecté {raw_df.shape[0]} filas × {raw_df.shape[1]} columnas.")

    st.subheader("2) Mapea columnas")
    suggested = _suggest_mappings(list(raw_df.columns))
    colmap = {}
    cols = list(raw_df.columns)
    for std_col in STANDARD_COLS:
        colmap[std_col] = suggested.get(std_col)
    norm_df = _normalize(raw_df, colmap)

    st.subheader("3) Previsualización normalizada")
    st.dataframe(norm_df, use_container_width=True)

    # 3b) Acción inmediata: subir justo después de visualizar
    st.markdown("---")
    st.subheader("4) Subir estas filas ahora → tabla `quotations`")
    st.caption("Usa anon key con RLS. Se hace upsert usando TODAS las columnas de negocio como clave de conflicto.")

    # ===== Helpers específicos para 'quotations' =====
    def _parse_volume_fields(vol_raw: pd.Series):
        s = vol_raw.astype(str).str.strip()
        num = s.str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False).astype(float)
        unit = s.str.extract(r"(CS|CT|CTN|P|PLT|LOAD|LB|KG|BOX|CASE|EA)", expand=False, flags=re.IGNORECASE)
        unit = unit.str.upper()
        std = unit.fillna("UNIT")
        num = num.where(~unit.eq("LOAD"), None)
        return num, unit, std

    def _normalize_to_quotations(df_norm: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        # >>> usa el normalizador robusto a M/D/YYYY (texto)
        out["cotization_date"] = df_norm["Date"].apply(_cot_date_to_mdy_text)
        ogcv = df_norm["OG/CV"].astype(str).str.upper().str.extract(r"(OG|CV)", expand=False)
        out["organic"] = (ogcv == "OG").astype(int)
        out["product"] = df_norm["Product"]
        out["price"] = df_norm["Price"]
        out["location"] = df_norm["Where"]
        out["concat"] = df_norm["Concat"]
        vol_num, vol_unit, vol_std = _parse_volume_fields(df_norm["Volume"])
        out["volume_num"] = vol_num
        out["volume_unit"] = vol_unit
        out["volume_standard"] = vol_std
        out["vendorclean"] = df_norm["Supplier"].astype(str).str.strip()
        out["source_chat_id"] = None
        out["source_message_id"] = None
        cols = [
            "cotization_date","organic","product","price","location","concat",
            "volume_num","volume_unit","volume_standard","vendorclean",
            "source_chat_id","source_message_id"
        ]
        return out[cols]

    def upload_to_supabase_for_quotations(df_norm: pd.DataFrame) -> str:
        url = _get_secret("SUPABASE_URL")
        key = _get_secret("SUPABASE_KEY")
        table_name = os.getenv("SUPABASE_TABLE", "quotations")
        if not url or not key:
            return "Faltan SUPABASE_URL o SUPABASE_KEY (en st.secrets o variables de entorno)."
        if create_client is None:
            return "Paquete 'supabase' no disponible. Instala 'supabase' (supabase-py)."
        client = create_client(url, key)
        df_q = _normalize_to_quotations(df_norm)
        df_q = df_q.astype(object).where(pd.notnull(df_q), None)
        records = df_q.to_dict(orient="records")
        try:
            client.table(table_name).upsert(
                records,
                on_conflict=(
                    "cotization_date,organic,product,price,location,concat,"
                    "volume_num,volume_unit,volume_standard,vendorclean,"
                    "source_chat_id,source_message_id"
                )
            ).execute()
            return f"Subida completa: {len(records)} filas a '{table_name}'."
        except Exception as e:
            return f"Error al subir: {e}"

    problems_preview = _validate(norm_df)
    allow_upload_now = True
    if problems_preview:
        with st.expander("Ver advertencias antes de subir"):
            for p in problems_preview:
                st.warning(p)
        allow_upload_now = st.checkbox("Entiendo las advertencias y deseo subir de todos modos", value=False, key="allow_upload_now_cb")

    if st.button("⬆️ Subir estas filas ahora", type="primary", disabled=not allow_upload_now, key="upload_now_main"):
        with st.spinner("Subiendo a Supabase..."):
            msg = upload_to_supabase_for_quotations(norm_df)
        (st.success if msg.startswith("Subida completa") else st.error)(msg)
    st.markdown("---")
    # Fin de la lógica de subida unificada
