# -*- coding: utf-8 -*-
# Streamlit page: Upload Quotes — Paste Mode
# Autor: Sergio + ChatGPT (FruttoFoods)
# Descripción: Pega cotizaciones (desde Excel/Email/Sheets), normaliza y sube a Supabase.

import os
import io
import re
import datetime as dt
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import streamlit as st

try:
    from supabase import create_client
except Exception:
    create_client = None  # Supabase opcional en local/dev

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
    """Primero busca en st.secrets, si no, en variables de entorno."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)

# Limpia cache al iniciar; útil si Streamlit conserva funciones viejas
st.cache_data.clear()
with st.sidebar:
    hoy = pd.Timestamp.now(tz="America/Bogota").strftime("%-m/%-d/%Y") if os.name != "nt" else pd.Timestamp.now(tz="America/Bogota").strftime("%#m/%#d/%Y")
    st.markdown(f"**Fecha (America/Bogota):** {hoy}")
    if st.button("🔄 Limpiar caché"):
        st.cache_data.clear()
        st.success("Cache limpia. Presiona Rerun.")

# ------------------------
# Constantes y helpers
# ------------------------
EXCEL_EPOCH = dt.date(1899, 12, 30)  # Excel date origin
STANDARD_COLS = [
    "Date", "Supplier", "OG/CV", "Product", "Size", "Volume", "Price", "Where", "Concat", "Date2"
]

def _cot_date_to_mmddyy_text(val):
    """
    Devuelve siempre 'mmddyy' (cero-rellenado, solo dígitos).
    Acepta:
      - pandas.Timestamp / datetime / date
      - 'mmddyy' (p.ej. '082525')
      - 'mm/dd/yy', 'mm/dd/yyyy', 'm/d/yyyy'
      - seriales de Excel (int/float razonables)
      - strings parseables por pandas
    """
    if val is None or (isinstance(val, float) and pd.isna(val)) or (isinstance(val, str) and val.strip() == ""):
        return None

    # 1) intento directo con pandas
    try:
        d = pd.to_datetime(val, errors="raise").date()
        return d.strftime("%m%d%y")
    except Exception:
        pass

    # 2) 'mmddyy' exacto -> valida y retorna estandarizado
    if isinstance(val, str) and re.fullmatch(r"\d{6}", val):
        mm, dd, yy = int(val[:2]), int(val[2:4]), int(val[4:])
        try:
            _ = dt.date(2000 + yy, mm, dd)
            return f"{mm:02d}{dd:02d}{yy:02d}"
        except Exception:
            return None

    # 3) serial de Excel
    if isinstance(val, (int, float)) and not pd.isna(val):
        try:
            d = EXCEL_EPOCH + dt.timedelta(days=int(val))
            return d.strftime("%m%d%y")
        except Exception:
            pass

    # 4) parse genérico
    if isinstance(val, str):
        try:
            d = pd.to_datetime(val, errors="raise").date()
            return d.strftime("%m%d%y")
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
    # Fallback: columnas separadas por múltiples espacios
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

    mapping["Date"]     = find("date", "fecha")
    mapping["Supplier"] = find("supplier", "seller", "vendor", "provee")
    mapping["OG/CV"]    = find("og/cv", "og", "cv")
    mapping["Product"]  = find("product", "item")
    mapping["Size"]     = find("size", "pack")
    mapping["Volume"]   = find("volume", "qty", "quantity", "volumen")
    mapping["Price"]    = find("price", "usd", "$", "precio")
    mapping["Where"]    = find("where", "city", "location", "loc", "nogales", "mcallen", "pharr")
    mapping["Concat"]   = lowered.get("concat")
    mapping["Date2"]    = lowered.get("date2")
    return mapping

@st.cache_data(show_spinner=False)
def _normalize(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    """Normaliza columnas base. NO parsea Volume; se hace luego con la nueva lógica."""
    out = pd.DataFrame()
    for std_col, src in colmap.items():
        if src and src in df.columns:
            out[std_col] = df[src]
        else:
            out[std_col] = None

    # Date → datetime; se convertirá a mmddyy al generar el payload
    if out["Date"].notna().any():
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=False, infer_datetime_format=True)
    else:
        out["Date"] = pd.NaT

    # OG/CV
    out["OG/CV"] = out["OG/CV"].astype(str).str.upper().str.extract(r"(OG|CV)", expand=False)

    # Limpieza básica de texto
    for col in ["Product", "Size", "Where", "Supplier", "Volume"]:
        out[col] = (
            out[col]
            .astype(str)
            .str.strip()
            .replace({"None": None, "nan": None, "": None})
        )

    # Price → num (sin $ ni comas; corrige decimales malformados)
    out["Price"] = (
        out["Price"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(r"(\d)\.(\.)+(\d)", r"\1.\3", regex=True)  # 10..95 → 10.95
        .str.replace(",", ".", regex=False)                    # 8,5 → 8.5
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
        .astype(float)
    )

    # Concat (si no viene): serial excel + Product + Size (sin nulos)
    needs_concat = out.get("Concat").isna().all()
    if needs_concat:
        excel_serial = out["Date"].dt.date.apply(lambda d: (d - EXCEL_EPOCH).days if pd.notna(d) else None)
        out["Concat"] = excel_serial.astype("Int64").astype(str) + out["Product"].fillna("") + out["Size"].fillna("")

    # Date2 (flag de compatibilidad)
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

# ------------------------
# Volume parsing (nueva lógica)
# ------------------------
_VOL_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(\d+)\s*(?:p|plt|plts|pallets?|tarimas?)\b", "PALLETS"),   # Pallets
    (r"\b(\d+)\s*(?:loads?|ld|tl|trucks?|trk)\b", "LOADS"),         # Loads/TL/Truck
    (r"\b(\d+)\s*(?:cs|ctn|cases?|cajas?|caj)\b", "CS"),            # Cases
]
_NEAR_SIZE = re.compile(
    r"[/x#–-]|(?:\b\d+\s*(?:ct|lb|lbs?|kg|g)\b)|(?:\b\d+\s*[-–]\s*\d+\b)",
    re.IGNORECASE
)

def _parse_volume_from_texts(product: str, size: str, volume: str):
    """
    Retorna (volume_num, volume_unit, volume_standard)
    """
    parts = [p for p in [str(volume) if volume else None, str(product) if product else None, str(size) if size else None] if p and p != "None"]
    if not parts:
        return None, None, None

    s = " " + " ".join(parts).lower().strip() + " "
    s_nosp = " ".join(s.split())

    # RANGOS tipo '6-8 plt' → nulls
    if re.search(r"\b\d+\s*[-–]\s*\d+\s*(?:p|plt|plts|pallets?|tarimas?|cs|ctn|cases?)\b", s_nosp, re.IGNORECASE):
        return None, None, None

    # CANTIDAD + UNIDAD (con antirruido de pack/talla)
    for pat, unit_norm in _VOL_PATTERNS:
        for m in re.finditer(pat, s_nosp, re.IGNORECASE):
            start = max(0, m.start() - 8)
            context = s_nosp[start:m.start()]
            if _NEAR_SIZE.search(context):
                continue
            try:
                n = int(m.group(1))
                if unit_norm == "PALLETS":
                    return n, "PALLETS", "pallet"
                if unit_norm == "LOADS":
                    return n, "LOADS", "load"
                if unit_norm == "CS":
                    return n, "CS", "case"
            except:
                continue

    # CATEGORÍAS SIN NÚMERO (aceptadas)
    if re.search(r"\bvolume|volumen\b", s_nosp):
        return None, "VOLUME", "category"
    if re.search(r"\blimited|limitado\b", s_nosp):
        return None, "LIMITED", "category"
    if re.search(r"\bn/?a\b", s_nosp):
        return None, "NA", "category"

    # (Opcional) Caso especial 'vol-#7s'
    m_special = re.search(r"vol-\s*#\s*(\d+)\s*s\b", s_nosp)
    if m_special:
        return int(m_special.group(1)), "s", None

    return None, None, None

def _parse_volume_fields(df_norm: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    nums, units, stds = [], [], []
    for _, row in df_norm.iterrows():
        n, u, s = _parse_volume_from_texts(row.get("Product"), row.get("Size"), row.get("Volume"))
        nums.append(n); units.append(u); stds.append(s)
    return pd.Series(nums), pd.Series(units), pd.Series(stds)

# ------------------------
# Supabase - normalización final y upload
# ------------------------
def _normalize_to_quotations(df_norm: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    # Fecha -> mmddyy (solo dígitos)
    out["cotization_date"] = df_norm["Date"].apply(_cot_date_to_mmddyy_text)

    # OG/CV -> 0/1
    ogcv = df_norm["OG/CV"].astype(str).str.upper().str.extract(r"(OG|CV)", expand=False)
    out["organic"] = (ogcv == "OG").astype(int)

    out["product"]  = df_norm["Product"]
    out["location"] = df_norm["Where"]
    out["concat"]   = df_norm["Concat"]
    out["price"]    = df_norm["Price"]

    # NUEVO: talla/grade original
    out["size_text"] = df_norm["Size"]

    # Volumen (nueva lógica)
    vol_num, vol_unit, vol_std = _parse_volume_fields(df_norm)
    out["volume_num"]      = vol_num
    out["volume_unit"]     = vol_unit
    out["volume_standard"] = vol_std

    out["vendorclean"]       = df_norm["Supplier"].astype(str).str.strip()
    out["source_chat_id"]    = None
    out["source_message_id"] = None

    cols = [
        "cotization_date","organic","product","price","location","concat",
        "size_text",  # <--- NUEVO
        "volume_num","volume_unit","volume_standard",
        "vendorclean","source_chat_id","source_message_id"
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
    # JSON-safe (None en lugar de NaN)
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

# =====================
# Diagnóstico rápido
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
            return False, "Faltan SUPABASE_URL o SUPABASE_KEY."
        if create_client is None:
            return False, "Paquete 'supabase' no disponible."
        client = create_client(url_probe, key_probe)
        now = pd.Timestamp.utcnow().tz_localize(None).date()
        payload = [{
            "cotization_date": _cot_date_to_mmddyy_text(now),
            "organic": 0,
            "product": "_probe_streamlit_",
            "price": 0.01,
            "location": "diagnostic",
            "concat": f"diag-{int(pd.Timestamp.utcnow().timestamp())}",
            "size_text": "Fancy",  # ejemplo
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

pasted = st.text_area("Pegar datos", value="", height=220, placeholder=(
    "Ejemplo de encabezados:\n"
    "Date\tSupplier\tOG/CV\tProduct\tSize\tVolume\tPrice\tWhere\n"
    "9/1/2025\tWholesum\tOG\teggplant\t\tVOLUME\t16.95\tMcAllen\n"
))

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

    # Normalizar base
    norm_df = _normalize(raw_df, colmap)

    # 3a) Previsualización normalizada (entrada estándar)
    st.subheader("3a) Previsualización normalizada (entrada estándar)")
    _preview_norm = norm_df.copy()
    if "Date" in _preview_norm.columns:
        try:
            _preview_norm["Date"] = _preview_norm["Date"].dt.date
        except Exception:
            pass
    st.dataframe(_preview_norm, use_container_width=True)

    # 3b) Payload a subir (lo que va a Supabase)
    st.subheader("3b) Payload a subir (Vista previa)")
    preview_payload = _normalize_to_quotations(norm_df).copy()
    st.dataframe(preview_payload, use_container_width=True)

    # 3c) Advertencias (si las hay)
    problems_preview = _validate(norm_df)

    st.markdown("---")
    st.subheader("4) Subir estas filas ahora → tabla `quotations`")
    st.caption("Usa anon key con RLS. Se hace upsert usando TODAS las columnas de negocio como clave de conflicto.")

    allow_upload_now = True
    if problems_preview:
        with st.expander("Ver advertencias antes de subir"):
            for p in problems_preview:
                st.warning(p)
        allow_upload_now = st.checkbox(
            "Entiendo las advertencias y deseo subir de todos modos",
            value=False,
            key="allow_upload_now_cb",
        )

    if st.button("⬆️ Subir estas filas ahora", type="primary", disabled=not allow_upload_now, key="upload_now_main"):
        with st.spinner("Subiendo a Supabase..."):
            msg = upload_to_supabase_for_quotations(norm_df)
        (st.success if msg.startswith("Subida completa") else st.error)(msg)

    st.markdown("---")
    st.caption("Fin de la lógica de subida unificada — Upload Quotes — Paste Mode")
