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

# ✅ Login simple
from simple_auth import ensure_login, logout_button

st.set_page_config(page_title="Upload Quotes — Paste Mode", page_icon="📋", layout="wide")

# Guard de sesión
user = ensure_login()
with st.sidebar:
    logout_button()

st.title("📋 Ingesta de Cotizaciones (pegar desde portapapeles)")
st.caption(f"Sesión: {user} — Pega tus cotizaciones tal cual salen de Excel/Email/Sheets. Valido, normalizo y subo a la base.")

# ------------------------
# Helpers de credenciales Supabase
# ------------------------
def _get_supabase_block(block: str = "supabase_quotes"):
    blk = st.secrets.get(block, {})
    url = blk.get("url")
    key = blk.get("anon_key")
    table = blk.get("table", "quotations")
    schema = blk.get("schema", "public")
    return url, key, schema, table

# ------------------------
# Constantes y helpers
# ------------------------
EXCEL_EPOCH = dt.date(1899, 12, 30)  # Excel date origin
STANDARD_COLS = [
    "Date", "Supplier", "OG/CV", "Product", "Size", "Volume", "Price", "Where", "Concat", "Date2"
]

def _cot_date_to_mdy_text(val):
    if val is None or (isinstance(val, float) and pd.isna(val)) or (isinstance(val, str) and val.strip() == ""):
        return None
    try:
        ts = pd.to_datetime(val, errors="raise")
        if not pd.isna(ts):
            ts = ts.to_pydatetime()
            return f"{ts.month}/{ts.day}/{ts.year}"
    except Exception:
        pass
    if isinstance(val, str) and re.fullmatch(r"\d{6}", val):
        mm, dd, yy = int(val[0:2]), int(val[2:4]), int(val[4:6])
        yyyy = 2000 + yy
        try:
            d = dt.date(yyyy, mm, dd)
            return f"{d.month}/{d.day}/{d.year}"
        except Exception:
            return None
    if isinstance(val, (int, float)) and not pd.isna(val):
        try:
            base = EXCEL_EPOCH + dt.timedelta(days=int(val))
            return f"{base.month}/{base.day}/{base.year}"
        except Exception:
            pass
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
    comma = sample.count(","); semicolon = sample.count(";"); pipe = sample.count("|")
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

@st.cache_data(show_spinner=False)
def _suggest_mappings(cols: List[str]) -> Dict[str, str]:
    mapping = {}
    lowered = {c.lower(): c for c in cols}
    def find(*cands):
        for cand in cands:
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
    out = pd.DataFrame()
    for std_col, src in colmap.items():
        if src and src in df.columns:
            out[std_col] = df[src]
        else:
            out[std_col] = None
    if out["Date"].notna().any():
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    else:
        out["Date"] = pd.NaT
    out["OG/CV"] = out["OG/CV"].astype(str).str.upper().str.extract(r"(OG|CV)", expand=False)
    for col in ["Product", "Size", "Where", "Supplier", "Volume"]:
        out[col] = out[col].astype(str).str.strip().replace({"None": None, "nan": None, "": None})
    out["Price"] = (
        out["Price"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(r"(\d)\.(\.)+(\d)", r"\1.\3", regex=True)
        .str.replace(",", ".", regex=False)
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
        issues.append("Valores OG/CV no detectados.")
    if (df["Price"].isna()).any():
        issues.append("Precios inválidos.")
    return issues

# ------------------------
# Volume parsing
# ------------------------
_VOL_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(\d+)\s*(?:p|plt|plts|pallets?|tarimas?)\b", "PALLETS"),
    (r"\b(\d+)\s*(?:loads?|ld|tl|trucks?|trk)\b", "LOADS"),
    (r"\b(\d+)\s*(?:cs|ctn|cases?|cajas?|caj)\b", "CS"),
]
_NEAR_SIZE = re.compile(r"[/x#–-]|\b\d+\s*(?:ct|lb|lbs?|kg|g)\b|\b\d+\s*[-–]\s*\d+\b", re.IGNORECASE)

def _parse_volume_from_texts(product, size, volume):
    parts = [p for p in [str(volume) if volume else None, str(product) if product else None, str(size) if size else None] if p and p != "None"]
    if not parts:
        return None, None, None
    s_nosp = " " + " ".join(parts).lower().strip() + " "
    if re.search(r"\b\d+\s*[-–]\s*\d+\s*(?:p|plt|plts|pallets?|cs|ctn|cases?)\b", s_nosp):
        return None, None, None
    for pat, unit_norm in _VOL_PATTERNS:
        for m in re.finditer(pat, s_nosp, re.IGNORECASE):
            start = max(0, m.start() - 8)
            context = s_nosp[start:m.start()]
            if _NEAR_SIZE.search(context):
                continue
            try:
                n = int(m.group(1))
                if unit_norm == "PALLETS": return n, "PALLETS", "pallet"
                if unit_norm == "LOADS": return n, "LOADS", "load"
                if unit_norm == "CS": return n, "CS", "case"
            except:
                continue
    if re.search(r"\bvolume|volumen\b", s_nosp): return None, "VOLUME", "category"
    if re.search(r"\blimited|limitado\b", s_nosp): return None, "LIMITED", "category"
    if re.search(r"\bn/?a\b", s_nosp): return None, "NA", "category"
    return None, None, None

def _parse_volume_fields(df_norm: pd.DataFrame):
    nums, units, stds = [], [], []
    for _, row in df_norm.iterrows():
        n, u, s = _parse_volume_from_texts(row.get("Product"), row.get("Size"), row.get("Volume"))
        nums.append(n); units.append(u); stds.append(s)
    return pd.Series(nums), pd.Series(units), pd.Series(stds)

# ------------------------
# Normalización final y subida
# ------------------------
def _normalize_to_quotations(df_norm: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["cotization_date"] = df_norm["Date"].apply(_cot_date_to_mdy_text)
    ogcv = df_norm["OG/CV"].astype(str).str.upper().str.extract(r"(OG|CV)", expand=False)
    out["organic"] = (ogcv == "OG").astype(int)
    out["product"] = df_norm["Product"]
    out["location"] = df_norm["Where"]
    out["concat"] = df_norm["Concat"]
    out["price"] = df_norm["Price"]
    out["size_text"] = df_norm["Size"].astype(str).str.strip().replace({"None": None, "nan": None, "": None})
    vol_num, vol_unit, vol_std = _parse_volume_fields(df_norm)
    out["volume_num"], out["volume_unit"], out["volume_standard"] = vol_num, vol_unit, vol_std
    out["vendorclean"] = df_norm["Supplier"].astype(str).str.strip()
    out["source_chat_id"] = None
    out["source_message_id"] = None
    cols = ["cotization_date","organic","product","price","location","concat",
            "size_text","volume_num","volume_unit","volume_standard","vendorclean",
            "source_chat_id","source_message_id"]
    return out[cols]

def upload_to_supabase_for_quotations(df_norm: pd.DataFrame) -> str:
    url, key, schema, table_name = _get_supabase_block("supabase_quotes")
    if not url or not key:
        return "Faltan credenciales en [supabase_quotes]."
    if create_client is None:
        return "Paquete 'supabase' no disponible."
    client = create_client(url, key)
    df_q = _normalize_to_quotations(df_norm)
    df_q = df_q.astype(object).where(pd.notnull(df_q), None)
    records = df_q.to_dict(orient="records")
    try:
        client.schema(schema).table(table_name).upsert(
            records,
            on_conflict=("cotization_date,organic,product,price,location,concat,"
                         "volume_num,volume_unit,volume_standard,vendorclean,"
                         "source_chat_id,source_message_id")
        ).execute()
        return f"Subida completa: {len(records)} filas a '{schema}.{table_name}'."
    except Exception as e:
        return f"Error al subir: {e}"

# ------------------------
# UI
# ------------------------
st.subheader("1) Pega tus cotizaciones aquí")
pasted = st.text_area("Pegar datos", value="", height=220)

if pasted:
    raw_df = _read_pasted(pasted)
    if raw_df.empty:
        st.error("No pude leer la tabla pegada.")
        st.stop()
    st.success(f"Detecté {raw_df.shape[0]} filas × {raw_df.shape[1]} columnas.")

    suggested = _suggest_mappings(list(raw_df.columns))
    colmap = {std_col: suggested.get(std_col) for std_col in STANDARD_COLS}
    norm_df = _normalize(raw_df, colmap)

    st.subheader("2) Previsualización normalizada")
    st.dataframe(norm_df, use_container_width=True)

    preview_payload = _normalize_to_quotations(norm_df).copy()
    st.subheader("3) Payload a subir (vista previa)")
    st.dataframe(preview_payload, use_container_width=True)

    problems_preview = _validate(norm_df)
    allow_upload = True
    if problems_preview:
        with st.expander("Ver advertencias"):
            for p in problems_preview:
                st.warning(p)
        allow_upload = st.checkbox("Entiendo las advertencias y deseo subir igual", value=False)

    if st.button("⬆️ Subir estas filas ahora", type="primary", disabled=not allow_upload):
        with st.spinner("Subiendo a Supabase..."):
            msg = upload_to_supabase_for_quotations(norm_df)
        (st.success if msg.startswith("Subida completa") else st.error)(msg)

st.caption("Fin de la lógica — Upload Quotes — Paste Mode")
