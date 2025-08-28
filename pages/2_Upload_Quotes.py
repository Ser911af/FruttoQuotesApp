import os
import io
import re
import datetime as dt
from typing import Dict, List

import pandas as pd
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
# Helper functions
# ------------------------
EXCEL_EPOCH = dt.date(1899, 12, 30)  # Excel date origin

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
    st.subheader("4) Subir estas filas ahora")
    st.caption("Si prefieres, puedes subir de inmediato. Si hay advertencias de validación, marca la confirmación.")

    def _get_secret(name: str):
        try:
            return st.secrets[name]
        except Exception:
            return os.getenv(name)

    table_name = os.getenv("SUPABASE_TABLE", "quotes")

    def upload_to_supabase(df: pd.DataFrame) -> str:
        url = _get_secret("SUPABASE_URL")
        key = _get_secret("SUPABASE_KEY")
        if not url or not key:
            return "Faltan SUPABASE_URL o SUPABASE_KEY (en st.secrets o variables de entorno)."
        if create_client is None:
            return "Paquete 'supabase' no disponible. Instala 'supabase' (supabase-py)."
        client = create_client(url, key)
        records = df.where(pd.notnull(df), None).to_dict(orient="records")
        try:
            client.table(table_name).upsert(records, on_conflict="Concat").execute()
            return f"Subida completa: {len(records)} filas a '{table_name}'."
        except Exception as e:
            return f"Error al subir: {e}"

    problems_preview = _validate(norm_df)
    allow_upload_now = True
    if problems_preview:
        with st.expander("Ver advertencias antes de subir"):
            for p in problems_preview:
                st.warning(p)
        allow_upload_now = st.checkbox("Entiendo las advertencias y deseo subir de todos modos", value=False)

    if st.button("⬆️ Subir estas filas ahora", type="primary", disabled=not allow_upload_now):
        with st.spinner("Subiendo a Supabase..."):
            msg = upload_to_supabase(norm_df)
        (st.success if msg.startswith("Subida completa") else st.error)(msg)

    st.markdown("---")

    problems = _validate(norm_df)
    if problems:
        for p in problems:
            st.warning(p)
        st.stop()

    st.success("Validación OK. Listo para subir.")

    # Subida automática con botón
    st.subheader("4) Subir a Supabase")
    table_name = os.getenv("SUPABASE_TABLE", "quotes")

    def upload_to_supabase(df: pd.DataFrame) -> str:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            return "Faltan SUPABASE_URL o SUPABASE_KEY."
        if create_client is None:
            return "Paquete 'supabase' no disponible."
        client = create_client(url, key)
        records = df.where(pd.notnull(df), None).to_dict(orient="records")
        try:
            res = client.table(table_name).upsert(records, on_conflict="Concat").execute()
            return f"Subida completa: {len(records)} filas a '{table_name}'."
        except Exception as e:
            return f"Error al subir: {e}"

    if st.button("⬆️ Subir a Supabase", type="primary"):
        with st.spinner("Subiendo a Supabase..."):
            msg = upload_to_supabase(norm_df)
        if msg.startswith("Subida completa"):
            st.success(msg)
        else:
            st.error(msg)
else:
    st.info("Pega tus datos arriba para empezar.")
