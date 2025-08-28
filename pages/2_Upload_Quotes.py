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
    """Heurística simple para detectar delimitador en texto pegado."""
    # Priorizar tabulaciones (copiado desde Excel)
    if "\t" in sample:
        return "\t"
    # Luego coma o punto y coma
    comma = sample.count(",")
    semicolon = sample.count(";")
    pipe = sample.count("|")
    if comma > semicolon and comma > pipe:
        return ","
    if semicolon >= comma and semicolon > pipe:
        return ";"
    if pipe > 0:
        return "|"
    # Fallback: múltiples espacios como separador
    return r"\s{2,}"

@st.cache_data(show_spinner=False)
def _read_pasted(text: str) -> pd.DataFrame:
    text = text.strip().replace("\r\n", "\n")
    if not text:
        return pd.DataFrame()

    sep = _detect_separator(text)
    buf = io.StringIO(text)

    if sep == r"\s{2,}":
        # Pandas no soporta regex en sep con engine=python para read_csv en todos los casos
        # Intento 1: dividir líneas manual
        rows = [re.split(r"\s{2,}", ln.strip()) for ln in text.splitlines() if ln.strip()]
        # Normalizar a mismo ancho
        width = max(len(r) for r in rows)
        rows = [r + [None] * (width - len(r)) for r in rows]
        df = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        df = pd.read_csv(buf, sep=sep)
    # Limpieza de headers
    df.columns = [str(c).strip() for c in df.columns]
    return df

STANDARD_COLS = [
    "Date", "Supplier", "OG/CV", "Product", "Size", "Volume", "Price", "Where", "Concat", "Date2"
]

@st.cache_data(show_spinner=False)
def _suggest_mappings(cols: List[str]) -> Dict[str, str]:
    """Mapeo heurístico columna origen -> estándar."""
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
    # Concat/Date2 se construyen más adelante, pero permitimos mapear si existen
    mapping["Concat"] = lowered.get("concat")
    mapping["Date2"] = lowered.get("date2")
    return mapping

@st.cache_data(show_spinner=False)
def _normalize(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame()
    # Transferir columnas mapeadas
    for std_col, src in colmap.items():
        if src and src in df.columns:
            out[std_col] = df[src]
        else:
            out[std_col] = None

    # Normalizaciones específicas
    # Fecha
    if out["Date"].notna().any():
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=False, infer_datetime_format=True)
    else:
        out["Date"] = pd.NaT

    # OG/CV
    out["OG/CV"] = out["OG/CV"].astype(str).str.upper().str.extract(r"(OG|CV)", expand=False)

    # Producto / Size
    for col in ["Product", "Size", "Where", "Supplier"]:
        out[col] = out[col].astype(str).str.strip().replace({"None": None, "nan": None})

    # Volume → extraer número
    out["Volume"] = (
        out["Volume"].astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
        .astype(float)
    )

    # Price → quitar símbolos y convertir
    out["Price"] = (
        out["Price"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
        .astype(float)
    )

    # Concat (si no viene): excelserial + product + size
    needs_concat = out.get("Concat").isna().all()
    if needs_concat:
        excel_serial = out["Date"].dt.date.apply(lambda d: (d - EXCEL_EPOCH).days if pd.notna(d) else None)
        out["Concat"] = excel_serial.astype("Int64").astype(str) + out["Product"].fillna("") + out["Size"].fillna("")

    # Date2 (si no viene): 1 fijo como flag (compatibilidad con flujo previo)
    if out.get("Date2").isna().all():
        out["Date2"] = 1

    # Ordenar y tipos
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
st.write("Asegúrate de incluir **encabezados** en la primera fila. Acepto Tabs (Excel), comas, punto y coma o múltiples espacios.")
example = (
    "Date\tSupplier\tOG/CV\tProduct\tSize\tVolume\tPrice\tWhere\n"
    "8/25/2025\tTPE\tOG\tBeef Steak Tomato\t18ct  32ct\tLOAD\t$12.95\tNogales\n"
    "8/25/2025\tCIRULLI\tOG\tTOV\t4 - 5\t6P\t$14.95\tNogales\n"
    "8/25/2025\tGLOBALMEX\tCV\tBeef Steak Tomato\t18ct  32ct\t2500 CS\t$11.23\tNogales\n"
)

pasted = st.text_area("Pegar datos", value="", height=220, placeholder=example)

if pasted:
    raw_df = _read_pasted(pasted)
    if raw_df.empty:
        st.error("No pude leer la tabla pegada. Revisa delimitadores o encabezados.")
        st.stop()
    st.success(f"Detecté {raw_df.shape[0]} filas × {raw_df.shape[1]} columnas.")

    st.subheader("2) Mapea columnas (si es necesario)")
    suggested = _suggest_mappings(list(raw_df.columns))
    colmap = {}
    cols = list(raw_df.columns)
    grid = st.columns(2)

    for i, std_col in enumerate(STANDARD_COLS):
        with grid[i % 2]:
            colmap[std_col] = st.selectbox(
                f"{std_col}",
                options=["(none)"] + cols,
                index=(cols.index(suggested.get(std_col)) + 1) if suggested.get(std_col) in cols else 0,
                help="Selecciona la columna origen que corresponde a este campo estándar."
            )
            if colmap[std_col] == "(none)":
                colmap[std_col] = None

    norm_df = _normalize(raw_df, colmap)

    st.subheader("3) Previsualización normalizada")
    st.dataframe(norm_df, use_container_width=True)

    st.subheader("4) Validación")
    problems = _validate(norm_df)
    if problems:
        for p in problems:
            st.warning(p)
        st.info("Corrige los encabezados/mapeos o edita los datos y vuelve a procesar.")
    else:
        st.success("Validación OK. Listo para subir.")

    # ------------------------
    # Subida a Supabase
    # ------------------------
    st.subheader("5) Subir a Supabase")
    st.caption("Usa variables de entorno SUPABASE_URL, SUPABASE_KEY y define la tabla destino en 'SUPABASE_TABLE' (por defecto: quotes).")

    table_name = os.getenv("SUPABASE_TABLE", "quotes")

    def upload_to_supabase(df: pd.DataFrame) -> str:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            return "Faltan SUPABASE_URL o SUPABASE_KEY en el entorno o st.secrets."
        if create_client is None:
            return "Paquete 'supabase' no disponible. Instala 'supabase' (supabase-py)."
        client = create_client(url, key)
        # convertir NaN → None
        records = df.where(pd.notnull(df), None).to_dict(orient="records")
        try:
            # upsert por 'Concat' como clave natural
            res = client.table(table_name).upsert(records, on_conflict="Concat").execute()
            count = len(records)
            return f"Subida completa: {count} filas a '{table_name}'."
        except Exception as e:
            return f"Error al subir: {e}"

    if st.button("⬆️ Subir a Supabase", type="primary", disabled=bool(problems)):
        with st.spinner("Subiendo a Supabase..."):
            msg = upload_to_supabase(norm_df)
        if msg.startswith("Subida completa"):
            st.success(msg)
        else:
            st.error(msg)

else:
    st.info("Pega tus datos arriba o copia el ejemplo para probar.")

with st.expander("Ver plantilla de encabezados esperados"):
    st.code(
        "Date\tSupplier\tOG/CV\tProduct\tSize\tVolume\tPrice\tWhere",
        language="text",
    )

st.caption("Versión: 2025-08-28 • Esta página acepta pegado directo, infiere delimitadores, normaliza columnas y hace upsert por 'Concat'.")
