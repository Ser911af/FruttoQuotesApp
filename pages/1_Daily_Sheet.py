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
VERSION = "Daily_Sheet v2025-09-05 - Size desde size_text (fallback volume_standard)"
st.caption(VERSION)

LOGO_PATH = "data/Asset 7@4x.png"

# ------------------------
# Helpers credenciales
# ------------------------
def _get_supabase_block(block: str = "supabase_quotes"):
    blk = st.secrets.get(block, {})
    url = blk.get("url")
    key = blk.get("anon_key")
    table = blk.get("table", "quotations")
    schema = blk.get("schema", "public")
    return url, key, schema, table

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
    stxt = row.get("size_text")
    if isinstance(stxt, str) and stxt.strip():
        return stxt.strip()
    vs = row.get("volume_standard")
    if isinstance(vs, str) and vs.strip():
        return vs.strip()
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
# Data fetch (Supabase)
# ------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_quotations_from_supabase():
    try:
        from supabase import create_client
    except Exception as e:
        st.error(f"Falta 'supabase' en requirements.txt: {e}")
        return pd.DataFrame()

    url, key, schema, table = _get_supabase_block("supabase_quotes")
    if not url or not key:
        st.error("No encontré credenciales en [supabase_quotes].")
        return pd.DataFrame()

    sb = create_client(url, key)

    frames, page_size = [], 1000
    for i in range(1000):
        start, end = i * page_size, i * page_size + page_size - 1
        try:
            resp = (
                sb.schema(schema)
                  .table(table)
                  .select(
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

        rows = resp.data or []
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < page_size:
            break

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df

    df["cotization_date"] = pd.to_datetime(df["cotization_date"], errors="coerce")
    df["Organic"] = pd.to_numeric(df["organic"], errors="coerce").astype("Int64")
    df["Price"]   = pd.to_numeric(df["price"], errors="coerce")
    df["volume_unit"] = df["volume_unit"].astype(str).fillna("unit")
    if "size_text" not in df.columns:
        df["size_text"] = pd.NA
    df = df.rename(columns={"product":"Product","location":"Location","vendorclean":"VendorClean"})
    return df
