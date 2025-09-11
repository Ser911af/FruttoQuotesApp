# ========= DROP-IN: Carga de ventas con diagnóstico y detectores =========
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import streamlit as st

# ✅ 1) Login obligatorio antes de cargar nada pesado
from simple_auth import ensure_login, logout_button

user = ensure_login()   # Si no hay sesión, este call debe bloquear la página (st.stop)
with st.sidebar:
    logout_button()

# (Opcional) mostrar quién es el usuario activo en la UI
st.caption(f"Sesión: {user}")

# Si no existe create_client, manten compatibilidad
try:
    from supabase import create_client
except Exception:
    create_client = None

# ---- Utilidad de errores centralizada
def _errlog(msg: str, exc: Exception | None = None):
    key = "_sales_errors"
    if key not in st.session_state:
        st.session_state[key] = []
    m = f"❌ {msg}" + (f" | {type(exc).__name__}: {exc}" if exc else "")
    st.session_state[key].append(m)
    # para ver también en consola
    print(m)

# ---- Cliente Supabase robusto
def _load_supabase_client(secret_key: str):
    sec = st.secrets.get(secret_key)
    if not sec or not create_client:
        _errlog(f"No hay secretos '{secret_key}' o supabase no instalado.")
        return None
    url, key = sec.get("url"), sec.get("anon_key")
    if not url or not key:
        _errlog(f"Faltan url/anon_key en secrets['{secret_key}'].")
        return None
    try:
        client = create_client(url, key)
        client._schema = sec.get("schema", "public")
        return client
    except Exception as e:
        _errlog("Fallo creando cliente Supabase.", e)
        return None

# ---- Rangos de tiempo
def utc_bounds_days_back(days_back: int) -> tuple[str, str]:
    tz_bog = ZoneInfo("America/Bogota")
    end = datetime.now(tz_bog)                # ahora Bogotá
    start = end - timedelta(days=days_back)   # ventana hacia atrás
    return start.astimezone(timezone.utc).isoformat(), end.astimezone(timezone.utc).isoformat()

def bogota_day_bounds(day: datetime.date) -> tuple[str, str]:
    tz = ZoneInfo("America/Bogota")
    start = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=tz)
    end   = start + timedelta(days=1)
    return start.astimezone(timezone.utc).isoformat(), end.astimezone(timezone.utc).isoformat()

# ---- Normalizadores base (reutiliza tus helpers si ya existen)
import re, math
def _normalize_txt(s): 
    s = "" if s is None else str(s).lower()
    s = re.sub(r"[\s/_\-]+", " ", s)
    s = re.sub(r"[^\w\s\.\+]", "", s)
    return s.strip()

SIZE_PATTERN = re.compile(r"(\d+\s?lb|\d+\s?ct|\d+\s?[xX]\s?\d+|bulk|jbo|xl|lg|med|fancy|4x4|4x5|5x5|60cs|15 lb|25 lb|2 layers?|layers?)", re.IGNORECASE)
def _extract_size(s: str) -> str:
    if not isinstance(s, str): return ""
    m = SIZE_PATTERN.search(s)
    return m.group(1).lower() if m else ""

def _vendor_clean(s: str) -> str: return _normalize_txt(s)
def _loc_clean(s: str) -> str: return _normalize_txt(s)

def _coerce_bool(x):
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if x is None or (isinstance(x, float) and math.isnan(x)): return False
    s = str(x).strip().lower()
    if s in {"cv"}: return False
    return s in {"true","t","1","yes","y","og","organic"}

# Si ya tienes PRODUCT_SYNONYMS y _canonical_product, usa los tuyos.
PRODUCT_SYNONYMS = {
    "round tomato": {"round tomato", "tomato round", "tomato rounds", "tomato", "vr round tomato", "1 layer vr round tomato", "1layer vr round tomato", "vr tomato round"},
    "beef tomato": {"beef", "beef tomato", "beefsteak", "beef steak tomato"},
    "kabocha": {"kabocha", "kabocha squash"},
    "spaghetti": {"spaghetti", "spaghetti squash"},
    "acorn": {"acorn", "acorn squash"},
    "butter": {"butter", "butternut", "butternut squash"},
    "eggplant": {"eggplant", "berenjena"},
}
def _build_reverse_synonyms(syno: dict[str, set]) -> dict[str, str]:
    rev = {}
    for canon, variants in syno.items():
        for v in variants:
            rev[_normalize_txt(v)] = canon
    return rev
REV_SYNONYMS = _build_reverse_synonyms(PRODUCT_SYNONYMS)
def _canonical_product(txt: str) -> str:
    n = _normalize_txt(txt)
    if n in REV_SYNONYMS:
        return REV_SYNONYMS[n]
    for canon, variants in PRODUCT_SYNONYMS.items():
        if any(_normalize_txt(v) in n for v in variants):
            return canon
    return n.split(" ")[0] if n else ""

# ---- Smoke test de tabla (para detectar RLS/tabla vacía/permisos)
def supabase_smoke_test(sb, table: str) -> dict:
    if not sb: return {"ok": False, "error": "Sin cliente Supabase"}
    try:
        r = sb.table(table).select("*", count="exact").limit(1).execute()
        return {"ok": True, "count_hint": r.count, "sample": (r.data or [None])[0]}
    except Exception as e:
        _errlog("Smoke test falló.", e)
        return {"ok": False, "error": str(e)}

# ---- Fetch con fallback según tipo de columna fecha
def _fetch_range(sb, table: str, date_col: str, start_iso: str, end_iso: str, date_col_type: str, columns: str="*") -> pd.DataFrame:
    rows, limit, offset = [], 5000, 0
    try:
        # Primer intento: ISO completo (UTC)
        while True:
            q = (sb.table(table).select(columns).gte(date_col, start_iso).lt(date_col, end_iso)
                 .range(offset, offset + limit - 1).execute())
            data = q.data or []
            rows.extend(data)
            if len(data) < limit: break
            offset += limit

        # Fallback si la col es DATE y no se trajo nada
        if not rows and date_col_type == "date":
            rows, limit, offset = [], 5000, 0
            start_day, end_day = start_iso[:10], end_iso[:10]   # 'YYYY-MM-DD'
            while True:
                q = (sb.table(table).select(columns).gte(date_col, start_day).lt(date_col, end_day)
                     .range(offset, offset + limit - 1).execute())
                data = q.data or []
                rows.extend(data)
                if len(data) < limit: break
                offset += limit
        return pd.DataFrame(rows)
    except Exception as e:
        _errlog("Error en _fetch_range.", e)
        return pd.DataFrame()

# ---- Normalización de ventas
def _normalize_sales(df: pd.DataFrame, date_col_type: str) -> pd.DataFrame:
    if df.empty:
        return df
    alias_map = {
        "received_date": ["received_date", "reqs_date", "created_at", "sale_date"],
        "product": ["product", "commoditie", "buyer_product"],
        "organic": ["organic", "is_organic", "OG/CV"],
        "unit": ["unit"],
        "customer": ["customer", "client", "buyer"],
        "vendor": ["vendor", "shipper", "supplier"],
        "sale_location": ["sale_location", "lot_location"],
        "quantity": ["quantity", "qty"],
        "price_per_unit": ["price_per_unit", "price", "unit_price", "sell_price"],
    }
    std = {k: pd.Series([None]*len(df)) for k in alias_map}
    for std_col, candidates in alias_map.items():
        for c in candidates:
            if c in df.columns:
                std[std_col] = df[c]
                break
    sdf = pd.DataFrame(std)

    # Parse fecha según tipo
    if date_col_type == "date":
        dt_col = pd.to_datetime(sdf["received_date"], errors="coerce")
        sdf["date"] = dt_col  # naive, día local
    else:
        dt_col = pd.to_datetime(sdf["received_date"], errors="coerce", utc=True)
        sdf["date"] = dt_col.dt.tz_convert("America/Bogota").dt.tz_localize(None)

    sdf["is_organic"]     = sdf["organic"].apply(_coerce_bool)
    sdf["product_raw"]    = sdf["product"].astype(str)
    sdf["product_canon"]  = sdf["product_raw"].apply(_canonical_product)
    sdf["size_std"]       = sdf["unit"].astype(str).apply(_extract_size)
    sdf["customer_c"]     = sdf["customer"].astype(str).apply(_normalize_txt)
    sdf["vendor_c"]       = sdf["vendor"].astype(str).apply(_vendor_clean)
    sdf["loc_c"]          = sdf["sale_location"].astype(str).apply(_loc_clean)
    sdf["quantity"]       = pd.to_numeric(sdf["quantity"], errors="coerce")
    sdf["price_per_unit"] = pd.to_numeric(sdf["price_per_unit"], errors="coerce")

    cols = ["date","is_organic","product_raw","product_canon","size_std",
            "customer_c","vendor_c","loc_c","quantity","price_per_unit"]
    return sdf[cols]

# ---- FUNCIÓN PRINCIPAL: recent, bench y familiaridad (con diagnósticos)
@st.cache_data(ttl=300, show_spinner=False)
def load_sales_recent_and_bench(days_recent: int, days_bench: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sb = _load_supabase_client("supabase_sales")
    if not sb:
        st.error("No se encontró fuente de ventas (supabase_sales). Revisa secrets y dependencias.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sec = st.secrets.get("supabase_sales", {})
    table = sec.get("table", "sales")
    date_col = sec.get("date_col", "received_date")
    date_col_type = sec.get("date_col_type", "timestamp")  # "timestamp" | "date"

    # Smoke test temprano para detectar RLS/permisos
    smoke = supabase_smoke_test(sb, table)
    if not smoke.get("ok", False):
        st.error("Supabase smoke test falló. Verifica RLS/permisos/tabla.")
        _errlog("Smoke test no OK.", Exception(smoke.get("error")))
    else:
        # count_hint puede ser None si la política no permite COUNT
        st.caption(f"🔌 Conexión OK. count_hint={smoke.get('count_hint')} sample_keys={list((smoke.get('sample') or {}).keys())[:5]}")

    # Rangos
    r_start, r_end = utc_bounds_days_back(days_recent)
    b_start, b_end = utc_bounds_days_back(days_bench)
    f_start, f_end = utc_bounds_days_back(max(days_recent, days_bench, 90))

    # Fetch
    df_recent = _fetch_range(sb, table, date_col, r_start, r_end, date_col_type)
    df_bench  = _fetch_range(sb, table, date_col, b_start, b_end, date_col_type)
    df_fam    = _fetch_range(sb, table, date_col, f_start, f_end, date_col_type, columns="customer,vendor")

    # Normaliza (recent/bench)
    sdf_recent = _normalize_sales(df_recent, date_col_type)
    sdf_bench  = _normalize_sales(df_bench,  date_col_type)

    return sdf_recent, sdf_bench, df_fam

# ---- PANEL DE DIAGNÓSTICO (llámalo donde cargas datos)
def render_sales_diagnostics(days_recent: int, days_bench: int, sdf_recent: pd.DataFrame, sdf_bench: pd.DataFrame, df_fam: pd.DataFrame):
    with st.expander("🔎 Diagnóstico de conexión y datos (Sales)", expanded=False):
        sec = st.secrets.get("supabase_sales", {})
        st.write("**Secrets presentes:**", list(st.secrets.keys()))
        st.json({
            "url": sec.get("url"), 
            "table": sec.get("table","sales"),
            "date_col": sec.get("date_col","received_date"),
            "date_col_type": sec.get("date_col_type","timestamp"),
        })

        r_start, r_end = utc_bounds_days_back(days_recent)
        b_start, b_end = utc_bounds_days_back(days_bench)
        st.write("**Recent (UTC)**:", r_start, "→", r_end)
        st.write("**Bench  (UTC)**:", b_start, "→", b_end)

        st.write(f"Rows recent: {len(sdf_recent)} | Rows bench: {len(sdf_bench)} | Rows fam: {len(df_fam)}")
        if len(sdf_recent) == 0:
            st.warning("⚠️ 'recent' vacío: revisa 'date_col', 'date_col_type' y RLS. Si la col es DATE, usa 'date_col_type = \"date\"' en secrets.")
        if len(sdf_bench) == 0:
            st.info("ℹ️ 'bench' vacío: puede ser normal si no hay ventas en esa ventana.")

        if not sdf_recent.empty:
            st.write("**recent.head():**")
            st.dataframe(sdf_recent.head())
            if "date" in sdf_recent.columns:
                st.write("recent date min/max:", str(sdf_recent["date"].min()), "→", str(sdf_recent["date"].max()))
        if not sdf_bench.empty:
            st.write("**bench.head():**")
            st.dataframe(sdf_bench.head())

        # Muestra errores acumulados
        errs = st.session_state.get("_sales_errors", [])
        if errs:
            st.error("Registro de errores:")
            for e in errs:
                st.write(e)
        else:
            st.success("Sin errores registrados en el loader 🎯")
# =================== FIN DROP-IN ===================
