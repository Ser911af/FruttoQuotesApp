import re
import pandas as pd
from datetime import datetime, timezone

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

def _to_title_case(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\b(\w)", lambda m: m.group(1).upper(), str(s).lower())

def _parse_decimal(v):
    if v is None or (isinstance(v, float) and pd.isna(v)): return None
    s = str(v).strip()
    s = s.replace("$", "").replace("€", "").replace("£", "")
    s = s.replace(" ", "")
    s = s.replace(",", ".")       # 8,50 -> 8.50
    s = re.sub(r"\.+", ".", s)   # 10..95 -> 10.95
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return None

def _parse_date_mdy(x):
    """Devuelve string M/D/YYYY; soporta mixto."""
    if pd.isna(x): return None
    s = str(x).strip()
    # Excel serial
    if s.isdigit() and 20000 < int(s) < 60000:
        dt = pd.to_datetime("1899-12-30") + pd.to_timedelta(int(s), unit="D")
        return f"{dt.month}/{dt.day}/{dt.year}"
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt): return None
    return f"{dt.month}/{dt.day}/{dt.year}"

def _excel_serial_from_mdy(mdy: str):
    parts = str(mdy).split("/")
    if len(parts) != 3: return None
    try:
        m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return None
    base = pd.to_datetime("1899-12-30")
    dt = pd.Timestamp(year=y, month=m, day=d)
    return int((dt - base).days)

def _parse_ogcv(x):
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    if s in ("og", "organic", "org", "1", "true", "sí", "si", "y", "yes"): return 1
    if s in ("cv", "conv", "conventional", "0", "false", "no", "n"): return 0
    return 0

_VOL_MAP = {
    "p": "pallet",
    "cs": "case",
    "ct": "count",
    "lb": "lb",
}

def _parse_volume(field: str):
    """
    '6P' -> (6, 'P', 'pallet')
    '2500 CS'/'2500CS' -> (2500, 'CS', 'case')
    '600cs' -> (600, 'CS', 'case')
    'LOAD' -> (None, None, 'load')
    otro -> (None, None, None)
    """
    if field is None or (isinstance(field, float) and pd.isna(field)):
        return (None, None, None)
    s = str(field).strip().lower().replace(" ", "")
    if s == "load":
        return (None, None, "load")
    m = re.match(r"^(\d+)([a-z]+)$", s)  # 2500cs / 6p
    if m:
        num = float(m.group(1))
        unit = m.group(2).upper()
        std = _VOL_MAP.get(m.group(2), None)
        return (num, unit, std)
    m = re.match(r"^(\d+)\s*([a-z]+)$", str(field).strip(), flags=re.IGNORECASE)
    if m:
        num = float(m.group(1))
        unit = m.group(2).upper()
        std = _VOL_MAP.get(m.group(2).lower(), None)
        return (num, unit, std)
    return (None, None, None)

def normalize_daily_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza el Excel diario (Date | Shipper | OG/CV | Product | Size | Volume? | Price | Where | Concat | Date2)
    -> DataFrame listo para `quotations`.
    """
    df = _rename_daily_headers(df_raw.copy())

    needed = ["date", "product", "price"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        # no es el formato diario
        return pd.DataFrame()

    # columnas opcionales que agregamos si faltan
    for c in ["shipper", "ogcv", "size", "volume", "where", "concat"]:
        if c not in df.columns:
            df[c] = None

    # fecha
    df["cotization_date"] = df["date"].apply(_parse_date_mdy)

    # organic
    df["organic"] = df["ogcv"].apply(_parse_ogcv)

    # product + size
    prod = df["product"].astype(str).str.strip()
    size = df["size"].fillna("").astype(str).str.strip()
    df["product_norm"] = (prod + " " + size).str.replace(r"\s+", " ", regex=True).str.strip()

    # price
    df["price_norm"] = df["price"].apply(_parse_decimal)

    # location
    df["location_norm"] = df["where"].fillna("").astype(str).str.strip()

    # volume tripleta
    vol_trip = df["volume"].apply(_parse_volume)
    df["volume_num"]      = vol_trip.apply(lambda t: t[0])
    df["volume_unit"]     = vol_trip.apply(lambda t: t[1])
    df["volume_standard"] = vol_trip.apply(lambda t: t[2])

    # vendor
    df["vendorclean"] = df["shipper"].fillna("").astype(str).apply(_to_title_case)

    # concat
    def _build_concat_row(r):
        if pd.notna(r.get("concat")) and str(r["concat"]).strip():
            return str(r["concat"]).replace(" ", "")
        serial = _excel_serial_from_mdy(r["cotization_date"])
        if serial is None:
            return None
        pname = str(r["product_norm"] or "").replace(" ", "")
        return f"{serial}{pname}"

    df["concat_norm"] = df.apply(_build_concat_row, axis=1)

    # armar salida
    out = pd.DataFrame({
        "cotization_date": df["cotization_date"],
        "organic": df["organic"].astype("Int64").fillna(0),
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

    # mínimos requeridos
    out = out.dropna(subset=["cotization_date", "product", "price", "concat"])
    return out.reset_index(drop=True)

