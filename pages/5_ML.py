# app.py
# FruttoFoods Daily Sheet — OG/CV + Family filter + hide Date/Family + Product emojis + custom order + Concat (emoji primero)

import streamlit as st
import pandas as pd
import os
import re

# ✅ 1) Mandatory login before loading anything heavy
from simple_auth import ensure_login, logout_button

user = ensure_login()   # If there’s no session, this call should block the page (st.stop)
with st.sidebar:
    logout_button()

# (Optional) show current user in the UI
st.caption(f"Session: {user}")

# ---- Optional Altair (with detection) ----
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

st.set_page_config(page_title="FruttoFoods Daily Sheet", layout="wide")

# ---- Visible version tag to confirm deployment ----
VERSION = "Daily_Sheet v2025-11-07 — Concat emoji-first"
st.caption(VERSION)

LOGO_PATH = "data/Asset 7@4x.png"

# ------------------------
# Helpers (parsing & formatting)
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
    # 1) Prioritize size_text (original grade/size)
    stxt = row.get("size_text")
    if isinstance(stxt, str) and stxt.strip():
        return stxt.strip()
    # 2) Legacy fallback: volume_standard
    vs = row.get("volume_standard")
    if isinstance(vs, str) and vs.strip():
        return vs.strip()
    # 3) Last resort: infer from Product
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
    if any(k in s for k in ["tomato", "roma", "round", "grape", "heirloom", "tov"]):
        return "Tomato"
    if any(k in s for k in ["squash", "zucchini", "gray", "grey", "kabocha", "acorn", "butternut", "delicata", "spaghetti"]):
        return "Soft Squash"
    if "cucumber" in s or "cuke" in s or "pickle" in s:
        return "Cucumbers"
    if any(k in s for k in ["pepper", "bell", "jalape", "habanero", "serrano", "pasilla", "anaheim", "shishito", "palermo"]):
        return "Bell Peppers"
    return "Others"

def _norm_name(x: str) -> str:
    if not isinstance(x, str):
        return ""
    s = x.strip()
    return s[:1].upper() + s[1:].lower() if s else s

# =====================
# Diccionario de emojis por commodity
# =====================
commodities_emojis = {
    "Acorn Squash": "🎃", "Anaheim": "🌶️", "Apple": "🍎", "Asparagus": "🥦", "Avocado": "🥑",
    "Banana": "🍌", "Beefsteak Tomato": "🍅", "Bi-Color Corn": "🌽", "Blueberries": "🫐",
    "Broccoli": "🥦", "Brussels": "🥬", "Butternut": "🎃", "Cantaloupe": "🍈", "Caribe": "🌶️",
    "Cauliflower": "🥦", "Celery": "🥒", "Cherry Tomato": "🍅", "Cilantro": "🌿",
    "Cocktail Cukes": "🥒", "Cocktail Tomato": "🍅", "Coleslaw": "🥗",
    "Cucumber European": "🥒", "Cucumber Persian": "🥒", "Cucumber Slicer": "🥒",
    "Delicata": "🎃", "Eggplant": "🍆", "Freight": "🚚", "Garlic": "🧄", "Ginger": "🫚",
    "Grape Tomato": "🍅", "Grapes Early Sweet": "🍇", "Green Beans": "🫛",
    "Green Bell Pepper": "🫑", "Green Onions": "🧅", "Green Plantain": "🍌", "Grey Squash": "🎃",
    "Habanero": "🌶️🔥", "Heirloom": "🍅", "Heirloom Tomato": "🍅", "Honeydew": "🍈",
    "Jalapeã±O": "🌶️", "Kabocha": "🎃", "Lemon": "🍋", "Lettuce": "🥬", "Logistic": "📦",
    "Mango": "🥭", "Material": "📦", "Medley": "🥗", "Minisweet Pepper": "🫑",
    "Orange Bell Pepper": "🟧🫑", "Other": "📦", "Palermo Pepper": "🌶️",
    "Papaya": "🍈", "Pasilla": "🌶️", "Pepper Jalapeño": "🌶️", "Persian Lime": "🍈",
    "Pickle": "🥒", "Pineapple": "🍍", "Poblano": "🌶️", "Raspberries": "🍓",
    "Red Bell Pepper": "🟥🫑", "Red Cabbage": "🥬", "Red Onion": "🧅",
    "Roma Tomato": "🍅", "Romaine": "🥬", "Round Tomato": "🍅", "Serrano": "🌶️",
    "Shishito": "🌶️", "Spaghetti": "🍝", "Strawberry": "🍓", "Tariff": "💲",
    "Thai Pepper": "🌶️🇹🇭", "Tomatillo": "🍏", "TOV (Tomato on Vine)": "🍅",
    "Watermelon": "🍉", "White Corn": "🌽", "White Onion": "🧅",
    "Yellow Bell Pepper": "🟨🫑", "Yellow Corn": "🌽", "Yellow Onion": "🧅",
    "Yellow Squash": "🎃", "Zucchini": "🥒", "English Cucumber": "🥒"
}

def add_emoji_to_product(p: str) -> str:
    """Para la tabla: antepone el emoji al Product si hay match (substring insensible)."""
    if not isinstance(p, str) or not p.strip():
        return ""
    for key, emoji in commodities_emojis.items():
        if key.lower() in p.lower():
            return f"{emoji} {p}"
    pl = p.lower()
    if "heirloom" in pl or "tomato" in pl: return f"🍅 {p}"
    if "english" in pl and "cucumber" in pl: return f"🥒 {p}"
    if "cucumber" in pl: return f"🥒 {p}"
    if "red bell pepper" in pl: return f"🟥🫑 {p}"
    if "yellow bell pepper" in pl: return f"🟨🫑 {p}"
    if "orange bell pepper" in pl: return f"🟧🫑 {p}"
    if "bell pepper" in pl: return f"🫑 {p}"
    return p

def product_emoji(p: str) -> str:
    """Para la concatenación: devuelve SOLO el emoji (sin modificar el Product)."""
    if not isinstance(p, str) or not p.strip():
        return ""
    for key, emoji in commodities_emojis.items():
        if key.lower() in p.lower():
            return emoji
    pl = p.lower()
    if "heirloom" in pl or "tomato" in pl: return "🍅"
    if "english" in pl and "cucumber" in pl: return "🥒"
    if "cucumber" in pl: return "🥒"
    if "red bell pepper" in pl: return "🟥🫑"
    if "yellow bell pepper" in pl: return "🟨🫑"
    if "orange bell pepper" in pl: return "🟧🫑"
    if "bell pepper" in pl: return "🫑"
    return ""

def _clean_concat_volume(v: str) -> str:
    """Vacía si es 'nan', 'none', 'nan none' o blanco."""
    s = (str(v) if v is not None else "").strip()
    sl = s.lower()
    if not s or sl in ("nan", "none", "nan none"):
        return ""
    return s

def build_concat_row(row) -> str:
    """
    {OG/CV} - {emoji} {Product} {Size} {Volume} {Price$}
    - Vendor y Where ignorados
    - Volume opcional (limpio)
    """
    ogcv = (row.get("OG/CV") or "").strip()
    product = (row.get("Product") or "").strip()
    size = (row.get("Size") or "").strip()
    vol = _clean_concat_volume(row.get("Volume"))
    price = (row.get("Price$") or "").strip()
    emoji = product_emoji(product)

    # Orden requerido: OG/CV → emoji → Product → Size → Volume → Price$
    parts = [ogcv, "-"]
    if emoji:
        parts.append(emoji)
    parts.append(product)
    if size:
        parts.append(size)
    if vol:
        parts.append(vol)
    if price:
        parts.append(price)
    return " ".join([p for p in parts if str(p).strip() != ""])

# ------------------------
# Supabase helpers (by sections)
# ------------------------
def _read_section(section_name: str) -> dict:
    try:
        sec = st.secrets[section_name]
    except Exception:
        raise KeyError(f"Section '{section_name}' not found in st.secrets.")
    for k in ("url", "anon_key"):
        if k not in sec or not str(sec[k]).strip():
            raise KeyError(f"Missing or empty '{k}' in st.secrets['{section_name}'].")
    sec = dict(sec)
    sec["table"] = sec.get("table", "").strip() or "quotations"
    sec["schema"] = sec.get("schema", "").strip() or "public"
    return sec

def _create_client(url: str, key: str):
    try:
        from supabase import create_client
    except Exception as e:
        raise ImportError(f"'supabase' is missing in requirements.txt: {e}")
    return create_client(url, key)

def _sb_table(sb, schema: str, table: str):
    try:
        return sb.schema(schema).table(table)  # supabase-py v2+
    except Exception:
        return sb.table(table)                 # fallback v1

# ------------------------
# Data fetch (Supabase — uses section supabase_quotes)
# ------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_quotations_from_supabase():
    try:
        cfg = _read_section("supabase_quotes")
        sb = _create_client(cfg["url"], cfg["anon_key"])
        tbl = _sb_table(sb, cfg["schema"], cfg["table"])
    except Exception as e:
        st.error(f"Invalid Supabase config/client: {e}")
        return pd.DataFrame()

    frames, page_size = [], 1000
    for i in range(1000):
        start, end = i * page_size, i * page_size + page_size - 1
        try:
            resp = (
                tbl.select(
                    "id,cotization_date,organic,product,price,location,"
                    "volume_num,volume_unit,volume_standard,vendorclean,"
                    "size_text"
                )
                .range(start, end)
                .execute()
            )
        except Exception as e:
            st.error(f"Error querying Supabase: {e}")
            return pd.DataFrame()

        rows = getattr(resp, "data", None) or []
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
    df = df.rename(columns={
        "product":"Product",
        "location":"Location",
        "vendorclean":"VendorClean"
    })
    return df

# ------------------------
# UI
# ------------------------
st.title("Daily Sheet")

# Centered logo + utilities
colA, colB, colC = st.columns([1, 2, 1])
with colB:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.info("Logo not found. Check 'data/Asset 7@4x.png'.")

cc1, cc2 = st.columns(2)
with cc1:
    if st.button("🧹 Clear data cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Reload the page or use 'Force rerun'.")
with cc2:
    if st.button("🔄 Force rerun"):
        st.rerun()

df = fetch_all_quotations_from_supabase()

if df.empty:
    st.info("No data available from Supabase yet.")
    st.caption("Page under construction — day view coming soon.")
    st.stop()

# Normalized date column
df["_date"] = pd.to_datetime(df["cotization_date"], errors="coerce").dt.date
valid_dates = df["_date"].dropna()
if valid_dates.empty:
    st.warning("Could not parse any date in 'cotization_date'.")
    st.stop()

# Date selector in mm/dd/yyyy
default_date = max(valid_dates)
sel_date = st.date_input("Date to display", value=default_date, format="MM/DD/YYYY")

# Day subset
day_df = df[df["_date"] == sel_date].copy()
if day_df.empty:
    st.warning("No quotations for the selected date.")
    st.stop()

# Derived fields
day_df["Vendor"]  = day_df["VendorClean"]
day_df["OG/CV"]   = day_df["Organic"].apply(_ogcv)
day_df["Where"]   = day_df["Location"]
day_df["Size"]    = day_df.apply(_choose_size, axis=1)
day_df["Volume"]  = day_df.apply(_volume_str, axis=1)   # ← Final display name
day_df["Price$"]  = day_df["Price"].apply(_format_price)
day_df["Family"]  = day_df["Product"].apply(_family_from_product)
day_df["Date"]    = pd.to_datetime(day_df["cotization_date"], errors="coerce").dt.strftime("%m/%d/%Y")

# ---------- Day view filters ----------
cols = st.columns(6)

with cols[0]:
    product_options = sorted([x for x in day_df["Product"].dropna().unique().tolist() if str(x).strip() != ""])
    sel_products = st.multiselect("Products (available)", options=product_options, default=product_options)

with cols[1]:
    locs = sorted([x for x in day_df["Where"].dropna().unique().tolist() if str(x).strip() != ""])
    sel_locs = st.multiselect("Locations", options=locs, default=locs)

with cols[2]:
    cat_options = ["Conventional (CV)", "Organic (OG)"]
    sel_cat = st.multiselect(
        "Category (OG/CV)",
        options=cat_options,
        default=cat_options,
        help="Filtra por productos Convencionales u Orgánicos."
    )

with cols[3]:
    fam_options = sorted([x for x in day_df["Family"].dropna().unique().tolist() if str(x).strip() != ""])
    sel_fams = st.multiselect("Family (filter only)", options=fam_options, default=fam_options)

with cols[4]:
    search = st.text_input("Search product (contains)", "")

with cols[5]:
    sort_opt = st.selectbox("Sort by", ["Product", "Vendor", "Where", "Price (asc)", "Price (desc)"])

# ---- Apply filters ----
if sel_products:
    day_df = day_df[day_df["Product"].isin(sel_products)]
if sel_locs:
    day_df = day_df[day_df["Where"].isin(sel_locs)]

if sel_cat:
    allowed = set()
    if "Conventional (CV)" in sel_cat: allowed.add("CV")
    if "Organic (OG)" in sel_cat: allowed.add("OG")
    if allowed:
        day_df = day_df[day_df["OG/CV"].isin(allowed)]

if sel_fams:
    day_df = day_df[day_df["Family"].isin(sel_fams)]

if search.strip():
    s = search.strip().lower()
    day_df = day_df[day_df["Product"].str.lower().str.contains(s, na=False)]

# Ordering
if sort_opt == "Price (asc)":
    day_df = day_df.sort_values("Price", ascending=True)
elif sort_opt == "Price (desc)":
    day_df = day_df.sort_values("Price", ascending=False)
else:
    day_df = day_df.sort_values(sort_opt)

# ---------- Read-only pretty table ----------
# 1) Copia para mostrar emojis en Product
display_df = day_df.copy()
display_df["Product"] = display_df["Product"].apply(add_emoji_to_product)

# 2) Columna de concatenación (emoji ANTES del producto)
concat_series = day_df.apply(build_concat_row, axis=1)

# Orden de columnas personalizado + Concat al final
ordered_cols = ["Product", "Price$", "Size", "Where", "Volume", "OG/CV", "Vendor"]
show = display_df[ordered_cols].copy()
show["Concat"] = concat_series.values  # última columna

st.dataframe(show, use_container_width=True)

# Bloque de texto para copiar rápidamente todas las líneas concatenadas
with st.expander("📋 Copy Concat lines"):
    st.code("\n".join(concat_series.tolist()), language="text")

# CSV export con el mismo orden + Concat
csv_bytes = show.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download CSV (day view)",
    data=csv_bytes,
    file_name=f"daily_sheet_{sel_date.strftime('%m-%d-%Y')}.csv",
    mime="text/csv"
)

# =========================
# 📊 Visualizations (BASED on the visible *data*, sin emojis)
# =========================
st.markdown("## 📊 Visualizations (current table)")

if not ALTAIR_OK:
    st.warning("Altair is not installed. Add `altair>=5` to requirements.txt and restart the app.")
else:
    viz_day = day_df.copy()

    if viz_day.empty:
        st.info("There are no rows in the current table to plot.")
    else:
        viz_day["price_num"] = pd.to_numeric(viz_day["Price"], errors="coerce")
        viz_day["volume_num"] = pd.to_numeric(viz_day["volume_num"], errors="coerce")
        viz_day["Where_norm"] = viz_day["Where"].apply(_norm_name)
        viz_day["Vendor_norm"] = viz_day["Vendor"].apply(_norm_name)

        # ---- KPIs ----
        c_k1, c_k2, c_k3, c_k4 = st.columns(4)
        with c_k1:
            mean_price = viz_day["price_num"].mean()
            st.metric("Average price (table)", f"{mean_price:.2f}" if pd.notna(mean_price) else "—")
        with c_k2:
            min_price = viz_day["price_num"].min()
            st.metric("Min price (table)", f"{min_price:.2f}" if pd.notna(min_price) else "—")
        with c_k3:
            max_price = viz_day["price_num"].max()
            st.metric("Max price (table)", f"{max_price:.2f}" if pd.notna(max_price) else "—")
        with c_k4:
            st.metric("Visible offers", f"{len(viz_day)}")

        # ---- Charts ----
        g_loc = (viz_day.groupby("Where_norm", as_index=False)
                        .agg(avg_price=("price_num","mean"),
                             offers=("Where_norm","count")))
        if not g_loc.empty:
            chart_loc = alt.Chart(g_loc).mark_bar().encode(
                x=alt.X("avg_price:Q", title="Average price"),
                y=alt.Y("Where_norm:N", sort="-x", title="Location"),
                tooltip=["Where_norm:N", alt.Tooltip("avg_price:Q", format=".2f"), "offers:Q"]
            ).properties(title="Average price by location (visible table)", height=320)
            st.altair_chart(chart_loc, use_container_width=True)

        g_vendor = (viz_day.groupby("Vendor_norm", as_index=False)
                         .agg(avg_price=("price_num","mean"),
                              offers=("Vendor_norm","count")))
        if not g_vendor.empty:
            chart_vendor = alt.Chart(g_vendor).mark_bar().encode(
                x=alt.X("avg_price:Q", title="Average price"),
                y=alt.Y("Vendor_norm:N", sort="-x", title="Vendor"),
                tooltip=["Vendor_norm:N", alt.Tooltip("avg_price:Q", format=".2f"), "offers:Q"]
            ).properties(title="Average price by vendor (visible table)", height=350)
            st.altair_chart(chart_vendor, use_container_width=True)

        g_vol = (viz_day.groupby("Vendor_norm", as_index=False)
                        .agg(total_volume=("volume_num","sum")))
        g_vol = g_vol[g_vol["total_volume"].fillna(0) > 0]
        if not g_vol.empty:
            chart_vol = alt.Chart(g_vol).mark_bar().encode(
                x=alt.X("total_volume:Q", title="Total volume"),
                y=alt.Y("Vendor_norm:N", sort="-x", title="Vendor"),
                tooltip=["Vendor_norm:N", alt.Tooltip("total_volume:Q", format=",.0f")]
            ).properties(title="Volume by vendor (visible table)", height=350)
            st.altair_chart(chart_vol, use_container_width=True)

        g_prod = (viz_day.groupby("Product", as_index=False)
                          .agg(avg_price=("price_num","mean"),
                               offers=("Product","count")))
        if not g_prod.empty and g_prod["Product"].nunique() > 1:
            chart_prod = alt.Chart(g_prod).mark_bar().encode(
                x=alt.X("avg_price:Q", title="Average price"),
                y=alt.Y("Product:N", sort="-x", title="Product"),
                tooltip=["Product:N", alt.Tooltip("avg_price:Q", format=".2f"), "offers:Q"]
            ).properties(title="Average price by product (visible table)", height=350)
            st.altair_chart(chart_prod, use_container_width=True)

        if viz_day["volume_num"].fillna(0).sum() > 0:
            scatter = alt.Chart(viz_day.dropna(subset=["price_num","volume_num"])).mark_circle(size=80).encode(
                x=alt.X("price_num:Q", title="Price"),
                y=alt.Y("volume_num:Q", title="Volume"),
                tooltip=["Product","Vendor_norm","Where_norm",
                         alt.Tooltip("price_num:Q", format=".2f"),
                         alt.Tooltip("volume_num:Q", format=",.0f")]
            ).properties(title="Price vs Volume (visible table)", height=320)
            st.altair_chart(scatter, use_container_width=True)
