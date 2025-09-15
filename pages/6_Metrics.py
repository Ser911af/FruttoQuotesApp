import time
import pandas as pd
import streamlit as st
from supabase import create_client, Client
from simple_auth import ensure_login, logout_button

# =============================================
# Daily Metrics — Supabase [sales] (simple_auth)
# Shows ONLY the current user's records in "Last records" (date + user name formatted)
# Requires permissive RLS (demo) OR server-side RLS aligned with your setup.
# =============================================

# ✅ Require login via your simple_auth (NOT Supabase Auth)
user = ensure_login()
with st.sidebar:
    logout_button()

st.set_page_config(page_title="Daily Metrics — Supabase Sales", page_icon="📈", layout="centered")
st.title("Daily Metrics — Supabase [sales] 📈")
st.caption(f"Session: {user}")
st.caption("Record commercial activity (Reached / Engaged / Closed) in the *supabase_sales* project.")

# --- Credentials from secrets (.streamlit/secrets.toml) ---
# [supabase_sales]
# url = "https://YOUR-SALES-PROJECT.supabase.co"
# anon_key = "ey..."
if "supabase_sales" not in st.secrets:
    st.error("Missing credentials: add [supabase_sales] (url, anon_key) to .streamlit/secrets.toml.")
    st.stop()

SUPABASE_URL = st.secrets["supabase_sales"].get("url", "")
SUPABASE_ANON_KEY = st.secrets["supabase_sales"].get("anon_key", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Please fill supabase_sales.url and supabase_sales.anon_key in secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Resolve display name from user_profiles (username -> full_name) ---
@st.cache_data(show_spinner=False)
def get_display_name(username: str) -> str:
    try:
        res = supabase.table("user_profiles").select("full_name").eq("username", username).limit(1).execute()
        if res.data:
            return (res.data[0].get("full_name") or username).strip()
        return username
    except Exception:
        return username

display_name = get_display_name(str(user))

TABLE_NAME = "daily_metrics"  # Must exist in supabase_sales

# ---------- Form: create a new record ----------
with st.form("metrics_form_sales", clear_on_submit=False):
    # User name comes from your session/profile and is not editable
    st.text_input("User", value=display_name, disabled=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        clients_reached_out = st.number_input("CLIENTS REACHED OUT", min_value=0, step=1, value=0)
    with c2:
        clients_engaged = st.number_input("CLIENTS ENGAGED", min_value=0, step=1, value=0)
    with c3:
        clients_closed = st.number_input("CLIENTS CLOSED", min_value=0, step=1, value=0)

    submitted = st.form_submit_button("Save to SALES")

if submitted:
    if clients_reached_out < (clients_engaged + clients_closed):
        st.warning("Reached Out cannot be less than Engaged + Closed.")
    else:
        try:
            payload = {
                "user_name": display_name,
                "clients_reached_out": int(clients_reached_out),
                "clients_engaged": int(clients_engaged),
                "clients_closed": int(clients_closed),
            }
            res = supabase.table(TABLE_NAME).insert(payload).execute()
            if getattr(res, "data", None):
                st.success(f"Saved! ID: {res.data[0].get('id', '—')}")
                time.sleep(0.3)
            else:
                st.info("Insert completed, but no data was returned.")
        except Exception as e:
            st.error(f"Insert error: {e}")

st.markdown("---")
# ---------- Last records: ONLY this user's rows, show Date + User ----------
st.subheader("Last records (your activity only)")

try:
    # Server-side filter to only fetch this user's rows
    query = (
        supabase
        .table(TABLE_NAME)
        .select("created_at,user_name")
        .eq("user_name", display_name)
        .order("created_at", desc=True)
        .limit(100)
    )
    res = query.execute()
    rows = res.data or []

    if not rows:
        st.info("No records yet for your user.")
    else:
        df = pd.DataFrame(rows)
        # Format date nicely (local time display via pandas)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
            # Example: Sep 15, 2025 14:05
            df["Date"] = df["created_at"].dt.strftime("%b %d, %Y %H:%M")
        else:
            df["Date"] = ""

        df["User"] = df.get("user_name", display_name)

        # Only show Date and User as requested
        df_show = df[["Date", "User"]]
        st.dataframe(df_show, use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"Query error: {e}")

st.caption("This page uses simple_auth (not Supabase Auth). The 'Last records' section shows only your rows, with formatted date and your display name.")
