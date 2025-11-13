# 🏠_Home.py
import streamlit as st
from typing import Dict

st.set_page_config(page_title="FruttoQuotes • Home", page_icon="🏠", layout="wide")
st.title("🏠 FruttoQuotes")
st.caption("Prototype with ultra-simple login")

# ----------------------------
# Helpers
# ----------------------------
def get_credentials() -> Dict[str, str]:
    # Allow running locally without st.secrets present
    creds = {}
    try:
        creds = dict(st.secrets.get("credentials", {}))
    except Exception:
        pass
    return creds

def do_login():
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
    if submitted:
        credentials = get_credentials()
        if username in credentials and credentials[username] == password:
            st.session_state.user = username
            st.success(f"Welcome {username} 👋")
            st.rerun()
        else:
            st.error("Invalid credentials")

def safe_page_link(path: str, label: str):
    # Render a link to a page; if it does not exist, show a gentle warning instead of crashing
    try:
        st.page_link(path, label=label)
    except Exception as e:
        with st.expander(f"⚠ Could not link: {label}", expanded=False):
            st.write(f"Path: `{path}`")
            st.write("Likely cause: the file does not exist, has a different name, or is not inside the `pages/` folder.")
            st.write("Error:")
            st.exception(e)

# ----------------------------
# Session state
# ----------------------------
if "user" not in st.session_state:
    st.session_state.user = None

# ----------------------------
# Auth
# ----------------------------
if st.session_state.user:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"Hello {st.session_state.user} 👋")
    with col2:
        if st.button("Sign out", use_container_width=True):
            st.session_state.user = None
            st.rerun()
else:
    do_login()

st.divider()
st.subheader("Pages")

# ----------------------------
# Navigation (con nombres de la foto, en ese orden)
# ----------------------------
links = [
    ("pages/0_Revenue.py",                 "💵 Revenue"),
    ("pages/1_Daily_Sheet.py",             "📊 Daily Sheet"),
    ("pages/2_Upload_Quotes.py",           "📤 Upload Quotes"),
    ("pages/3_Prod. Coverage.py",          "📦 Product Coverage"),
    ("pages/4_VendorProduct_Customers.py", "🔗 VendorProduct Customers"),
    ("pages/5_Metrics.py",                 "📈 Metrics"),
    ("pages/6_Vendor_Retention.py",        "🔁 Vendor Retention"),
    ("pages/7_Customer_Retention.py",      "🧲 Customer Retention"),
]

left, right = st.columns(2)
for i, (path, label) in enumerate(links):
    with (left if i % 2 == 0 else right):
        safe_page_link(path, label)
