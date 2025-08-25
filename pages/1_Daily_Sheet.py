st.set_page_config(page_title="FruttoFoods Daily Sheet", layout="wide")

import streamlit as st
import os

st.set_page_config(page_title="Daily Sheet", layout="wide")

LOGO_PATH = "data/Asset 7@4x.png"

st.title("Daily Sheet")

# Logo centrado
colA, colB, colC = st.columns([1, 2, 1])
with colB:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=True)
    else:
        st.info("Logo no encontrado. Verifica 'data/Asset 7@4x.png'.")

st.caption("Página en construcción — aquí agregaremos la vista del día.")

