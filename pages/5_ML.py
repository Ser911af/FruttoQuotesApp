# pages/02_SQL_Agent.py
# Streamlit page: Natural language → SQL (ventas_frutto) con DeepSeek API (versión robusta sin f-strings en prompts)

import os
import json
import re
import textwrap
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import requests

st.set_page_config(page_title="SQL Agent — ventas_frutto (DeepSeek)", layout="wide")

st.title("🧠 Analista SQL — ventas_frutto (DeepSeek)")
st.caption("Consulta tu tabla con lenguaje natural, con reglas de Frutto Foods.")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    st.error("⚠️ Falta DEEPSEEK_API_KEY en secrets o entorno.")

DB_URL = (
    os.getenv("DATABASE_URL")
    or st.secrets.get("DATABASE_URL")
    or st.secrets.get("POSTGRES_URL")
    or ""
)

@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    if not DB_URL:
        raise ValueError("DATABASE_URL no configurada en secrets o variables de entorno.")
    eng = create_engine(DB_URL, pool_pre_ping=True)
    return eng

FRUTTO_RULES = textwrap.dedent(
    r"""
    Rol: Eres un analista de datos senior de Frutto Foods especializado en PostgreSQL.
    Tarea: traducir preguntas de negocio a consultas SQL correctas y eficientes contra la tabla ventas_frutto.
    Usa date := COALESCE(received_date, reqs_date, most_recent_invoice_paid_date, created_at::date) y excluye canceladas.
    Formato de salida JSON estricto con campos: assumptions, explanation, sql, suggestions.
    """
)

SCHEMA_HINT = textwrap.dedent(
    """
    Columnas frecuentes: product, commoditie, sales_order, invoice_num, invoice_payment_status, sale_location, sales_rep,
    customer, vendor, quantity, total_revenue, total_profit_usd, total_profit_pct, received_date, reqs_date, created_at.
    """
)

FEW_SHOT = [
    {
        "q": "Clientes de nuestro equipo y qué commodities consumen (sin canceladas)",
        "a": textwrap.dedent(
            """
            WITH base AS (
              SELECT
                COALESCE(received_date, reqs_date, most_recent_invoice_paid_date, created_at::date) AS date,
                customer, commoditie, sales_rep, sales_order, invoice_payment_status
              FROM ventas_frutto
            )
            SELECT sales_rep, customer, commoditie, COUNT(*) AS lineas_venta, COUNT(DISTINCT sales_order) AS ordenes_unicas
            FROM base
            WHERE COALESCE(sales_rep, '') <> ''
              AND COALESCE(customer, '') <> ''
              AND COALESCE(commoditie, '') <> ''
              AND COALESCE(invoice_payment_status, '') NOT ILIKE '%cancel%'
            GROUP BY sales_rep, customer, commoditie
            ORDER BY sales_rep, customer, lineas_venta DESC
            LIMIT 5000;
            """
        ).strip(),
    }
]

ROW_LIMIT_DEFAULT = 5000
SQL_ONLY_SELECT = re.compile(r"^\s*SELECT\b", re.IGNORECASE | re.DOTALL)

def enforce_select_and_limit(sql: str, default_limit: int = ROW_LIMIT_DEFAULT) -> str:
    sql_clean = sql.strip().rstrip(";")
    if not SQL_ONLY_SELECT.search(sql_clean):
        raise ValueError("Sólo se permiten consultas SELECT.")
    if re.search(r"\bLIMIT\b", sql_clean, re.IGNORECASE) is None:
        sql_clean = f"{sql_clean}\nLIMIT {default_limit}"
    return sql_clean

def run_query(engine: Engine, sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn)

def call_deepseek(question: str) -> dict:
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    user_prompt = "Consulta de negocio: {}\n\nPistas de esquema:\n{}\n\nEjemplo:\n{}".format(
        question.strip(), SCHEMA_HINT, FEW_SHOT[0]["a"]
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": FRUTTO_RULES},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1200,
    }
    url = "https://api.deepseek.com/v1/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    txt = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

    s = txt.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\\s*", "", s)
        s = re.sub(r"\\s*```$", "", s)
    first_brace, last_brace = s.find('{'), s.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        s = s[first_brace:last_brace+1]

    try:
        parsed = json.loads(s)
    except Exception:
        parsed = json.loads(s.encode('utf-8', 'ignore').decode('utf-8', 'ignore'))

    for k in ("assumptions", "explanation", "sql", "suggestions"):
        parsed.setdefault(k, "")
    return parsed

# === UI ===
with st.sidebar:
    st.subheader("Ajustes")
    enforced_limit = st.number_input("LIMIT por defecto", 100, 20000, ROW_LIMIT_DEFAULT, step=100)
    show_sql = st.checkbox("Mostrar SQL", value=True)
    run_auto = st.checkbox("Ejecutar automáticamente", value=True)

st.markdown("**Sugerencias rápidas**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Clientes por sales_rep y commodity"):
        st.session_state["_demo_q"] = "Lista los clientes de cada sales_rep y los commodities que consumen (excluye canceladas)."
with col2:
    if st.button("Top 20 commodities por revenue (90 días)"):
        st.session_state["_demo_q"] = "Top 20 commodities por revenue últimos 90 días, sin canceladas."
with col3:
    if st.button("Margen por cliente (YTD)"):
        st.session_state["_demo_q"] = "Margen % por cliente y commodity año actual, sin canceladas."

q_default = st.session_state.get("_demo_q", "Clientes de nuestro equipo y qué commodities consumen (sin canceladas)")
question = st.text_area("Pregunta de negocio", value=q_default, height=100)
colA, colB = st.columns(2)
with colA:
    go = st.button("🔍 Generar SQL")
with colB:
    clear = st.button("🧹 Limpiar")
if clear:
    st.session_state.pop("_out", None)
    st.session_state.pop("_res", None)

if go and question.strip():
    try:
        out = call_deepseek(question)
        st.session_state["_out"] = out
    except Exception as e:
        st.error(f"Error generando SQL: {e}")

out = st.session_state.get("_out")
if out:
    if out.get("assumptions"):
        st.info(out["assumptions"])
    st.write(out.get("explanation", ""))

    sql_raw = out.get("sql", "").strip()
    try:
        sql_safe = enforce_select_and_limit(sql_raw, default_limit=int(enforced_limit))
    except Exception as e:
        st.error(f"SQL inválido: {e}")
        sql_safe = ""

    if show_sql and sql_safe:
        st.code(sql_safe, language="sql")

    if sql_safe and (run_auto or st.button("▶️ Ejecutar consulta")):
        try:
            df = run_query(get_engine(), sql_safe)
            st.session_state["_res"] = df
        except Exception as e:
            st.error(f"Error al ejecutar la consulta: {e}")

    if out.get("suggestions"):
        with st.expander("💡 Recomendaciones"):
            st.markdown(out["suggestions"])

res = st.session_state.get("_res")
if isinstance(res, pd.DataFrame):
    st.subheader("Resultados")
    if res.empty:
        st.warning("La consulta no devolvió filas.")
    else:
        st.dataframe(res, use_container_width=True)
        try:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Filas", f"{len(res):,}")
            with c2:
                if "total_revenue" in res.columns:
                    st.metric("Revenue (sum)", f"${res['total_revenue'].sum():,.0f}")
            with c3:
                if "total_profit_usd" in res.columns:
                    st.metric("Profit (sum)", f"${res['total_profit_usd'].sum():,.0f}")
            with c4:
                if set(["total_profit_usd", "total_revenue"]).issubset(res.columns):
                    num = res["total_profit_usd"].sum()
                    den = res["total_revenue"].sum()
                    pct = (num / den * 100.0) if den else None
                    st.metric("Margen %", f"{pct:.1f}%" if pct is not None else "—")
        except Exception:
            pass
        csv = res.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", data=csv, file_name="sql_agent_result.csv", mime="text/csv")

st.caption("Solo se ejecutan SELECT y se aplica LIMIT por defecto. DeepSeek produce JSON robusto.")
