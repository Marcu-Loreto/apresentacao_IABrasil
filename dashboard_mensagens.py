import os
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "/app/data/mensagens.sqlite3")

def carregar_dados():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT id, datahora, texto, sentimento, confianca, emocao, score "
            "FROM mensagens ORDER BY id DESC",
            conn,
        )
    finally:
        conn.close()
    if "datahora" in df.columns:
        df["datahora"] = pd.to_datetime(df["datahora"], errors="coerce")
    return df

def baixar_csv(df):
    return df.to_csv(index=False).encode("utf-8")

st.set_page_config(page_title="Dashboard de Mensagens", layout="wide")
st.title("📊 Dashboard de Análise de Mensagens")

df = carregar_dados()
if df.empty:
    st.warning("Nenhuma mensagem encontrada no banco de dados.")
    st.stop()

st.sidebar.header("Filtros")
sent_opts = sorted([x for x in df["sentimento"].dropna().unique()])
selecionados = st.sidebar.multiselect("Sentimentos", options=sent_opts, default=sent_opts)

min_dt = df["datahora"].min().date() if df["datahora"].notna().any() else datetime.now().date()
max_dt = df["datahora"].max().date() if df["datahora"].notna().any() else datetime.now().date()
d_ini = st.sidebar.date_input("Data inicial", min_dt)
d_fim = st.sidebar.date_input("Data final", max_dt)

mask = (
    df["sentimento"].isin(selecionados)
    & (df["datahora"].dt.date >= d_ini)
    & (df["datahora"].dt.date <= d_fim)
)
filtrado = df.loc[mask].copy()

c1, c2, c3 = st.columns(3)
c1.metric("Total", len(filtrado))
c2.metric("Positivas", (filtrado["sentimento"] == "positivo").sum())
c3.metric("Negativas", (filtrado["sentimento"] == "negativo").sum())

st.subheader("📋 Mensagens")
st.dataframe(filtrado, use_container_width=True)

st.subheader("📈 Distribuição de Sentimentos")
st.bar_chart(filtrado["sentimento"].value_counts())

csv = baixar_csv(filtrado)
st.download_button("📥 Baixar CSV", csv, "mensagens_filtradas.csv", "text/csv")





# import streamlit as st
# import sqlite3
# import pandas as pd
# from datetime import datetime

# # Caminho do banco
# DB_PATH = "db.py"

# def carregar_dados():
#     conn = sqlite3.connect(DB_PATH)
#     df = pd.read_sql_query("SELECT * FROM mensagens ORDER BY id DESC", conn)
#     conn.close()
#     return df

# def baixar_csv(df):
#     return df.to_csv(index=False).encode("utf-8")

# st.set_page_config(page_title="Dashboard de Mensagens", layout="wide")
# st.title("📊 Dashboard de Análise de Mensagens")

# # Carrega dados
# df = carregar_dados()

# if df.empty:
#     st.warning("Nenhuma mensagem encontrada no banco de dados.")
#     st.stop()

# # Filtros
# st.sidebar.header("Filtros")
# sentimentos = st.sidebar.multiselect("Filtrar por Sentimento", options=df["sentimento"].unique(), default=list(df["sentimento"].unique()))
# data_inicio = st.sidebar.date_input("Data inicial", df["datahora"].min())
# data_fim = st.sidebar.date_input("Data final", df["datahora"].max())

# # Aplica filtros
# mask = (
#     df["sentimento"].isin(sentimentos) &
#     (pd.to_datetime(df["datahora"]).dt.date >= data_inicio) &
#     (pd.to_datetime(df["datahora"]).dt.date <= data_fim)
# )
# filtrado = df.loc[mask]

# # Métricas
# col1, col2, col3 = st.columns(3)
# col1.metric("Total de Mensagens", len(filtrado))
# col2.metric("Positivas", (filtrado["sentimento"] == "positivo").sum())
# col3.metric("Negativas", (filtrado["sentimento"] == "negativo").sum())

# # Tabela
# st.subheader("📋 Mensagens Filtradas")
# st.dataframe(filtrado, use_container_width=True)

# # Gráfico
# st.subheader("📈 Distribuição de Sentimentos")
# st.bar_chart(filtrado["sentimento"].value_counts())

# # Exportar CSV
# csv = baixar_csv(filtrado)
# st.download_button("📥 Baixar CSV", csv, "mensagens_filtradas.csv", "text/csv")
