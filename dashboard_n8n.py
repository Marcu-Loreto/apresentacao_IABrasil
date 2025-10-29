# dashboard_n8n.py
import time
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# qat-voxmap-atendimento-dashboard.nl9itp.easypanel.host

# Integrações locais
from shared_state import SharedState  # usa PostgreSQL > Redis > JSON
from analysis import processar_lista_mensagens  # NÃO alteramos a lógica de análise

st.set_page_config(page_title="Dashboard N8N", layout="wide")
st.title("📊 Análises de Atendimento")
st.caption("Mensagens armazenadas no banco (via N8N/API) + análise de sentimento")

# === Sidebar: controles de leitura do banco ===
st.sidebar.subheader("⚙️ Fonte de dados")
session_id = st.sidebar.text_input("Session ID", value="n8n_default", help="Informe a sessão a consultar")
limit = st.sidebar.number_input("Limite de mensagens", min_value=1, max_value=2000, value=200, step=10)
auto_refresh = st.sidebar.toggle("Auto-refresh (a cada 10s)", value=False)
status_cols = st.sidebar.columns(2)
with status_cols[0]:
    st.metric("PostgreSQL", "ON" if getattr(SharedState, "DATABASE_AVAILABLE", False) else "OFF")
with status_cols[1]:
    st.metric("Redis", "ON" if getattr(SharedState, "REDIS_AVAILABLE", False) else "OFF")

reload_now = st.sidebar.button("🔄 Recarregar agora", use_container_width=True)

@st.cache_data(ttl=5)
def carregar_mensagens(session: str, n: int) -> list[str]:
    """
    Busca mensagens da sessão diretamente do estado compartilhado (DB > Redis > JSON)
    e retorna APENAS os textos das mensagens de usuário, em ordem cronológica.
    """
    try:
        registros = SharedState.get_messages(session, limit=n) or []
        # apenas mensagens do usuário (role == 'user')
        textos = [m.get("content", "") for m in registros if (m.get("role") == "user" and m.get("content"))]
        return textos
    except Exception as e:
        st.error(f"Erro ao carregar mensagens: {e}")
        return []

if reload_now:
    st.cache_data.clear()

mensagens = carregar_mensagens(session_id, limit)

# Auto refresh simples
# if auto_refresh:
#     # Registra um noop para invalidar cache de tempos em tempos
#     st.write(f"⏱️ Atualizando em ~10s • {time.strftime('%H:%M:%S')}")
#     st.experimental_set_query_params(ts=int(time.time()))
#     # Pequena espera para evitar loop muito agressivo ao vivo
#     time.sleep(0.2)
if auto_refresh:
    st.write(f"⏱️ Atualizando em ~10s • {time.strftime('%H:%M:%S')}")
    st.query_params.update({"ts": str(int(time.time()))})
    time.sleep(0.2)


if not mensagens:
    st.info("Sem mensagens para analisar. Envie mensagens via N8N/API para a sessão selecionada.")
    st.stop()

with st.spinner("🔍 Processando mensagens..."):
    sentimentos, grafo, wordcloud_img = processar_lista_mensagens(mensagens)

# ===== UI de resultados =====
col1, col2 = st.columns(2)
with col1:
    st.subheader("📥 Mensagens (mais recentes por último)")
    for i, msg in enumerate(mensagens, start=1):
        st.markdown(f"**{i}.** {msg}")

with col2:
    st.subheader("🧠 Análise de Sentimento (por mensagem)")
    df_sent = pd.DataFrame(sentimentos)
    df_sent.index = [f"Msg {i+1}" for i in range(len(mensagens))]
    st.dataframe(df_sent, use_container_width=True)
    if "score" in df_sent.columns:
        st.line_chart(df_sent["score"], height=180, use_container_width=True)

st.divider()
st.subheader("☁️ Nuvem de Palavras")
if wordcloud_img:
    st.image(wordcloud_img, caption="Termos mais frequentes", use_container_width=True)
else:
    st.info("Sem imagem gerada.")

st.divider()
st.subheader("🔗 Grafo: Palavras Relacionadas")
if grafo and isinstance(grafo, nx.Graph) and len(grafo.nodes) > 0:
    net = Network(height="520px", width="100%")
    net.barnes_hut()
    for node, data in grafo.nodes(data=True):
        net.add_node(node, label=node, title=f"Freq: {data.get('count', 1)}")
    for u, v, data in grafo.edges(data=True):
        net.add_edge(u, v, value=data.get("weight", 1))
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        graph_html = f.read()
    components.html(graph_html, height=540, scrolling=True)
else:
    st.warning("Grafo indisponível ou sem dados suficientes.")
