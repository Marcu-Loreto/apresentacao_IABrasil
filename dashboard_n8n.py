# dashboard_n8n.py
import os, requests, time
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from utils.analysis import processar_lista_mensagens
import streamlit.components.v1 as components

st.set_page_config(page_title="Dashboard N8N", layout="wide")
st.title("📊 Análises de Atendimento")
st.caption("Mensagens recebidas do N8N em tempo real")

API_BASE = st.secrets.get("API_BASE", os.getenv("API_BASE", "https://qat.voxmap.neori.tech"))
REFRESH = st.sidebar.number_input("Auto-refresh (s)", 5, 300, 15)
st.sidebar.write(f"API: {API_BASE}")

@st.cache_data(ttl=5)
def _fetch(n: int = 200):
    r = requests.get(f"{API_BASE}/ultimas", params={"n": n}, timeout=10)
    r.raise_for_status()
    data = r.json() or {}
    return data.get("mensagens", [])

# força rerun para simular auto-refresh
if "tick" not in st.session_state:
    st.session_state.tick = 0
placeholder = st.empty()
with placeholder.container():
    mensagens = _fetch(200)

if st.sidebar.button("Recarregar agora"):
    st.cache_data.clear()
    st.session_state.tick += 1
    st.rerun()

# timer simples de rerun
time.sleep(REFRESH)
st.cache_data.clear()
st.session_state.tick += 1
st.rerun()

if not mensagens:
    st.info("Sem mensagens ainda. Envie via N8N para /mensagens.")
    st.stop()

with st.spinner("🔍 Processando mensagens..."):
    sentimentos, grafo, wordcloud_img = processar_lista_mensagens(mensagens)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📥 Mensagens Recebidas")
    for i, msg in enumerate(mensagens, start=1):
        st.markdown(f"**{i}.** {msg}")

with col2:
    st.subheader("🧠 Análise de Sentimento")
    df_sent = pd.DataFrame(sentimentos)
    df_sent.index = [f"Msg {i+1}" for i in range(len(mensagens))]
    st.dataframe(df_sent, use_container_width=True)
    if "score" in df_sent.columns:
        st.line_chart(df_sent["score"], height=160, use_container_width=True)

st.divider()
st.subheader("☁️ Nuvem de Palavras")
if wordcloud_img:
    st.image(wordcloud_img, caption="Termos mais frequentes", use_container_width=True)
else:
    st.info("Sem imagem gerada.")

st.divider()
st.subheader("🔗 Grafo : Palavras Relacionadas")
if grafo and isinstance(grafo, nx.Graph) and len(grafo.nodes) > 0:
    net = Network(height="500px", width="100%")
    net.barnes_hut()
    for node, data in grafo.nodes(data=True):
        net.add_node(node, label=node, title=f"Freq: {data.get('count', 1)}")
    for u, v, data in grafo.edges(data=True):
        net.add_edge(u, v, value=data.get("weight", 1))
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        graph_html = f.read()
    components.html(graph_html, height=520, scrolling=True)
else:
    st.warning("Grafo indisponível ou sem dados suficientes.")



# import os, time, requests
# import streamlit as st
# import pandas as pd
# import networkx as nx
# from pyvis.network import Network
# from utils.analysis import processar_lista_mensagens
# import streamlit.components.v1 as components

# st.set_page_config(page_title="Dashboard N8N", layout="wide")
# st.title("📊 Análises de Atendimento")
# st.caption("Visualização dinâmica das mensagens recebidas do N8N")

# API_BASE = st.secrets.get("API_BASE", os.getenv("API_BASE", "https://qat.voxmap.neori.tech"))
# REFRESH_SEC = st.sidebar.number_input("Auto-refresh (s)", 5, 300, 10, help="Intervalo para atualizar os dados")
# st.sidebar.write(f"API: {API_BASE}")

# @st.cache_data(ttl=1)
# def _fetch(n=200):
#     r = requests.get(f"{API_BASE}/ultimas", params={"n": n}, timeout=10)
#     r.raise_for_status()
#     data = r.json() or {}
#     return data.get("mensagens", [])

# # auto refresh
# st_autorefresh = st.sidebar.toggle("Auto-refresh", value=True)
# if st_autorefresh:
#     st.experimental_set_query_params(ts=int(time.time()))

# try:
#     mensagens = _fetch(200)
# except Exception as e:
#     st.error(f"Falha ao buscar dados da API: {e}")
#     mensagens = []

# if not mensagens:
#     st.info("Ainda não há mensagens recebidas. Envie algo pelo N8N para /mensagens.")
# else:
#     with st.spinner("🔍 Processando mensagens..."):
#         sentimentos, grafo, wordcloud_img = processar_lista_mensagens(mensagens)

#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("📥 Mensagens Recebidas")
#         for i, msg in enumerate(mensagens, start=1):
#             st.markdown(f"**{i}.** {msg}")

#     with col2:
#         st.subheader("🧠 Análise de Sentimento")
#         df_sent = pd.DataFrame(sentimentos)
#         df_sent.index = [f"Msg {i+1}" for i in range(len(mensagens))]
#         st.dataframe(df_sent, use_container_width=True)
#         if "score" in df_sent.columns:
#             st.line_chart(df_sent["score"], height=160, use_container_width=True)

#     st.divider()
#     st.subheader("☁️ Nuvem de Palavras")
#     if wordcloud_img:
#         st.image(wordcloud_img, caption="Termos mais frequentes", use_container_width=True)
#     else:
#         st.info("Sem imagem gerada.")

#     st.divider()
#     st.subheader("🔗 Grafo : Palavras Relacionadas")
#     if grafo and isinstance(grafo, nx.Graph) and len(grafo.nodes) > 0:
#         net = Network(height="500px", width="100%")
#         net.barnes_hut()
#         for node, data in grafo.nodes(data=True):
#             net.add_node(node, label=node, title=f"Freq: {data.get('count', 1)}")
#         for u, v, data in grafo.edges(data=True):
#             net.add_edge(u, v, value=data.get("weight", 1))
#         net.save_graph("graph.html")
#         with open("graph.html", "r", encoding="utf-8") as f:
#             graph_html = f.read()
#         components.html(graph_html, height=520, scrolling=True)
#     else:
#         st.warning("Grafo indisponível ou sem dados suficientes.")



# # dashboard_n8n.py
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent.parent))

# import streamlit as st
# import pandas as pd
# import networkx as nx
# from pyvis.network import Network
# from utils.analysis import processar_lista_mensagens
# import streamlit.components.v1 as components

# # Simulação de dados recebidos do N8N
# # Em produção, você pode carregar isso de um endpoint, banco ou arquivo
# EXEMPLO_MENSAGENS = [
#     "Oi, meu sinal de internet está muito ruim.",
#     "A velocidade caiu bastante ontem à noite.",
#     "Não estou conseguindo assistir aos canais em HD.",
#     "Poderiam verificar se há manutenção na minha região?",
#     "Agora voltou ao normal, obrigado pelo suporte.",
#     "Estou enfrentando problemas com a fatura.",
#     "O atendimento de voces é pessimo, nunca resolvem nada."
# ]

# # Processa mensagens uma vez por execução
# with st.spinner("🔍 Processando mensagens recebidas do N8N..."):
#     sentimentos, grafo, wordcloud_img = processar_lista_mensagens(EXEMPLO_MENSAGENS)

# # ===== UI Principal =====
# st.set_page_config(page_title="Dashboard N8N", layout="wide")
# st.title("📊 Análises de Atendimento ")
# st.caption("Visualização das mensagens recebidas e processadas")

# col1, col2 = st.columns(2)
# with col1:
#     st.subheader("📥 Mensagens Recebidas")
#     for i, msg in enumerate(EXEMPLO_MENSAGENS, start=1):
#         st.markdown(f"**{i}.** {msg}")

# with col2:
#     st.subheader("🧠 Análise de Sentimento")
#     df_sent = pd.DataFrame(sentimentos)
#     df_sent.index = [f"Msg {i+1}" for i in range(len(EXEMPLO_MENSAGENS))]
#     st.dataframe(df_sent, use_container_width=True)
#     st.line_chart(df_sent['score'], height=160, use_container_width=True)

# st.divider()
# st.subheader("☁️ Nuvem de Palavras")
# if wordcloud_img:
#     st.image(wordcloud_img, caption="Termos mais frequentes", use_container_width=True)
#     st.download_button("📥 Baixar Nuvem", data=wordcloud_img, mime="image/png", file_name="wordcloud.png")
# else:
#     st.info("Sem imagem gerada.")

# st.divider()
# st.subheader("🔗 Grafo :  Palavras Relacionadas")
# if grafo and isinstance(grafo, nx.Graph) and len(grafo.nodes) > 0:
#     net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="#000000")
#     net.barnes_hut()
#     for node, data in grafo.nodes(data=True):
#         net.add_node(node, label=node, title=f"Freq: {data.get('count', 1)}")
#     for u, v, data in grafo.edges(data=True):
#         net.add_edge(u, v, value=data.get("weight", 1))
#     net.save_graph("graph.html")
#     with open("graph.html", "r", encoding="utf-8") as f:
#         graph_html = f.read()
#     components.html(graph_html, height=520, scrolling=True)
# else:
#     st.warning("Grafo indisponível ou sem dados suficientes.")
