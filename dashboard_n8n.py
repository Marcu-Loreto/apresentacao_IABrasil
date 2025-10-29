# dashboard_n8n.py
from datetime import datetime
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

st.set_page_config(
    page_title="Painel de Sentimentos", layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
    )

st.title("📊 Painel de Sentimentos")
# st.caption("Mensagens armazenadas no banco  + análise de sentimento")
st.caption(
     f"""
            <p style="color:#ef4444; font-size:0.95rem; margin-top:0;">
            <b>Powered by Neori.Tech</b> | Versão 1.1 | {datetime.now().strftime('%Y')}
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

# --- Sidebar: fonte de dados (APENAS selectbox) ---
st.sidebar.title("⚙️ Fonte de dados")

@st.cache_data(ttl=30)
def listar_sessoes(limit=500) -> list[str]:
    try:
        sessoes = SharedState.list_sessions(limit=limit) or []
        return sorted({s for s in sessoes if s})
    except Exception as e:
        st.sidebar.error(f"Erro ao listar sessões: {e}")
        return []

# dois botões lado a lado
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    atualizar_lista = st.button("🔁 Atualizar lista", use_container_width=True)
with col_btn2:
    reload_now = st.button("🔄 Recarregar dados", use_container_width=True)

if atualizar_lista:
    listar_sessoes.clear()

sessoes = listar_sessoes()
if not sessoes:
    st.sidebar.info("Nenhuma sessão encontrada no banco.")
    st.stop()

# pré-seleção pela URL (opcional)
pre = st.query_params.get("session")
idx = sessoes.index(pre) if pre in sessoes else 0

session_id = st.sidebar.selectbox("ID da sessão", options=sessoes, index=idx, key="sessao_select")
st.query_params.update({"session": session_id})  # opcional: persistir na URL

limit = st.sidebar.number_input("Limite de mensagens", min_value=1, max_value=2000, value=200, step=10)
auto_refresh = st.sidebar.toggle("Auto-refresh (a cada 10s)", value=False)

status_cols = st.sidebar.columns(2)
with status_cols[0]:
    st.metric("PostgreSQL", "ON" if getattr(SharedState, "DATABASE_AVAILABLE", False) else "OFF")
with status_cols[1]:
    st.metric("Redis", "ON" if getattr(SharedState, "REDIS_AVAILABLE", False) else "OFF")



#reload_now = st.sidebar.button("🔄 Recarregar agora", use_container_width=True)

@st.cache_data(ttl=5)
def carregar_mensagens(session: str, n: int) -> list[str]:
    try:
        registros = SharedState.get_messages(session, limit=n) or []
        return [m.get("content", "") for m in registros
                if (m.get("role") == "user" and m.get("content"))]
    except Exception as e:
        st.error(f"Erro ao carregar mensagens: {e}")
        return []

if reload_now:
    carregar_mensagens.clear()

mensagens = carregar_mensagens(session_id, limit)

# evita chamada com sessão vazia
if not session_id:
    st.info("Selecione ou digite um ID de sessão na barra lateral.")
    st.stop()

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

# st.divider()
# st.subheader("🔗 Grafo: Palavras Relacionadas")
# if grafo and isinstance(grafo, nx.Graph) and len(grafo.nodes) > 0:
#     net = Network(height="520px", width="100%")
#     net.barnes_hut()
#     for node, data in grafo.nodes(data=True):
#         net.add_node(node, label=node, title=f"Freq: {data.get('count', 1)}")
#     for u, v, data in grafo.edges(data=True):
#         net.add_edge(u, v, value=data.get("weight", 1))
#     net.save_graph("graph.html")
#     with open("graph.html", "r", encoding="utf-8") as f:
#         graph_html = f.read()
#     components.html(graph_html, height=540, scrolling=True)
# else:
#     st.warning("Grafo indisponível ou sem dados suficientes.")

st.divider()
st.subheader("🔗 Grafo: Palavras Relacionadas")

# Toggle de tema só para o grafo
dark_mode = st.toggle("🌙 Dark mode do grafo", value=True, help="Fundo escuro + nós em tons de verde")

if grafo and isinstance(grafo, nx.Graph) and len(grafo.nodes) > 0:
    # Cores do tema
    bg = "#0b1220" if dark_mode else "#ffffff"   # fundo
    fg = "#e5e7eb" if dark_mode else "#111827"   # cor do texto
    edge_col = "#64748b" if dark_mode else "#94a3b8"

    # Cria a rede com tema
    net = Network(height="520px", width="100%", bgcolor=bg, font_color=fg)
    net.barnes_hut()

    # ----- escala por frequência (data['count']) e cor em gradiente de verde -----
    counts = [d.get("count", 1) for _, d in grafo.nodes(data=True)]
    cmin, cmax = (min(counts), max(counts)) if counts else (1, 1)

    def scale(x: int | float) -> float:
        if cmax == cmin:
            return 0.5
        return (x - cmin) / (cmax - cmin)

    def mix_hex(c1: str, c2: str, t: float) -> str:
        """Interpola entre duas cores hex (#RRGGBB)."""
        a = tuple(int(c1[i:i+2], 16) for i in (1, 3, 5))
        b = tuple(int(c2[i:i+2], 16) for i in (1, 3, 5))
        m = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
        return f"#{m[0]:02x}{m[1]:02x}{m[2]:02x}"

    # Verde claro -> verde forte
    GREEN_LOW  = "#bbf7d0"   # light (Emerald-100)
    GREEN_HIGH = "#16a34a"   # strong (Emerald-600)
    BORDER     = "#10b981" if dark_mode else "#059669"  # borda mais saturada

    # Nós
    for node, data in grafo.nodes(data=True):
        freq = int(data.get("count", 1))
        t = scale(freq)
        color_bg = mix_hex(GREEN_LOW, GREEN_HIGH, t)  # mais verde quanto maior a frequência
        net.add_node(
            node,
            label=node,
            title=f"Freq: {freq}",
            value=max(5, freq),  # controla o tamanho do nó
            color={
                "background": color_bg,
                "border": BORDER,
                "highlight": {"background": color_bg, "border": BORDER},
            },
        )

    # Arestas
    for u, v, data in grafo.edges(data=True):
        net.add_edge(u, v, value=data.get("weight", 1), color=edge_col)

    # Opções visuais adicionais (tamanhos e suavização)
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "scaling": {"min": 6, "max": 36},
        "font": {"size": 14}
      },
      "edges": {
        "smooth": true
      },
      "physics": {
        "barnesHut": {"gravitationalConstant": -8000, "springLength": 160}
      }
    }
    """)

    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        graph_html = f.read()
    components.html(graph_html, height=540, scrolling=True)

else:
    st.warning("Grafo indisponível ou sem dados suficientes.")

