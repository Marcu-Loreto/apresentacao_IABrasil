# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from streamlit.components.v1 import html as st_html
from openai import OpenAI
import json
from pathlib import Path
import re
from io import BytesIO
from collections import Counter
import base64
import time
import pickle
from datetime import datetime

# Imports opcionais
try:
    from difflib import SequenceMatcher
    _SEQUENCEMATCHER_AVAILABLE = True
except Exception:
    SequenceMatcher = None
    _SEQUENCEMATCHER_AVAILABLE = False

try:
    from wordcloud import WordCloud
    _WORDCLOUD_AVAILABLE = True
except Exception:
    _WORDCLOUD_AVAILABLE = False

try:
    import networkx as nx
    from pyvis.network import Network
    _GRAPH_AVAILABLE = True
except Exception:
    nx = None
    Network = None
    _GRAPH_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except Exception:
    pd = None
    _PANDAS_AVAILABLE = False

# Carrega variáveis do .env
load_dotenv()

# ═══════════════════════════════════════════════════════════════
# CORRETOR ORTOGRÁFICO INTEGRADO
# ═══════════════════════════════════════════════════════════════

CORREÇÕES_ORTOGRÁFICAS = {
    # Erros comuns de digitação
    "tbm": "também",
    "vc": "você",
    "tb": "também",
    "q": "que",
    "eh": "é",
    "mt": "muito",
    "td": "tudo",
    "blz": "beleza",
    "obg": "obrigado",
    "vlw": "valeu",
    "pq": "porque",
    "ñ": "não",
    "oq": "o que",
    "dps": "depois",
    "hj": "hoje",
    "amg": "amigo",
    "msg": "mensagem",
    "msm": "mesmo",
    "cmg": "comigo",
    # Erros de acentuação comuns
    "nao": "não",
    "entao": "então",
    "voce": "você",
    "esta": "está",
    "ate": "até",
    "porem": "porém",
    "tambem": "também",
    "numero": "número",
    "telefone": "telefone",
    "codigo": "código",
    "pedido": "pedido",
    "prazo": "prazo",
    "endereco": "endereço",
    "reclamacao": "reclamação",
    "solucao": "solução",
    "atencao": "atenção",
    "informacao": "informação",
}


def corrigir_palavra(palavra: str) -> str:
    """Corrige uma palavra usando dicionário de correções."""
    palavra_lower = palavra.lower()
    
    if palavra_lower in CORREÇÕES_ORTOGRÁFICAS:
        correcao = CORREÇÕES_ORTOGRÁFICAS[palavra_lower]
        
        # Preserva capitalização
        if palavra[0].isupper():
            return correcao.capitalize()
        return correcao
    
    return palavra


def corrigir_texto(texto: str) -> str:
    """Corrige ortografia de um texto completo."""
    tokens = re.findall(r'\b\w+\b|[^\w\s]', texto)
    
    corrigido = []
    for token in tokens:
        if re.match(r'\w+', token):
            corrigido.append(corrigir_palavra(token))
        else:
            corrigido.append(token)
    
    return ' '.join(corrigido)


# ═══════════════════════════════════════════════════════════════
# ASSISTENTE DEFINIDO 100% NO CÓDIGO
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Você é o Assistente de Atendimento e Conciliação da empresa.
Missão: resolver solicitações de clientes com rapidez, cordialidade e foco em acordos justos.
Você é um assistente que responde apenas após a primeira mensagem do usuário.
Não peça nome nem dados pessoais por padrão.
Se a conversa estiver vazia, não diga nada.

Princípios:
1) Clareza, objetividade e empatia; trate o cliente pelo nome se fornecido.
2) Confirme entendimento do caso em 1 frase antes de propor solução.
3) Traga opções de conciliação: reenvio, abatimento, reembolso (parcial/total), crédito em conta, cupom.
4) Explique prazos, documentos necessários e próximos passos com bullets curtos.
5) Se faltar informação, faça no máximo 2 perguntas diretas e relevantes.
6) Evite jargões; linguagem simples e educada.
7) Respeite políticas: não prometa o que não pode cumprir; se necessário, escale ao time responsável.
8) Proteção de dados: não invente dados do cliente; confirme somente o que foi informado.

Formato da resposta:
- Resumo do caso:
- Solução proposta:
- Próximos passos:
- Observações:

Exemplo de tom:
"Entendi o ocorrido e quero resolver isso da forma mais rápida e justa para você."
"""

CONFIG = {
    "modelo_padrao": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "modelo_sentimento": os.getenv("OPENAI_SENTIMENT_MODEL", "gpt-4o-mini"),
    "temperatura_padrao": 0.3,
    "max_tokens_padrao": 500,
    "max_contexto_mensagens": 20,
    "max_contexto_rag": 3,
    "sentimento_habilitado": True,
    "correcao_ortografica": True,
}

# ═══════════════════════════════════════════════════════════════
# VALIDAÇÃO E CLIENTE OPENAI
# ═══════════════════════════════════════════════════════════════

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("🔒 OPENAI_API_KEY não encontrada. Defina no arquivo .env")
    st.stop()

if not OPENAI_API_KEY.startswith("sk-"):
    st.error("🔒 OPENAI_API_KEY inválida. Deve começar com 'sk-'")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


def obter_mensagens_completas():
    """Retorna mensagens com janela deslizante para otimizar tokens."""
    max_msgs = CONFIG["max_contexto_mensagens"]
    msgs_usuario = st.session_state.get("lista_mensagens", [])
    
    msgs_recentes = msgs_usuario[-max_msgs:] if len(msgs_usuario) > max_msgs else msgs_usuario
    
    return [{"role": "system", "content": SYSTEM_PROMPT}] + msgs_recentes


def call_llm(
    user_message: str,
    *,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
) -> str:
    """Chamada robusta à API OpenAI com parâmetros configuráveis."""
    model = model or CONFIG["modelo_padrao"]
    temperature = temperature if temperature is not None else CONFIG["temperatura_padrao"]
    max_tokens = max_tokens or CONFIG["max_tokens_padrao"]
    
    messages = obter_mensagens_completas()
    messages.append({"role": "user", "content": user_message})
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.error(f"❌ Erro na API OpenAI: {str(e)}")
        return f"Desculpe, ocorreu um erro ao processar sua mensagem: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# ANÁLISE DE SENTIMENTO
# ═══════════════════════════════════════════════════════════════

def _formatar_prompt_sentimento(texto: str) -> str:
    return (
        "Você é um classificador de sentimento. Classifique a mensagem a seguir.\n"
        "Responda APENAS com JSON válido com as chaves exatamente assim:\n"
        '{"label":"positivo|neutro|negativo","confidence":0.0-1.0,"emotions":["..."],"reason":"..."}\n'
        "Mensagem:\n"
        f"{texto.strip()}"
    )


def analisar_sentimento(texto: str, modelo_sentimento: str):
    """Analisa sentimento usando LLM."""
    try:
        resp = client.chat.completions.create(
            model=modelo_sentimento,
            messages=[
                {"role": "system", "content": "Retorne JSON estrito."},
                {"role": "user", "content": _formatar_prompt_sentimento(texto)},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        
        raw = resp.choices[0].message.content.strip()
        
        if raw.startswith("```"):
            raw = re.sub(r'```json\s*|\s*```', '', raw)
        
        data = json.loads(raw)
        
        label = str(data.get("label", "neutro")).lower()
        if label not in {"positivo", "neutro", "negativo"}:
            label = "neutro"
        
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        
        emotions = data.get("emotions", [])
        if not isinstance(emotions, list):
            emotions = [str(emotions)]
        
        reason = str(data.get("reason", "")).strip()
        
        return {
            "label": label,
            "confidence": conf,
            "emotions": [str(e) for e in emotions if str(e).strip()],
            "reason": reason,
        }
        
    except Exception as e:
        return {
            "label": "neutro",
            "confidence": 0.0,
            "emotions": [],
            "reason": f"Falha na análise: {e}",
        }


def _score_from_label(label: str, confidence: float) -> float:
    """Converte rótulo + confiança em score ∈ [-1, 1]."""
    sgn = 1 if label == "positivo" else (-1 if label == "negativo" else 0)
    c = max(0.0, min(1.0, float(confidence)))
    return round(sgn * c, 3)


# ═══════════════════════════════════════════════════════════════
# TOKENIZAÇÃO PT-BR
# ═══════════════════════════════════════════════════════════════

_PT_STOPWORDS = {
    "a", "à", "às", "ao", "aos", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "dá", "dão", "em", "no", "na", "nos", "nas",
    "por", "para", "pra", "com", "sem", "entre", "sobre", "sob", "até", "após",
    "que", "se", "é", "ser", "são", "era", "eram", "foi", "fui", "vai", "vou",
    "e", "ou", "mas", "como", "quando", "onde", "qual", "quais", "porque", "porquê",
    "já", "não", "sim", "também", "mais", "menos", "muito", "muita", "muitos", "muitas",
    "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas",
    "depois", "antes", "este", "esta", "estes", "estas", "isso", "isto",
    "aquele", "aquela", "aqueles", "aquelas", "lhe", "lhes", "ele", "ela", "eles", "elas",
    "você", "vocês", "nós", "nosso", "nossa", "nossos", "nossas",
}


def tokenize_pt(texto: str, corrigir: bool = True):
    """Tokeniza texto em PT-BR, remove stopwords e opcionalmente corrige ortografia."""
    if corrigir and CONFIG.get("correcao_ortografica", True):
        texto = corrigir_texto(texto)
    
    texto = texto.lower()
    tokens = re.findall(r'[a-zA-ZÀ-ÿ]+', texto)
    tokens = [t for t in tokens if len(t) >= 3 and t not in _PT_STOPWORDS]
    
    return tokens


def gerar_wordcloud(corpus_text: str, width: int = 450, height: int = 280):
    """Gera WordCloud a partir do corpus."""
    if not corpus_text.strip():
        return None, "Digite algo para iniciar a nuvem de palavras."
    
    if not _WORDCLOUD_AVAILABLE:
        return None, "Pacote 'wordcloud' não encontrado. Instale: pip install wordcloud"
    
    try:
        wc = WordCloud(
            width=width,
            height=height,
            background_color="white",
            collocations=False,
            max_words=100,
            relative_scaling=0.5,
            min_font_size=8
        )
        wc.generate(corpus_text)
        
        img = wc.to_image()
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        
        return buf, None
        
    except Exception as e:
        return None, f"Erro ao gerar wordcloud: {e}"


# ═══════════════════════════════════════════════════════════════
# GRAFO DE PALAVRAS
# ═══════════════════════════════════════════════════════════════

def build_word_graph(token_sequences, min_edge_weight: int = 1, max_nodes: int = 500):
    """Constrói grafo de coocorrências com limite de nós."""
    if not _GRAPH_AVAILABLE:
        return None
    
    G = nx.Graph()
    node_counts = Counter()
    edge_counts = Counter()
    
    for seq in token_sequences:
        node_counts.update(seq)
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a == b:
                continue
            edge = tuple(sorted((a, b)))
            edge_counts[edge] += 1
    
    if len(node_counts) > max_nodes:
        top_words = set([w for w, _ in node_counts.most_common(max_nodes)])
        node_counts = {w: c for w, c in node_counts.items() if w in top_words}
        edge_counts = {
            (a, b): c for (a, b), c in edge_counts.items()
            if a in top_words and b in top_words
        }
    
    for w, c in node_counts.items():
        G.add_node(w, count=int(c))
    
    for (a, b), w in edge_counts.items():
        if w >= max(1, int(min_edge_weight)):
            G.add_edge(a, b, weight=int(w))
    
    return G


def subgraph_paths_to_target(G, target: str, max_depth: int = 4):
    """Extrai subgrafo com caminhos até o alvo."""
    if G is None or target not in G:
        return None
    
    visited = {target}
    frontier = {target}
    depth = 0
    
    while frontier and depth < max_depth:
        next_frontier = set()
        for u in frontier:
            for v in G.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    next_frontier.add(v)
        frontier = next_frontier
        depth += 1
    
    return G.subgraph(visited).copy()


def render_graph_pyvis(
    G,
    highlight_target: str = None,
    height_px: int = 600,
    dark_mode: bool = False
):
    """Renderiza grafo com PyVis."""
    if not _GRAPH_AVAILABLE or G is None or len(G) == 0:
        return None, "Grafo indisponível ou sem dados."
    
    bg = "#0f172a" if dark_mode else "#ffffff"
    fg = "#e5e7eb" if dark_mode else "#333333"
    
    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor=bg,
        font_color=fg,
        notebook=False,
        directed=False,
    )
    
    net.barnes_hut(
        gravity=-2000,
        central_gravity=0.3,
        spring_length=160,
        spring_strength=0.01,
        damping=0.9,
    )
    
    node_counts = nx.get_node_attributes(G, "count")
    max_count = max(node_counts.values()) if node_counts else 1
    
    for node, data in G.nodes(data=True):
        count = int(data.get("count", 1))
        size = 10 + (30 * (count / max_count))
        
        color_high = "#34d399" if dark_mode else "#10b981"
        color_norm = "#93c5fd" if dark_mode else "#60a5fa"
        color = color_high if node == highlight_target else color_norm
        
        title = f"{node}<br/>freq: {count}"
        net.add_node(node, label=node, size=size, color=color, title=title)
    
    for u, v, data in G.edges(data=True):
        w = int(data.get("weight", 1))
        width = 1 + min(10, w)
        title = f"{u} — {v}<br/>coocorrências: {w}"
        net.add_edge(u, v, value=w, width=width, title=title)
    
    return net.generate_html(), None


# ═══════════════════════════════════════════════════════════════
# PROCESSAMENTO DE ARQUIVOS
# ═══════════════════════════════════════════════════════════════

def processar_txt(uploaded_file):
    """Processa arquivo .txt"""
    try:
        texto = uploaded_file.read().decode('utf-8')
        return texto, None
    except UnicodeDecodeError:
        try:
            uploaded_file.seek(0)
            texto = uploaded_file.read().decode('latin-1')
            return texto, None
        except Exception as e:
            return None, f"Erro ao decodificar TXT: {e}"


def processar_csv(uploaded_file):
    """Processa arquivo .csv e extrai texto"""
    if not _PANDAS_AVAILABLE:
        return None, "Instale pandas: pip install pandas"
    
    try:
        df = pd.read_csv(uploaded_file)
        
        colunas_possiveis = ['mensagem', 'message', 'texto', 'text', 'content', 'conteudo']
        coluna_msg = None
        
        for col in df.columns:
            if col.lower() in colunas_possiveis:
                coluna_msg = col
                break
        
        if not coluna_msg:
            texto = df.to_string(index=False)
        else:
            texto = '\n'.join(df[coluna_msg].astype(str).tolist())
        
        return texto, None
        
    except Exception as e:
        return None, f"Erro ao processar CSV: {e}"


def processar_docx(uploaded_file):
    """Processa arquivo .docx"""
    try:
        import docx
        doc = docx.Document(uploaded_file)
        texto = '\n'.join([paragrafo.text for paragrafo in doc.paragraphs])
        return texto, None
    except ImportError:
        return None, "Instale python-docx: pip install python-docx"
    except Exception as e:
        return None, f"Erro ao processar DOCX: {e}"


def processar_pdf(uploaded_file):
    """Processa arquivo .pdf"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        texto = ''
        for page in pdf_reader.pages:
            texto += page.extract_text() + '\n'
        return texto, None
    except ImportError:
        return None, "Instale PyPDF2: pip install PyPDF2"
    except Exception as e:
        return None, f"Erro ao processar PDF: {e}"


def analisar_arquivo_importado(texto: str):
    """Analisa texto importado de arquivo externo."""
    if not texto or not texto.strip():
        return None, "Arquivo vazio ou sem texto válido"
    
    if CONFIG.get("correcao_ortografica", True):
        texto_corrigido = corrigir_texto(texto)
    else:
        texto_corrigido = texto
    
    tokens = tokenize_pt(texto_corrigido, corrigir=False)
    
    if not tokens:
        return None, "Nenhuma palavra válida encontrada no arquivo"
    
    linhas = [l.strip() for l in texto_corrigido.split('\n') if l.strip()]
    
    sentimentos = []
    for i, linha in enumerate(linhas[:50]):
        if len(linha) > 10:
            sent = analisar_sentimento(linha, CONFIG["modelo_sentimento"])
            sentimentos.append({
                "linha": i + 1,
                "texto": linha[:100] + "..." if len(linha) > 100 else linha,
                "sentimento": sent
            })
    
    stats = {
        "total_caracteres": len(texto),
        "total_linhas": len(linhas),
        "total_palavras": len(tokens),
        "palavras_unicas": len(set(tokens)),
        "riqueza_vocabular": len(set(tokens)) / len(tokens) * 100 if tokens else 0,
        "sentimentos_analisados": len(sentimentos),
        "top_palavras": Counter(tokens).most_common(20),
    }
    
    if sentimentos:
        scores = [_score_from_label(s["sentimento"]["label"], s["sentimento"]["confidence"]) 
                  for s in sentimentos]
        stats["sentimento_medio"] = sum(scores) / len(scores)
        stats["sentimento_geral"] = (
            "positivo" if stats["sentimento_medio"] > 0.2 else
            "negativo" if stats["sentimento_medio"] < -0.2 else
            "neutro"
        )
    
    return {
        "texto_original": texto,
        "texto_corrigido": texto_corrigido,
        "tokens": tokens,
        "linhas": linhas,
        "sentimentos": sentimentos,
        "stats": stats
    }, None


# ═══════════════════════════════════════════════════════════════
# PERSISTÊNCIA
# ═══════════════════════════════════════════════════════════════

def salvar_sessao():
    """Salva estado da sessão em arquivo."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sessao_{timestamp}.pkl"
    
    try:
        data = {
            "mensagens": st.session_state.get("lista_mensagens", []),
            "sentiment_history": st.session_state.get("sentiment_history", []),
            "corpus": st.session_state.get("user_corpus_text", ""),
            "tokens": st.session_state.get("user_token_sequences", []),
        }
        
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        
        return filename
        
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")
        return None


def carregar_sessao(uploaded_file):
    """Carrega sessão de arquivo."""
    try:
        data = pickle.load(uploaded_file)
        
        st.session_state["lista_mensagens"] = data.get("mensagens", [])
        st.session_state["sentiment_history"] = data.get("sentiment_history", [])
        st.session_state["user_corpus_text"] = data.get("corpus", "")
        st.session_state["user_token_sequences"] = data.get("tokens", [])
        
        return True
        
    except Exception as e:
        st.error(f"Erro ao carregar: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA INTERFACE
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Assistente de Atendimento",
    page_icon="🧑‍💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧑‍💬 Analisador de Conversas com Correção Ortográfica")
st.write("---")
st.caption("• 🧠 Sentimento  • ☁️ WordCloud  • 🔗 Grafo de Palavras  • ✏️ Correção Automática")

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

st.sidebar.title("⚙️ PAINEL DE CONTROLE")

# Correção Ortográfica
st.sidebar.write("### ✏️ Correção Ortográfica")
correcao_habilitada = st.sidebar.toggle(
    "Ativar correção automática",
    value=CONFIG.get("correcao_ortografica", True),
    help="Corrige erros de digitação antes da análise"
)
CONFIG["correcao_ortografica"] = correcao_habilitada

if correcao_habilitada:
    st.sidebar.caption("✅ Palavras serão corrigidas automaticamente")
else:
    st.sidebar.caption("⚠️ Usando texto original (pode ter erros)")

st.sidebar.write("---")

# Sentimento
st.sidebar.write("### 🧠 Análise de Sentimento")
sentimento_habilitado = st.sidebar.toggle(
    "Ativar análise de sentimento",
    value=CONFIG.get("sentimento_habilitado", True),
)

sent_container = st.sidebar.container()
sent_container.caption("Última mensagem do usuário")

# Evolução do Sentimento - GRÁFICO MELHORADO
st.sidebar.write("### 📈 Evolução do Sentimento")
with st.sidebar.container():
    _hist = st.session_state.get("sentiment_history", [])
    if _hist:
        _scores = [h.get("score", 0.0) for h in _hist]
        
        # Cria DataFrame para melhor controle do gráfico
        if _PANDAS_AVAILABLE:
            df_sent = pd.DataFrame({
                'Mensagem': range(1, len(_scores) + 1),
                'Score': _scores
            })
            
            # Gráfico de linha com espaçamento reduzido
            st.line_chart(
                df_sent.set_index('Mensagem'),
                height=180,
                use_container_width=True
            )
        else:
            # Fallback sem pandas
            st.line_chart(_scores, height=180, use_container_width=True)
        
        _last = _hist[-1]
        
        # Estatísticas resumidas
        col_s1, col_s2 = st.sidebar.columns(2)
        with col_s1:
            st.caption(f"**Total:** {len(_scores)}")
        with col_s2:
            st.caption(f"**Último:** {_last.get('label', '?')}")
        
        # Média e tendência
        media_score = sum(_scores) / len(_scores)
        tendencia = "↗️" if len(_scores) > 1 and _scores[-1] > _scores[-2] else "↘️" if len(_scores) > 1 and _scores[-1] < _scores[-2] else "→"
        
        st.sidebar.caption(f"**Média:** {media_score:.2f} {tendencia}")
        
    else:
        st.info("Envie uma mensagem para ver o gráfico.")

st.sidebar.write("---")

# WordCloud
st.sidebar.write("### ☁️ Nuvem de Palavras")
wc_container = st.sidebar.container()

col_wc1, col_wc2 = st.sidebar.columns(2)
with col_wc1:
    if st.button("🗑️ Limpar nuvem", use_container_width=True):
        st.session_state["user_corpus_text"] = ""
        st.session_state["user_token_sequences"] = []
        st.rerun()

st.sidebar.write("---")

# Grafo
st.sidebar.write("### 🔗 Grafo de Palavras")
graph_container = st.sidebar.container()

with graph_container:
    min_edge_weight = st.slider(
        "Mín. coocorrências (aresta)",
        1, 5, 1,
        help="Filtra arestas fracas"
    )
    
    max_path_depth = st.slider(
        "Profundidade máx. caminho",
        1, 8, 4,
        help="Caminhos até a palavra alvo"
    )
    
    show_paths_only = st.toggle(
        "Mostrar apenas caminhos até palavra alvo",
        value=True
    )
    
    graph_dark_mode = st.toggle(
        "Modo escuro (grafo)",
        value=True
    )

st.sidebar.write("---")

# Gerenciar Dados - NOVO COM TABS
st.sidebar.write("### 💾 Gerenciar Dados")

tab_sessao, tab_importar = st.sidebar.tabs(["💾 Sessão", "📁 Importar"])

with tab_sessao:
    st.caption("Salve/carregue conversas do sistema")
    
    col_save, col_load = st.columns(2)
    
    with col_save:
        if st.button("💾 Salvar", use_container_width=True, key="save_session"):
            filename = salvar_sessao()
            if filename:
                st.success(f"✅ {filename}")
    
    with col_load:
        uploaded_session = st.file_uploader(
            "Carregar sessão",
            type=["pkl"],
            label_visibility="collapsed",
            key="upload_session"
        )
        if uploaded_session:
            if carregar_sessao(uploaded_session):
                st.success("✅ Carregada!")
                st.rerun()

with tab_importar:
    st.caption("Analise arquivos externos")
    
    uploaded_file = st.file_uploader(
        "Upload de arquivo",
        type=["txt", "csv", "pdf", "docx"],
        help="Formatos: TXT, CSV, PDF, DOCX",
        label_visibility="collapsed",
        key="upload_file"
    )
    
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner(f"📄 Processando {file_ext.upper()}..."):
            # Processa conforme extensão
            if file_ext == 'txt':
                texto, erro = processar_txt(uploaded_file)
            elif file_ext == 'csv':
                texto, erro = processar_csv(uploaded_file)
            elif file_ext == 'docx':
                texto, erro = processar_docx(uploaded_file)
            elif file_ext == 'pdf':
                texto, erro = processar_pdf(uploaded_file)
            else:
                texto, erro = None, "Formato não suportado"
            
            if erro:
                st.error(f"❌ {erro}")
            elif texto:
                # Analisa o arquivo
                resultado, erro_analise = analisar_arquivo_importado(texto)
                
                if erro_analise:
                    st.error(f"❌ {erro_analise}")
                else:
                    st.success(f"✅ Arquivo processado!")
                    
                    # Salva resultado
                    st.session_state["arquivo_importado"] = resultado
                    
                    # Opções
                    st.write("**Ações:**")
                    
                    col_a1, col_a2 = st.columns(2)
                    
                    with col_a1:
                        if st.button("➕ Adicionar", use_container_width=True, key="add_file"):
                            # Adiciona ao corpus
                            st.session_state["user_corpus_text"] += " " + " ".join(resultado["tokens"])
                            st.session_state["user_token_sequences"].append(resultado["tokens"])
                            
                            # Adiciona sentimentos
                            for sent_data in resultado["sentimentos"]:
                                sent = sent_data["sentimento"]
                                st.session_state["sentiment_history"].append({
                                    "idx": len(st.session_state["sentiment_history"]) + 1,
                                    "label": sent["label"],
                                    "confidence": sent["confidence"],
                                    "score": _score_from_label(sent["label"], sent["confidence"])
                                })
                            
                            st.success("✅ Integrado!")
                            time.sleep(1)
                            st.rerun()
                    
                    with col_a2:
                        if st.button("📊 Ver", use_container_width=True, key="view_report"):
                            st.session_state["mostrar_relatorio_arquivo"] = True
                            st.rerun()

st.sidebar.write("---")

# Ações
st.sidebar.write("### 🛠️ Ações")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🗑️ Limpar chat", use_container_width=True):
        st.session_state["lista_mensagens"] = []
        st.session_state["sentimento_atual"] = None
        st.session_state["user_corpus_text"] = ""
        st.session_state["user_token_sequences"] = []
        st.session_state["sentiment_history"] = []
        st.rerun()

with col2:
    if st.button("🔄 Recarregar", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# ESTADO DA APLICAÇÃO
# ═══════════════════════════════════════════════════════════════

if "lista_mensagens" not in st.session_state:
    st.session_state["lista_mensagens"] = []

if "sentimento_atual" not in st.session_state:
    st.session_state["sentimento_atual"] = None

if "user_corpus_text" not in st.session_state:
    st.session_state["user_corpus_text"] = ""

if "user_token_sequences" not in st.session_state:
    st.session_state["user_token_sequences"] = []

if "sentiment_history" not in st.session_state:
    st.session_state["sentiment_history"] = []

if "grafo_html" not in st.session_state:
    st.session_state["grafo_html"] = ""

if "_rerun_flag" not in st.session_state:
    st.session_state["_rerun_flag"] = False

if "arquivo_importado" not in st.session_state:
    st.session_state["arquivo_importado"] = None

if "mostrar_relatorio_arquivo" not in st.session_state:
    st.session_state["mostrar_relatorio_arquivo"] = False

# ═══════════════════════════════════════════════════════════════
# RENDERIZAÇÃO DO HISTÓRICO
# ═══════════════════════════════════════════════════════════════

for msg in st.session_state["lista_mensagens"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])

# ═══════════════════════════════════════════════════════════════
# ENTRADA DO USUÁRIO
# ═══════════════════════════════════════════════════════════════

mensagem_usuario = st.chat_input("💭 Digite sua mensagem aqui...")

if mensagem_usuario:
    # Mostra mensagem original
    st.chat_message("user").write(mensagem_usuario)
    
    # Correção ortográfica
    if correcao_habilitada:
        texto_corrigido = corrigir_texto(mensagem_usuario)
        if texto_corrigido != mensagem_usuario:
            with st.expander("✏️ Texto corrigido automaticamente"):
                col_antes, col_depois = st.columns(2)
                with col_antes:
                    st.caption("**Original:**")
                    st.text(mensagem_usuario)
                with col_depois:
                    st.caption("**Corrigido:**")
                    st.text(texto_corrigido)
    else:
        texto_corrigido = mensagem_usuario
    
    # Adiciona ao histórico
    st.session_state["lista_mensagens"].append(
        {"role": "user", "content": texto_corrigido}
    )
    
    # Tokeniza
    tokens = tokenize_pt(texto_corrigido, corrigir=False)
    
    if tokens:
        st.session_state["user_corpus_text"] += " " + " ".join(tokens)
        st.session_state["user_token_sequences"].append(tokens)
    
    # Análise de Sentimento
    if sentimento_habilitado:
        with st.spinner("🧠 Analisando sentimento..."):
            resultado_sentimento = analisar_sentimento(
                texto_corrigido,
                modelo_sentimento=CONFIG["modelo_sentimento"]
            )
            st.session_state["sentimento_atual"] = resultado_sentimento
            
            idx_user = sum(
                1 for m in st.session_state["lista_mensagens"]
                if m.get("role") == "user"
            )
            
            st.session_state["sentiment_history"].append({
                "idx": idx_user,
                "label": resultado_sentimento.get("label", "neutro"),
                "confidence": float(resultado_sentimento.get("confidence", 0.0)),
                "score": _score_from_label(
                    resultado_sentimento.get("label", "neutro"),
                    float(resultado_sentimento.get("confidence", 0.0))
                ),
            })
    
    # Resposta do Assistente
    with st.chat_message("assistant"):
        with st.spinner("🤔 Pensando na resposta..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            progress_bar.empty()
            
            try:
                resposta = client.chat.completions.create(
                    model=CONFIG["modelo_padrao"],
                    messages=obter_mensagens_completas(),
                    temperature=CONFIG["temperatura_padrao"],
                    max_tokens=CONFIG["max_tokens_padrao"],
                    top_p=0.9,
                    frequency_penalty=0.1,
                )
                
                resposta_ia = resposta.choices[0].message.content or ""
                st.write(resposta_ia)
                
                st.session_state["lista_mensagens"].append(
                    {"role": "assistant", "content": resposta_ia}
                )
                
                # Recarrega visualizações
                if not st.session_state.get("_rerun_flag"):
                    st.session_state["_rerun_flag"] = True
                    st.rerun()
                else:
                    st.session_state["_rerun_flag"] = False
                
            except Exception as e:
                st.error(f"❌ Erro na API: {str(e)}")
                st.info("💡 Verifique sua chave API e conexão.")


# ═══════════════════════════════════════════════════════════════
# SIDEBAR: VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════

def _badge(label: str) -> str:
    """Cria badge colorido para o sentimento."""
    colors = {
        "positivo": "#16a34a",
        "neutro": "#6b7280",
        "negativo": "#dc2626"
    }
    color = colors.get(label, "#6b7280")
    return (
        f"<span style='background:{color};color:white;padding:4px 10px;"
        f"border-radius:999px;font-weight:600;font-size:12px;'>"
        f"{label.upper()}</span>"
    )


# Sentimento
with sent_container:
    data = st.session_state.get("sentimento_atual")
    
    if sentimento_habilitado and data:
        st.markdown(_badge(data["label"]), unsafe_allow_html=True)
        st.metric("Confiança", f"{round(data['confidence'] * 100):d}%")
        
        if data["emotions"]:
            emotes = " ".join([f"`{e}`" for e in data["emotions"][:6]])
            st.write(f"**Emoções:** {emotes}")
        
        if data.get("reason"):
            with st.expander("📝 Justificativa"):
                st.write(data["reason"])
    
    elif sentimento_habilitado:
        st.info("Envie uma mensagem para ver o sentimento.")


# WordCloud
with wc_container:
    corpus = st.session_state.get("user_corpus_text", "")
    
    if corpus.strip():
        buf, err = gerar_wordcloud(corpus)
        
        if err:
            st.warning(err)
        elif buf:
            st.image(buf, caption="Nuvem de Palavras (Corrigidas)", use_container_width=True)
            
            st.download_button(
                "📥 Baixar PNG",
                data=buf,
                file_name=f"wordcloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True,
            )
            
            tokens_unicos = len(set(corpus.split()))
            tokens_totais = len(corpus.split())
            st.caption(f"📊 {tokens_totais} palavras | {tokens_unicos} únicas")
    else:
        st.info("Digite mensagens para gerar a nuvem.")


# Grafo
with graph_container:
    token_seqs = st.session_state.get("user_token_sequences", [])
    
    if not _GRAPH_AVAILABLE:
        st.info("Instale: pip install networkx pyvis")
    
    elif len(token_seqs) == 0:
        st.info("Envie mensagens para gerar o grafo.")
    
    else:
        with st.spinner("🔗 Construindo grafo..."):
            G_full = build_word_graph(
                token_seqs,
                min_edge_weight=min_edge_weight,
                max_nodes=500
            )
        
        if G_full is None or len(G_full) == 0:
            st.warning("Grafo vazio. Envie mais mensagens.")
        
        else:
            if len(G_full.nodes()) >= 500:
                st.warning("⚠️ Mostrando top 500 palavras.")
            
            counts = nx.get_node_attributes(G_full, "count")
            words_sorted = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
            top_words = [w for w, c in words_sorted[:200]]
            
            target = st.selectbox(
                "🎯 Palavra alvo:",
                options=["(nenhuma)"] + top_words,
                help="Destaca palavra no grafo"
            )
            
            G_view = G_full
            
            if show_paths_only and target and target != "(nenhuma)":
                G_tmp = subgraph_paths_to_target(G_full, target, max_depth=max_path_depth)
                
                if G_tmp is not None and len(G_tmp) > 0:
                    G_view = G_tmp
                    st.caption(f"🔍 {len(G_view.nodes())} nós conectados a '{target}'")
                else:
                    st.info(f"Sem caminhos para '{target}'")
                    G_view = None
            
            if G_view is not None and len(G_view) > 0:
                html, gerr = render_graph_pyvis(
                    G_view,
                    highlight_target=target if target != "(nenhuma)" else None,
                    height_px=520,
                    dark_mode=graph_dark_mode
                )
                
                if gerr:
                    st.error(gerr)
                else:
                    st.session_state["grafo_html"] = html
                    
                    st.components.v1.html(html, height=540, scrolling=True)
                    
                    st.caption(
                        f"📊 {len(G_view.nodes())} nós | "
                        f"{len(G_view.edges())} arestas | "
                        f"Densidade: {nx.density(G_view):.3f}"
                    )
                    
                    col_g1, col_g2 = st.sidebar.columns(2)
                    
                    with col_g1:
                        if st.button("📱 Expandir", use_container_width=True, key="expand_graph"):
                            st.session_state["grafo_expand_main"] = True
                            st.rerun()
                    
                    with col_g2:
                        st.download_button(
                            "📥 HTML",
                            data=html,
                            file_name=f"grafo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            use_container_width=True,
                        )


# ═══════════════════════════════════════════════════════════════
# GRAFO EXPANDIDO
# ═══════════════════════════════════════════════════════════════

if st.session_state.get("grafo_expand_main") and st.session_state.get("grafo_html"):
    st.markdown("---")
    st.markdown("## 🔗 Grafo de Palavras (Visualização Expandida)")
    
    st_html(st.session_state["grafo_html"], height=820, scrolling=True)
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        if st.button("↩️ Recolher", use_container_width=True):
            st.session_state["grafo_expand_main"] = False
            st.rerun()
    
    with col_exp2:
        st.download_button(
            "📥 Baixar HTML",
            data=st.session_state["grafo_html"],
            file_name=f"grafo_expandido_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True,
        )
    
    with col_exp3:
        if _GRAPH_AVAILABLE:
            token_seqs = st.session_state.get("user_token_sequences", [])
            total_palavras = sum(len(seq) for seq in token_seqs)
            st.metric("Total Palavras", total_palavras)


# ═══════════════════════════════════════════════════════════════
# RELATÓRIO DE ARQUIVO IMPORTADO
# ═══════════════════════════════════════════════════════════════

if st.session_state.get("mostrar_relatorio_arquivo") and st.session_state.get("arquivo_importado"):
    st.markdown("---")
    st.markdown("## 📊 Análise do Arquivo Importado")
    
    resultado = st.session_state["arquivo_importado"]
    stats = resultado["stats"]
    
    # Métricas principais
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric("📝 Total Palavras", f"{stats['total_palavras']:,}")
    
    with col_m2:
        st.metric("🔤 Palavras Únicas", f"{stats['palavras_unicas']:,}")
    
    with col_m3:
        st.metric("📈 Riqueza Vocabular", f"{stats['riqueza_vocabular']:.1f}%")
    
    with col_m4:
        if "sentimento_geral" in stats:
            st.metric(
                "🎭 Sentimento Geral",
                stats["sentimento_geral"].capitalize(),
                delta=f"{stats['sentimento_medio']:.2f}"
            )
    
    # Tabs com visualizações
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Estatísticas", "🧠 Sentimentos", "☁️ WordCloud", "📄 Texto"])
    
    with tab1:
        st.write("### Top 20 Palavras Mais Frequentes")
        
        if _PANDAS_AVAILABLE:
            palavras_freq = stats["top_palavras"]
            df_palavras = pd.DataFrame(palavras_freq, columns=["Palavra", "Frequência"])
            
            st.bar_chart(df_palavras.set_index("Palavra"))
            st.dataframe(df_palavras, use_container_width=True)
        else:
            for palavra, freq in stats["top_palavras"]:
                st.write(f"**{palavra}**: {freq} vezes")
    
    with tab2:
        st.write("### Análise de Sentimento por Segmento")
        
        if resultado["sentimentos"]:
            for sent_data in resultado["sentimentos"][:20]:
                sent = sent_data["sentimento"]
                
                col_s1, col_s2 = st.columns([3, 1])
                
                with col_s1:
                    st.write(f"**Linha {sent_data['linha']}:** {sent_data['texto']}")
                
                with col_s2:
                    st.markdown(_badge(sent["label"]), unsafe_allow_html=True)
                    st.caption(f"Conf: {int(sent['confidence']*100)}%")
                
                st.divider()
        else:
            st.info("Nenhum segmento analisado")
    
    with tab3:
        st.write("### Nuvem de Palavras do Arquivo")
        
        corpus_arquivo = " ".join(resultado["tokens"])
        buf, err = gerar_wordcloud(corpus_arquivo, width=800, height=500)
        
        if err:
            st.warning(err)
        elif buf:
            st.image(buf, use_container_width=True)
            
            st.download_button(
                "📥 Baixar WordCloud",
                data=buf,
                file_name=f"wordcloud_arquivo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
            )
    
    with tab4:
        st.write("### Texto Original vs Corrigido")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.write("**Original:**")
            st.text_area(
                "original",
                resultado["texto_original"][:5000],
                height=400,
                label_visibility="collapsed"
            )
        
        with col_t2:
            st.write("**Corrigido:**")
            st.text_area(
                "corrigido",
                resultado["texto_corrigido"][:5000],
                height=400,
                label_visibility="collapsed"
            )
    
    # Botão fechar
    if st.button("✖️ Fechar Relatório"):
        st.session_state["mostrar_relatorio_arquivo"] = False
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# EXPORTAÇÃO DE RELATÓRIO
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("📊 Exportar Relatório de Análise")

col_report1, col_report2 = st.columns(2)

with col_report1:
    if st.button("📄 Gerar Relatório TXT", use_container_width=True):
        relatorio = f"""
═══════════════════════════════════════════════════════════════
RELATÓRIO DE ANÁLISE DE CONVERSAS
═══════════════════════════════════════════════════════════════

Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Modelo: {CONFIG['modelo_padrao']}

─────────────────────────────────────────────────────────────
ESTATÍSTICAS GERAIS
─────────────────────────────────────────────────────────────

Total de Mensagens: {len(st.session_state.get('lista_mensagens', []))}
Mensagens do Usuário: {sum(1 for m in st.session_state.get('lista_mensagens', []) if m['role'] == 'user')}
Mensagens do Assistente: {sum(1 for m in st.session_state.get('lista_mensagens', []) if m['role'] == 'assistant')}

─────────────────────────────────────────────────────────────
ANÁLISE DE SENTIMENTO
─────────────────────────────────────────────────────────────
"""
        
        hist = st.session_state.get("sentiment_history", [])
        if hist:
            positivos = sum(1 for h in hist if h["label"] == "positivo")
            neutros = sum(1 for h in hist if h["label"] == "neutro")
            negativos = sum(1 for h in hist if h["label"] == "negativo")
            
            relatorio += f"""
Mensagens Positivas: {positivos} ({positivos/len(hist)*100:.1f}%)
Mensagens Neutras: {neutros} ({neutros/len(hist)*100:.1f}%)
Mensagens Negativas: {negativos} ({negativos/len(hist)*100:.1f}%)

Score Médio: {sum(h['score'] for h in hist)/len(hist):.3f}
Confiança Média: {sum(h['confidence'] for h in hist)/len(hist)*100:.1f}%
"""
        else:
            relatorio += "\nNenhuma análise disponível.\n"
        
        relatorio += f"""
─────────────────────────────────────────────────────────────
ANÁLISE DE VOCABULÁRIO
─────────────────────────────────────────────────────────────
"""
        
        corpus = st.session_state.get("user_corpus_text", "")
        if corpus:
            tokens = corpus.split()
            palavras_unicas = set(tokens)
            
            relatorio += f"""
Total de Palavras: {len(tokens)}
Palavras Únicas: {len(palavras_unicas)}
Riqueza Vocabular: {len(palavras_unicas)/len(tokens)*100:.1f}%

Top 10 Palavras:
"""
            counter = Counter(tokens)
            for palavra, freq in counter.most_common(10):
                relatorio += f"  {palavra}: {freq} vezes\n"
        
        relatorio += f"""
─────────────────────────────────────────────────────────────
HISTÓRICO DE MENSAGENS
─────────────────────────────────────────────────────────────
"""
        
        for i, msg in enumerate(st.session_state.get("lista_mensagens", []), 1):
            role = "USUÁRIO" if msg["role"] == "user" else "ASSISTENTE"
            relatorio += f"\n[{i}] {role}:\n{msg['content']}\n"
        
        relatorio += "\n═══════════════════════════════════════════════════════════════\n"
        
        st.download_button(
            "📥 Baixar Relatório (.txt)",
            data=relatorio,
            file_name=f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

with col_report2:
    if st.button("📊 Gerar Relatório JSON", use_container_width=True):
        relatorio_json = {
            "metadata": {
                "data_geracao": datetime.now().isoformat(),
                "modelo": CONFIG["modelo_padrao"],
                "temperatura": CONFIG["temperatura_padrao"],
                "correcao_ortografica": correcao_habilitada,
            },
            "estatisticas": {
                "total_mensagens": len(st.session_state.get("lista_mensagens", [])),
                "mensagens_usuario": sum(1 for m in st.session_state.get("lista_mensagens", []) if m["role"] == "user"),
                "mensagens_assistente": sum(1 for m in st.session_state.get("lista_mensagens", []) if m["role"] == "assistant"),
            },
            "sentimento": {
                "historico": st.session_state.get("sentiment_history", []),
            },
            "vocabulario": {
                "corpus": st.session_state.get("user_corpus_text", ""),
                "sequencias_tokens": st.session_state.get("user_token_sequences", []),
            },
            "mensagens": st.session_state.get("lista_mensagens", []),
        }
        
        json_str = json.dumps(relatorio_json, ensure_ascii=False, indent=2)
        
        st.download_button(
            "📥 Baixar Relatório (.json)",
            data=json_str,
            file_name=f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════
# RODAPÉ
# ═══════════════════════════════════════════════════════════════

st.markdown("---")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.caption(f"**Modelo:** {CONFIG['modelo_padrao']}")
    st.caption(f"**Temperatura:** {CONFIG['temperatura_padrao']}")

with col_info2:
    total_msgs = len(st.session_state.get("lista_mensagens", []))
    msgs_user = sum(1 for m in st.session_state.get("lista_mensagens", []) if m["role"] == "user")
    st.caption(f"**Mensagens:** {total_msgs} ({msgs_user} do usuário)")

with col_info3:
    if correcao_habilitada:
        st.caption("✅ **Correção:** Ativa")
    else:
        st.caption("⚠️ **Correção:** Desativada")

st.caption(
    "🤖 Assistente de Atendimento com IA | "
    f"Powered by OpenAI | Versão 2.1 | {datetime.now().strftime('%Y')}"
)


# ═══════════════════════════════════════════════════════════════
# AJUDA E DICAS
# ═══════════════════════════════════════════════════════════════

with st.expander("💡 Dicas de Uso"):
    st.markdown("""
    ### Como usar este assistente:
    
    **🔧 Correção Ortográfica**
    - Ative na sidebar para corrigir automaticamente erros
    - Correções: "vc" → "você", "nao" → "não", etc.
    - Texto corrigido usado em todas as análises
    
    **📁 Importar Arquivos**
    - Formatos aceitos: TXT, CSV, PDF, DOCX
    - CSV: detecta automaticamente coluna de mensagens
    - Opções: adicionar ao chat atual ou ver relatório separado
    
    **🧠 Análise de Sentimento**
    - Verde = Positivo | Cinza = Neutro | Vermelho = Negativo
    - Gráfico mostra evolução em tempo real
    - Espaçamento reduzido para visualizar mudanças
    
    **☁️ Nuvem de Palavras**
    - Palavras maiores = mais frequentes
    - Apenas palavras corrigidas são exibidas
    
    **🔗 Grafo de Palavras**
    - Mostra coocorrências (palavras que aparecem juntas)
    - Use "Palavra alvo" para focar em conexões específicas
    - Limite de 500 nós para melhor performance
    
    **💾 Salvar/Carregar**
    - Salve sessões completas (.pkl)
    - Carregue sessões anteriores para continuar
    
    **📊 Relatórios**
    - TXT: relatório formatado para leitura
    - JSON: dados estruturados para análise externa
    """)

with st.expander("⚙️ Configurações Avançadas"):
    st.markdown(f"""
    ### Parâmetros do Sistema:
    
    **Modelo:** `{CONFIG['modelo_padrao']}`
    - Modelo OpenAI usado para respostas
    - gpt-4o-mini = rápido e econômico
    - gpt-4o = mais preciso e contextual
    
    **Temperatura:** `{CONFIG['temperatura_padrao']}`
    - Controla criatividade (0.0 a 1.0)
    - Baixo = mais determinístico
    - Alto = mais variado
    
    **Contexto:** `{CONFIG['max_contexto_mensagens']}` mensagens
    - Quantas mensagens são enviadas à API
    - Mais contexto = melhor memória, mais custo
    
    **Correção Ortográfica:**
    - Dicionário: {len(CORREÇÕES_ORTOGRÁFICAS)} correções
    - Processamento local (sem API)
    - Preserva capitalização original
    
    **Formatos Suportados:**
    - TXT: texto puro (UTF-8 ou Latin-1)
    - CSV: colunas automáticas ou manual
    - PDF: extração de texto (PyPDF2)
    - DOCX: parágrafos do Word (python-docx)
    """)

with st.expander("🔧 Dependências e Instalação"):
    st.markdown("""
    ### Pacotes Necessários:
    
    **Obrigatórios:**
```bash
    pip install streamlit openai python-dotenv
```
    
    **Opcionais (para recursos extras):**
```bash
    # Para WordCloud
    pip install wordcloud
    
    # Para Grafo de Palavras
    pip install networkx pyvis
    
    # Para análise de arquivos
    pip install pandas PyPDF2 python-docx
```
    
    **Arquivo .env:**
```
    OPENAI_API_KEY=sk-sua-chave-aqui
    OPENAI_MODEL=gpt-4o-mini
    OPENAI_SENTIMENT_MODEL=gpt-4o-mini
```
    
    **Executar:**
```bash
    streamlit run app.py
```
    """)