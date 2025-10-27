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


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY não encontrada. Defina no ambiente (ex.: arquivo .env).")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_data
def SYSTEM_PROMPT(caminho_arquivo:str) -> str:
    try:
        with open(caminho_arquivo, "r",encoding ="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Arquivo de prompt não encontrado."    
    except Exception as e:
        return "Erro ao ler arquivo de prompt."
    
opcoes_prompt = {
        "Assistência" : "Prompts//assistencia_tecnica.txt",
        "Suporte" : "Prompts//suporte.txt",
        "Operadora" : "Prompts//operadora_tv.txt",
            }
     
st.title("Escolha o Assistente ")

# Seleção e exibição pelo sidebar
st.sidebar.header("Assistente")
escolha_usuario = st.sidebar.selectbox("Selecione o tipo de assistente:", list(opcoes_prompt.keys()))

# Obtém o caminho do arquivo com base na escolha
caminho_prompt = opcoes_prompt[escolha_usuario]

# Carrega o conteúdo do prompt
prompt = SYSTEM_PROMPT(caminho_prompt)

# Exibe o conteúdo do prompt carregado na sidebar
st.sidebar.subheader(f"Prompt carregado: {escolha_usuario}")
st.sidebar.text_area("Conteúdo do Prompt", prompt, height=300)

# Configuração do assistente (inline)

    ##################################################import os
import streamlit as st

# =========================
# Defaults por ambiente
# =========================
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")  # foco nos baratos
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))     # usado na família 4.1
DEFAULT_TOP_P = float(os.getenv("TOP_P", "1.0"))                 # usado na família 4.1
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "400"))         # todos os modelos

# Parâmetros estilo GPT-5
DEFAULT_REASONING_EFFORT = os.getenv("REASONING_EFFORT", "minimal")  # minimal|low|medium|high
DEFAULT_VERBOSITY = os.getenv("VERBOSITY", "low")                    # low|medium|high
DEFAULT_RESPONSE_FORMAT = os.getenv("RESPONSE_FORMAT", "text")       # text|json

# =========================
# Modelos baratos (OpenAI)
# =========================
MODELOS_DISPONIVEIS = [
    "gpt-5-nano",   # família GPT-5: sem temperatura/top_p
    "gpt-5-mini",   # família GPT-5: sem temperatura/top_p
    "gpt-4.1-nano", # família 4.1: usa temperatura/top_p
    "gpt-4.1-mini", # família 4.1: usa temperatura/top_p
]

# =========================
# Helpers
# =========================
def _eh_gpt5(nome: str) -> bool:
    return (nome or "").lower().startswith("gpt-5")  # cobre gpt-5-nano/mini

def config(modelo_escolhido: str) -> dict:
    """
    Retorna a configuração adequada ao modelo:
    - GPT-5 (nano/mini): reasoning_effort/verbosity/response_format + max_tokens
    - GPT-4.1 (nano/mini): temperature/top_p + max_tokens
    """
    modelo = (modelo_escolhido or "").strip()
    if _eh_gpt5(modelo):
        return {
            "model": modelo,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
            "verbosity": DEFAULT_VERBOSITY,
            "response_format": DEFAULT_RESPONSE_FORMAT,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
    else:
        return {
            "model": modelo,
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }

# =========================
# UI
# =========================
st.sidebar.header("Seleção da IA")

default_index = MODELOS_DISPONIVEIS.index(DEFAULT_OPENAI_MODEL) if DEFAULT_OPENAI_MODEL in MODELOS_DISPONIVEIS else 0

modelo_escolhido = st.sidebar.selectbox("Escolha o modelo da LLM:", MODELOS_DISPONIVEIS, index=default_index)
st.sidebar.subheader("Modelo escolhido:")
st.sidebar.write(modelo_escolhido)
cfg = config(modelo_escolhido)

# with st.expander("Ajustes avançados (opcional)"):
#     if _eh_gpt5(modelo_escolhido):
#         reasoning_options = ["minimal", "low", "medium", "high"]
#         verbosity_options = ["low", "medium", "high"]
#         response_format_options = ["text", "json"]

#         cfg["reasoning_effort"] = st.selectbox("Reasoning effort (GPT‑5)", reasoning_options, index=reasoning_options.index(cfg["reasoning_effort"]))
#         cfg["verbosity"] = st.selectbox("Verbosity (GPT‑5)", verbosity_options, index=verbosity_options.index(cfg["verbosity"]))
#         cfg["response_format"] = st.selectbox("Response format (GPT‑5)", response_format_options, index=response_format_options.index(cfg["response_format"]))
#         cfg["max_tokens"] = int(st.number_input("Máximo de tokens", min_value=64, max_value=32768, step=64, value=cfg["max_tokens"]))
#     else:
#         cfg["temperature"] = float(st.slider("Temperatura (4.1)", min_value=0.0, max_value=1.0, step=0.05, value=cfg["temperature"]))
#         cfg["top_p"] = float(st.slider("Top-p (4.1)", min_value=0.0, max_value=1.0, step=0.05, value=cfg["top_p"]))
#         cfg["max_tokens"] = int(st.number_input("Máximo de tokens", min_value=64, max_value=8192, step=64, value=cfg["max_tokens"]))

# st.subheader("Parâmetros efetivos")
# for k, v in cfg.items():
#     st.write(f"{k}: {v}")

# st.markdown("Exemplo de chamada (pseudo-código):")
# st.code(
#     """
# # Exemplo genérico: adapte conforme seu SDK
# params = cfg.copy()
# model = params.pop("model")
# messages = [{"role": "user", "content": "Pergunta do usuário"}]

# # openai.chat.completions.create(model=model, messages=messages, **params)
#     """,
#     language="python"

