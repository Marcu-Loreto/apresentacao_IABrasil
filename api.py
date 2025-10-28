# api.py
"""
FastAPI - Porta 8000
Recebe mensagens de N8N e outros sistemas
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import os
from dotenv import load_dotenv
import uvicorn

from shared_state import SharedState

load_dotenv()

app = FastAPI(
    title="API Atendimento IA",
    description="Integração com N8N, webhooks e sistemas externos",
    version="2.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Segurança
API_KEY = os.getenv("API_KEY", "change-me-in-production")


# Modelos
class MessageRequest(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = Field(default="default")
    user_id: Optional[str] = Field(default=None)
    metadata: Optional[Dict] = Field(default={})


class MessageResponse(BaseModel):
    success: bool
    session_id: str
    timestamp: str
    message: str


# Endpoints
@app.get("/")
async def root():
    return {
        "service": "API Atendimento IA",
        "version": "2.1.0",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis": SharedState.REDIS_AVAILABLE if hasattr(SharedState, 'REDIS_AVAILABLE') else False
    }


@app.post("/message", response_model=MessageResponse)
async def receive_message(
    request: MessageRequest,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """
    Recebe mensagem de sistema externo
    
    Exemplo curl:
    curl -X POST https://seu-dominio.com/message \
      -H "Content-Type: application/json" \
      -H "X-API-Key: sua-chave" \
      -d '{"user_message": "Olá", "session_id": "test_123"}'
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    
    try:
        message = SharedState.add_message(
            session_id=request.session_id,
            role="user",
            content=request.user_message,
            metadata={
                "user_id": request.user_id,
                "source": "api",
                **request.metadata
            }
        )
        
        return MessageResponse(
            success=True,
            session_id=request.session_id,
            timestamp=message["timestamp"],
            message="Mensagem recebida e processada"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/n8n")
async def n8n_webhook(data: dict):
    """
    Webhook simplificado para N8N (sem auth)
    
    Exemplo N8N:
    POST https://seu-dominio.com/webhook/n8n
    Body: {"message": "texto", "session_id": "opcional"}
    """
    try:
        user_message = data.get("message") or data.get("text") or data.get("body")
        session_id = data.get("session_id") or data.get("chat_id") or "n8n_default"
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Campo 'message' obrigatório")
        
        message = SharedState.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
            metadata={"source": "n8n", **data}
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "timestamp": message["timestamp"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session(
    session_id: str,
    limit: Optional[int] = None,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """Obtém mensagens de uma sessão"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    
    try:
        messages = SharedState.get_messages(session_id, limit)
        return {
            "success": True,
            "session_id": session_id,
            "message_count": len(messages),
            "messages": messages
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/sessions")
async def list_sessions(x_api_key: str = Header(..., alias="X-API-Key")):
    """Lista todas as sessões"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    
    try:
        sessions = SharedState.list_sessions()
        return {
            "success": True,
            "count": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
# # api.py
# """
# API FastAPI para receber mensagens de sistemas externos (N8N, webhooks, etc)
# Porta: 8000
# """

# from fastapi import FastAPI, HTTPException, Header, Request
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from typing import Optional, List, Dict
# from datetime import datetime
# import os
# from dotenv import load_dotenv
# import uvicorn
# import hashlib
# import hmac

# from shared_state import SharedState
# # No início do app.py, após os imports
# from shared_state import SharedState
# import time

# # Adicione esta função após a configuração inicial
# def sincronizar_mensagens_api(session_id: str = "default"):
#     """
#     Sincroniza mensagens recebidas via API com o Streamlit
#     """
#     try:
#         # Obtém mensagens da API
#         mensagens_api = SharedState.get_messages(session_id)
#         mensagens_atuais = st.session_state.get("lista_mensagens", [])
        
#         # Identifica novas mensagens
#         ids_atuais = set(
#             f"{m.get('timestamp', '')}{m.get('content', '')}"
#             for m in mensagens_atuais
#         )
        
#         novas_mensagens = []
#         for msg_api in mensagens_api:
#             msg_id = f"{msg_api.get('timestamp', '')}{msg_api.get('content', '')}"
#             if msg_id not in ids_atuais and msg_api.get("role") == "user":
#                 novas_mensagens.append(msg_api)
        
#         # Adiciona novas mensagens ao Streamlit
#         for msg in novas_mensagens:
#             # Processa com correção ortográfica
#             texto_corrigido = corrigir_texto(msg["content"]) if CONFIG.get("correcao_ortografica") else msg["content"]
            
#             # Adiciona ao histórico
#             st.session_state["lista_mensagens"].append({
#                 "role": "user",
#                 "content": texto_corrigido,
#                 "timestamp": msg.get("timestamp"),
#                 "metadata": msg.get("metadata", {})
#             })
            
#             # Tokeniza
#             tokens = tokenize_pt(texto_corrigido, corrigir=False)
#             if tokens:
#                 st.session_state["user_corpus_text"] += " " + " ".join(tokens)
#                 st.session_state["user_token_sequences"].append(tokens)
            
#             # Analisa sentimento
#             if CONFIG.get("sentimento_habilitado"):
#                 resultado_sentimento = analisar_sentimento(texto_corrigido, CONFIG["modelo_sentimento"])
#                 st.session_state["sentiment_history"].append({
#                     "idx": len(st.session_state["sentiment_history"]) + 1,
#                     "label": resultado_sentimento.get("label", "neutro"),
#                     "confidence": float(resultado_sentimento.get("confidence", 0.0)),
#                     "score": _score_from_label(
#                         resultado_sentimento.get("label", "neutro"),
#                         float(resultado_sentimento.get("confidence", 0.0))
#                     )
#                 })
        
#         return len(novas_mensagens)
        
#     except Exception as e:
#         st.error(f"Erro ao sincronizar: {e}")
#         return 0

# # Adicione no sidebar, após os controles existentes
# st.sidebar.write("---")
# st.sidebar.write("### 🔄 Sincronização API")

# col_sync1, col_sync2 = st.sidebar.columns(2)

# with col_sync1:
#     session_id_api = st.text_input(
#         "Session ID",
#         value="default",
#         key="session_id_input",
#         help="ID da sessão para sincronizar com API"
#     )

# with col_sync2:
#     if st.button("🔄 Sincronizar", use_container_width=True):
#         with st.spinner("Sincronizando..."):
#             novas = sincronizar_mensagens_api(session_id_api)
#             if novas > 0:
#                 st.success(f"✅ {novas} nova(s) mensagem(ns)")
#                 time.sleep(1)
#                 st.rerun()
#             else:
#                 st.info("Nenhuma mensagem nova")

# # Auto-sincronização (opcional)
# auto_sync = st.sidebar.toggle("Auto-sync (5s)", value=False, help="Sincroniza automaticamente a cada 5 segundos")

# if auto_sync:
#     if "last_sync" not in st.session_state:
#         st.session_state["last_sync"] = time.time()
    
#     if time.time() - st.session_state["last_sync"] > 5:
#         novas = sincronizar_mensagens_api(session_id_api)
#         st.session_state["last_sync"] = time.time()
#         if novas > 0:
#             st.rerun()

# load_dotenv()

# # Configuração
# app = FastAPI(
#     title="API de Atendimento",
#     description="Recebe mensagens de sistemas externos e integra com assistente IA",
#     version="2.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Em produção, especifique domínios
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Segurança: API Key
# API_KEY = os.getenv("API_KEY", "sua-chave-secreta-aqui")
# WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "webhook-secret")


# # Modelos Pydantic
# class MessageRequest(BaseModel):
#     user_message: str = Field(..., min_length=1, max_length=5000, description="Mensagem do usuário")
#     session_id: Optional[str] = Field(default="default", description="ID da sessão")
#     user_id: Optional[str] = Field(default=None, description="ID do usuário")
#     metadata: Optional[Dict] = Field(default={}, description="Metadados adicionais")
    
#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "user_message": "Olá, preciso de ajuda com meu pedido",
#                 "session_id": "cliente_12345",
#                 "user_id": "user_789",
#                 "metadata": {
#                     "source": "n8n",
#                     "channel": "whatsapp",
#                     "priority": "high"
#                 }
#             }
#         }


# class MessageResponse(BaseModel):
#     success: bool
#     message_id: str
#     session_id: str
#     timestamp: str
#     message: str
#     metadata: Dict


# class SessionInfo(BaseModel):
#     session_id: str
#     message_count: int
#     last_update: str
#     messages: List[Dict]


# # Middleware de autenticação
# def verify_api_key(x_api_key: Optional[str] = Header(None)):
#     """Verifica API Key no header"""
#     if x_api_key != API_KEY:
#         raise HTTPException(
#             status_code=401,
#             detail="API Key inválida ou ausente. Use header 'X-API-Key'"
#         )
#     return x_api_key


# def verify_webhook_signature(request: Request, x_signature: Optional[str] = Header(None)):
#     """Verifica assinatura de webhook (para N8N, Zapier, etc)"""
#     if not x_signature:
#         return True  # Opcional
    
#     # Implementação HMAC-SHA256
#     body = request.body()
#     expected_signature = hmac.new(
#         WEBHOOK_SECRET.encode(),
#         body,
#         hashlib.sha256
#     ).hexdigest()
    
#     if not hmac.compare_digest(x_signature, expected_signature):
#         raise HTTPException(status_code=403, detail="Assinatura inválida")
#     return True


# # ═══════════════════════════════════════════════════════════════
# # ENDPOINTS
# # ═══════════════════════════════════════════════════════════════

# @app.get("/")
# async def root():
#     """Endpoint raiz - informações da API"""
#     return {
#         "service": "API de Atendimento com IA",
#         "version": "2.0.0",
#         "status": "online",
#         "endpoints": {
#             "POST /message": "Enviar mensagem",
#             "GET /session/{session_id}": "Obter sessão",
#             "GET /sessions": "Listar sessões",
#             "DELETE /session/{session_id}": "Limpar sessão"
#         },
#         "docs": "/docs"
#     }


# @app.get("/health")
# async def health_check():
#     """Health check para monitoramento"""
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "redis_available": SharedState.REDIS_AVAILABLE if hasattr(SharedState, 'REDIS_AVAILABLE') else False
#     }


# @app.post("/message", response_model=MessageResponse)
# async def receive_message(
#     request: MessageRequest,
#     x_api_key: str = Header(..., alias="X-API-Key")
# ):
#     """
#     Recebe mensagem de sistema externo (N8N, webhook, etc)
    
#     Headers:
#         X-API-Key: Chave de autenticação
        
#     Body:
#         user_message: Mensagem do usuário
#         session_id: ID da sessão (opcional, default: "default")
#         user_id: ID do usuário (opcional)
#         metadata: Dados adicionais (opcional)
#     """
#     # Verifica API Key
#     verify_api_key(x_api_key)
    
#     try:
#         # Adiciona mensagem ao estado compartilhado
#         message = SharedState.add_message(
#             session_id=request.session_id,
#             role="user",
#             content=request.user_message,
#             metadata={
#                 "user_id": request.user_id,
#                 "source": "api",
#                 **request.metadata
#             }
#         )
        
#         # Gera ID único
#         message_id = hashlib.md5(
#             f"{request.session_id}{message['timestamp']}".encode()
#         ).hexdigest()[:12]
        
#         return MessageResponse(
#             success=True,
#             message_id=message_id,
#             session_id=request.session_id,
#             timestamp=message["timestamp"],
#             message="Mensagem recebida com sucesso. Processando...",
#             metadata={
#                 "message_count": len(SharedState.get_messages(request.session_id)),
#                 "user_id": request.user_id
#             }
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Erro ao processar mensagem: {str(e)}")


# @app.get("/session/{session_id}", response_model=SessionInfo)
# async def get_session(
#     session_id: str,
#     limit: Optional[int] = None,
#     x_api_key: str = Header(..., alias="X-API-Key")
# ):
#     """
#     Obtém informações de uma sessão
    
#     Parâmetros:
#         session_id: ID da sessão
#         limit: Limite de mensagens retornadas (opcional)
#     """
#     verify_api_key(x_api_key)
    
#     try:
#         session = SharedState.get_session(session_id)
#         messages = SharedState.get_messages(session_id, limit)
        
#         return SessionInfo(
#             session_id=session_id,
#             message_count=len(messages),
#             last_update=session.get("metadata", {}).get("last_update", "N/A"),
#             messages=messages
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=f"Sessão não encontrada: {str(e)}")


# @app.get("/sessions")
# async def list_sessions(x_api_key: str = Header(..., alias="X-API-Key")):
#     """Lista todas as sessões ativas"""
#     verify_api_key(x_api_key)
    
#     try:
#         sessions = SharedState.list_sessions()
#         return {
#             "success": True,
#             "count": len(sessions),
#             "sessions": sessions
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.delete("/session/{session_id}")
# async def clear_session(
#     session_id: str,
#     x_api_key: str = Header(..., alias="X-API-Key")
# ):
#     """Limpa uma sessão específica"""
#     verify_api_key(x_api_key)
    
#     try:
#         SharedState.clear_session(session_id)
#         return {
#             "success": True,
#             "message": f"Sessão {session_id} limpa com sucesso"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/webhook/n8n")
# async def n8n_webhook(request: Request):
#     """
#     Endpoint específico para N8N (sem autenticação por API Key)
#     Use assinatura de webhook para segurança
#     """
#     try:
#         body = await request.json()
        
#         # Extrai dados do N8N
#         user_message = body.get("message") or body.get("text") or body.get("body")
#         session_id = body.get("session_id") or body.get("chat_id") or "n8n_default"
        
#         if not user_message:
#             raise HTTPException(status_code=400, detail="Campo 'message' obrigatório")
        
#         # Adiciona mensagem
#         message = SharedState.add_message(
#             session_id=session_id,
#             role="user",
#             content=user_message,
#             metadata={"source": "n8n", **body}
#         )
        
#         return {
#             "success": True,
#             "session_id": session_id,
#             "timestamp": message["timestamp"],
#             "message": "Mensagem recebida via N8N"
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     uvicorn.run(
#         "api:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )