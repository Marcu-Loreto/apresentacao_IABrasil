# shared_state.py
"""
Estado compartilhado entre FastAPI e Streamlit
Usa PostgreSQL via database.py
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading

# Tenta importar Database
try:
    from database import Database, DATABASE_AVAILABLE as DB_AVAILABLE
    print("✅ database.py importado com sucesso")
except ImportError as e:
    print(f"⚠️ Erro ao importar database.py: {e}")
    Database = None
    DB_AVAILABLE = False

# Configuração Redis (fallback)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Tenta Redis
try:
    import redis
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=0,
        decode_responses=True,
        socket_connect_timeout=5
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
    print(f"✅ Redis conectado: {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    REDIS_AVAILABLE = False
    redis_client = None
    print(f"⚠️ Redis indisponível: {e}")

# Fallback: JSON
STATE_FILE = Path("shared_state.json")
_file_lock = threading.Lock()


class SharedState:
    """Gerencia estado compartilhado com prioridade: PostgreSQL > Redis > JSON"""
    
    # Expõe status para debug
    DATABASE_AVAILABLE = DB_AVAILABLE
    REDIS_AVAILABLE = REDIS_AVAILABLE
    
    @staticmethod
    def add_message(session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Adiciona mensagem - prioriza PostgreSQL"""
        
        # 1ª opção: PostgreSQL
        if DB_AVAILABLE and Database:
            try:
                return Database.add_message(session_id, role, content, metadata)
            except Exception as e:
                print(f"❌ PostgreSQL add_message falhou: {e}")
        
        # 2ª opção: Redis
        if REDIS_AVAILABLE:
            try:
                session = SharedState._get_session_redis(session_id)
                message = {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
                session["mensagens"].append(message)
                redis_client.setex(
                    f"session:{session_id}",
                    86400,
                    json.dumps(session, ensure_ascii=False)
                )
                return message
            except Exception as e:
                print(f"❌ Redis add_message falhou: {e}")
        
        # 3ª opção: JSON
        return SharedState._add_message_json(session_id, role, content, metadata)
    
    @staticmethod
    def get_messages(session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Obtém mensagens - prioriza PostgreSQL"""
        
        # 1ª opção: PostgreSQL
        if DB_AVAILABLE and Database:
            try:
                messages = Database.get_messages(session_id, limit)
                if messages is not None:
                    return messages
            except Exception as e:
                print(f"❌ PostgreSQL get_messages falhou: {e}")
        
        # 2ª opção: Redis
        if REDIS_AVAILABLE:
            try:
                session = SharedState._get_session_redis(session_id)
                messages = session.get("mensagens", [])
                return messages[-limit:] if limit else messages
            except Exception as e:
                print(f"❌ Redis get_messages falhou: {e}")
        
        # 3ª opção: JSON
        return SharedState._get_messages_json(session_id, limit)
    
    @staticmethod
    def list_sessions() -> List[str]:
        """Lista sessões"""
        
        # PostgreSQL
        if DB_AVAILABLE and Database:
            try:
                sessions = Database.list_sessions()
                if sessions is not None:
                    return sessions
            except Exception as e:
                print(f"❌ PostgreSQL list_sessions falhou: {e}")
        
        # Redis
        if REDIS_AVAILABLE:
            try:
                keys = redis_client.keys("session:*")
                return [k.replace("session:", "") for k in keys]
            except Exception as e:
                print(f"❌ Redis list_sessions falhou: {e}")
        
        # JSON
        return SharedState._list_sessions_json()
    
    @staticmethod
    def clear_session(session_id: str):
        """Limpa sessão"""
        
        if DB_AVAILABLE and Database:
            try:
                Database.clear_session(session_id)
                return
            except Exception as e:
                print(f"❌ PostgreSQL clear_session falhou: {e}")
        
        if REDIS_AVAILABLE:
            try:
                redis_client.delete(f"session:{session_id}")
                return
            except Exception as e:
                print(f"❌ Redis clear_session falhou: {e}")
        
        SharedState._clear_session_json(session_id)
    
    # Métodos auxiliares Redis
    @staticmethod
    def _get_session_redis(session_id: str) -> Dict:
        data = redis_client.get(f"session:{session_id}")
        return json.loads(data) if data else {"mensagens": [], "metadata": {}}
    
    # Métodos auxiliares JSON
    @staticmethod
    def _add_message_json(session_id: str, role: str, content: str, metadata: Optional[Dict]):
        with _file_lock:
            state = SharedState._load_json()
            if session_id not in state["sessions"]:
                state["sessions"][session_id] = {"mensagens": [], "metadata": {}}
            
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            state["sessions"][session_id]["mensagens"].append(message)
            SharedState._save_json(state)
            return message
    
    @staticmethod
    def _get_messages_json(session_id: str, limit: Optional[int]) -> List[Dict]:
        with _file_lock:
            state = SharedState._load_json()
            messages = state.get("sessions", {}).get(session_id, {}).get("mensagens", [])
            return messages[-limit:] if limit else messages
    
    @staticmethod
    def _list_sessions_json() -> List[str]:
        with _file_lock:
            state = SharedState._load_json()
            return list(state.get("sessions", {}).keys())
    
    @staticmethod
    def _clear_session_json(session_id: str):
        with _file_lock:
            state = SharedState._load_json()
            if session_id in state.get("sessions", {}):
                del state["sessions"][session_id]
                SharedState._save_json(state)
    
    @staticmethod
    def _load_json() -> Dict:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"sessions": {}}
        return {"sessions": {}}
    
    @staticmethod
    def _save_json(data: Dict):
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)