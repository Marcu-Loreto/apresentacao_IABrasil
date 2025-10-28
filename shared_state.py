# shared_state.py
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading

# Configuração Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Tenta conectar Redis
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
    print(f"✅ Redis conectado em {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    REDIS_AVAILABLE = False
    redis_client = None
    print(f"⚠️ Redis indisponível: {e}. Usando arquivo JSON.")

# Fallback: arquivo JSON
STATE_FILE = Path("shared_state.json")
_file_lock = threading.Lock()


class SharedState:
    """Gerencia estado compartilhado"""
    
    @staticmethod
    def _load_from_file() -> Dict:
        """Carrega do arquivo"""
        with _file_lock:
            if STATE_FILE.exists():
                try:
                    with open(STATE_FILE, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except:
                    return {"sessions": {}}
            return {"sessions": {}}
    
    @staticmethod
    def _save_to_file(data: Dict):
        """Salva no arquivo"""
        with _file_lock:
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def get_session(session_id: str = "default") -> Dict:
        """Obtém sessão"""
        if REDIS_AVAILABLE:
            try:
                data = redis_client.get(f"session:{session_id}")
                return json.loads(data) if data else {"mensagens": [], "metadata": {}}
            except:
                pass
        
        state = SharedState._load_from_file()
        return state.get("sessions", {}).get(session_id, {"mensagens": [], "metadata": {}})
    
    @staticmethod
    def update_session(session_id: str, data: Dict):
        """Atualiza sessão"""
        if REDIS_AVAILABLE:
            try:
                redis_client.setex(
                    f"session:{session_id}",
                    86400,  # 24h
                    json.dumps(data, ensure_ascii=False)
                )
                return
            except:
                pass
        
        state = SharedState._load_from_file()
        if "sessions" not in state:
            state["sessions"] = {}
        state["sessions"][session_id] = data
        SharedState._save_to_file(state)
    
    @staticmethod
    def add_message(session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Adiciona mensagem"""
        session = SharedState.get_session(session_id)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        if "mensagens" not in session:
            session["mensagens"] = []
        
        session["mensagens"].append(message)
        session["metadata"]["last_update"] = datetime.now().isoformat()
        session["metadata"]["message_count"] = len(session["mensagens"])
        
        SharedState.update_session(session_id, session)
        return message
    
    @staticmethod
    def get_messages(session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Obtém mensagens"""
        session = SharedState.get_session(session_id)
        messages = session.get("mensagens", [])
        
        if limit:
            return messages[-limit:]
        return messages
    
    @staticmethod
    def list_sessions() -> List[str]:
        """Lista sessões"""
        if REDIS_AVAILABLE:
            try:
                keys = redis_client.keys("session:*")
                return [k.replace("session:", "") for k in keys]
            except:
                pass
        
        state = SharedState._load_from_file()
        return list(state.get("sessions", {}).keys())
    
    @staticmethod
    def clear_session(session_id: str):
        """Limpa sessão"""
        if REDIS_AVAILABLE:
            try:
                redis_client.delete(f"session:{session_id}")
                return
            except:
                pass
        
        state = SharedState._load_from_file()
        if session_id in state.get("sessions", {}):
            del state["sessions"][session_id]
            SharedState._save_to_file(state)