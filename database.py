# database.py
"""
Gerenciamento de mensagens com PostgreSQL
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

# Configuração PostgreSQL
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
DB_NAME = os.getenv("POSTGRES_DB", "atendimento_db")
DB_USER = os.getenv("POSTGRES_USER", "atendimento_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# Pool de conexões
_pool = None


def get_pool():
    """Cria ou retorna pool de conexões"""
    global _pool
    if _pool is None:
        try:
            _pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            print(f"✅ PostgreSQL conectado: {DB_HOST}:{DB_PORT}/{DB_NAME}")
            init_db()
        except Exception as e:
            print(f"❌ Erro ao conectar PostgreSQL: {e}")
            _pool = None
    return _pool


def get_connection():
    """Obtém conexão do pool"""
    pool = get_pool()
    if pool:
        return pool.getconn()
    return None


def release_connection(conn):
    """Devolve conexão ao pool"""
    pool = get_pool()
    if pool and conn:
        pool.putconn(conn)


def init_db():
    """Inicializa tabelas do banco"""
    conn = get_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Tabela de mensagens
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Índices para performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id 
            ON messages(session_id, created_at DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON messages(timestamp)
        """)
        
        conn.commit()
        cursor.close()
        print("✅ Tabelas PostgreSQL inicializadas")
        
    except Exception as e:
        print(f"❌ Erro ao inicializar banco: {e}")
        conn.rollback()
    finally:
        release_connection(conn)


class Database:
    """Gerencia mensagens no PostgreSQL"""
    
    @staticmethod
    def add_message(session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Adiciona mensagem ao banco"""
        conn = get_connection()
        if not conn:
            raise Exception("PostgreSQL não disponível")
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            timestamp = datetime.now()
            metadata_json = json.dumps(metadata or {})
            
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, timestamp, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, session_id, role, content, timestamp, metadata
            """, (session_id, role, content, timestamp, metadata_json))
            
            result = cursor.fetchone()
            conn.commit()
            cursor.close()
            
            return {
                "id": result["id"],
                "session_id": result["session_id"],
                "role": result["role"],
                "content": result["content"],
                "timestamp": result["timestamp"].isoformat(),
                "metadata": json.loads(result["metadata"]) if result["metadata"] else {}
            }
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            release_connection(conn)
    
    @staticmethod
    def get_messages(session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Obtém mensagens de uma sessão"""
        conn = get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if limit:
                cursor.execute("""
                    SELECT role, content, timestamp, metadata
                    FROM messages
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT role, content, timestamp, metadata
                    FROM messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                """, (session_id,))
            
            rows = cursor.fetchall()
            cursor.close()
            
            messages = []
            for row in rows:
                messages.append({
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"].isoformat(),
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                })
            
            if limit:
                messages.reverse()  # Ordem cronológica
            
            return messages
            
        except Exception as e:
            print(f"❌ Erro ao buscar mensagens: {e}")
            return []
        finally:
            release_connection(conn)
    
    @staticmethod
    def list_sessions() -> List[str]:
        """Lista todas as sessões"""
        conn = get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT session_id
                FROM messages
                ORDER BY MAX(created_at) DESC
            """)
            
            rows = cursor.fetchall()
            cursor.close()
            
            return [row[0] for row in rows]
            
        except Exception as e:
            print(f"❌ Erro ao listar sessões: {e}")
            return []
        finally:
            release_connection(conn)
    
    @staticmethod
    def clear_session(session_id: str):
        """Limpa mensagens de uma sessão"""
        conn = get_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
            conn.commit()
            cursor.close()
            
        except Exception as e:
            conn.rollback()
            print(f"❌ Erro ao limpar sessão: {e}")
        finally:
            release_connection(conn)
    
    @staticmethod
    def get_session_stats(session_id: str) -> Dict:
        """Estatísticas de uma sessão"""
        conn = get_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_messages,
                    MIN(created_at) as first_message,
                    MAX(created_at) as last_message,
                    COUNT(DISTINCT role) as unique_roles
                FROM messages
                WHERE session_id = %s
            """, (session_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            return {
                "total_messages": result["total_messages"],
                "first_message": result["first_message"].isoformat() if result["first_message"] else None,
                "last_message": result["last_message"].isoformat() if result["last_message"] else None,
                "unique_roles": result["unique_roles"]
            }
            
        except Exception as e:
            print(f"❌ Erro ao buscar stats: {e}")
            return {}
        finally:
            release_connection(conn)


# Testa conexão ao importar
try:
    get_pool()
    DATABASE_AVAILABLE = True
except:
    DATABASE_AVAILABLE = False