# database.py
"""
Gerenciamento de mensagens com PostgreSQL
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import SimpleConnectionPool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("⚠️ psycopg2 não instalado")

# Configuração
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
DB_NAME = os.getenv("POSTGRES_DB", "atendimento_db")
DB_USER = os.getenv("POSTGRES_USER", "atendimento_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

_pool = None


def get_pool():
    """Cria ou retorna pool de conexões"""
    global _pool
    
    if not PSYCOPG2_AVAILABLE:
        return None
    
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
            print(f"❌ Erro PostgreSQL: {e}")
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
    """Inicializa tabelas"""
    conn = get_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
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
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id 
            ON messages(session_id, created_at DESC)
        """)
        
        conn.commit()
        cursor.close()
        print("✅ Tabelas PostgreSQL inicializadas")
        
    except Exception as e:
        print(f"❌ Erro ao inicializar: {e}")
        conn.rollback()
    finally:
        release_connection(conn)


class Database:
    """Classe para gerenciar mensagens no PostgreSQL"""
    
    @staticmethod
    def add_message(session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> Dict:
        """Adiciona mensagem ao banco"""
        conn = get_connection()
        if not conn:
            raise Exception("PostgreSQL não disponível")
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            timestamp = datetime.now()
            
            # Trata metadata
            if isinstance(metadata, str):
                metadata_json = metadata
            elif isinstance(metadata, dict):
                metadata_json = json.dumps(metadata)
            else:
                metadata_json = json.dumps({})
            
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, timestamp, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, session_id, role, content, timestamp, metadata
            """, (session_id, role, content, timestamp, metadata_json))
            
            result = cursor.fetchone()
            conn.commit()
            cursor.close()
            
            return {
                "id": int(result["id"]),
                "session_id": str(result["session_id"]),
                "role": str(result["role"]),
                "content": str(result["content"]),
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
                messages.reverse()
            
            return messages
            
        except Exception as e:
            print(f"❌ Erro ao buscar: {e}")
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
            print(f"❌ Erro ao listar: {e}")
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
            print(f"❌ Erro ao limpar: {e}")
        finally:
            release_connection(conn)


# Inicializa e testa
try:
    if PSYCOPG2_AVAILABLE:
        get_pool()
        DATABASE_AVAILABLE = True
    else:
        DATABASE_AVAILABLE = False
except:
    DATABASE_AVAILABLE = False

# Testa se os métodos estão acessíveis
if DATABASE_AVAILABLE:
    try:
        # Testa se consegue chamar os métodos
        assert hasattr(Database, 'get_messages'), "❌ Database.get_messages não encontrado"
        assert hasattr(Database, 'add_message'), "❌ Database.add_message não encontrado"
        assert hasattr(Database, 'list_sessions'), "❌ Database.list_sessions não encontrado"
        print("✅ Todos os métodos da classe Database estão acessíveis")
    except AssertionError as e:
        print(f"❌ {e}")
        DATABASE_AVAILABLE = False