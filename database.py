# database.py
"""
Gerenciamento de mensagens com PostgreSQL (com .env)
Cria tabela `messages` e expõe operações simples.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import List, Dict, Optional

from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Tenta carregar psycopg2
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import SimpleConnectionPool
    PSYCOPG2_AVAILABLE = True
except Exception as e:
    print(f"❌ psycopg2 indisponível: {e}")
    PSYCOPG2_AVAILABLE = False

# Config de conexão (usa .env, SEM defaults de localhost quando possível)
PG_HOST = os.getenv("POSTGRES_HOST") or "localhost"
PG_PORT = int(os.getenv("POSTGRES_PORT") or 5432)
PG_DB   = os.getenv("POSTGRES_DB") or "atendimento_db"
PG_USER = os.getenv("POSTGRES_USER") or "atendimento_user"
PG_PASS = os.getenv("POSTGRES_PASSWORD") or ""

PG_SSLMODE = os.getenv("POSTGRES_SSLMODE", "disable")  # compatível com clusters internos

_POOL: Optional[SimpleConnectionPool] = None
DATABASE_AVAILABLE: bool = False


def _dsn() -> str:
    # Monta DSN explícito para logs e criação do pool
    # (não logamos a senha)
    return (
        f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} "
        f"password={'***' if PG_PASS else ''} sslmode={PG_SSLMODE} connect_timeout=5"
    )


def _init_pool() -> None:
    global _POOL, DATABASE_AVAILABLE
    if not PSYCOPG2_AVAILABLE:
        DATABASE_AVAILABLE = False
        return

    try:
        # pool mínimo 1, máximo 10 conexões
        _POOL = SimpleConnectionPool(
            1, 10,
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASS,
            sslmode=PG_SSLMODE,
            connect_timeout=5,
        )
        # Sanity check
        conn = _POOL.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            _POOL.putconn(conn)
        DATABASE_AVAILABLE = True
        print(f"✅ PostgreSQL conectado em {PG_HOST}:{PG_PORT} • DB={PG_DB} • SSL={PG_SSLMODE}")
    except Exception as e:
        _POOL = None
        DATABASE_AVAILABLE = False
        print(f"❌ Falha ao conectar PostgreSQL em {PG_HOST}:{PG_PORT} • {e}")


def get_connection():
    if not _POOL:
        return None
    try:
        return _POOL.getconn()
    except Exception as e:
        print(f"❌ Erro ao obter conexão do pool: {e}")
        return None


def put_connection(conn) -> None:
    try:
        if _POOL and conn:
            _POOL.putconn(conn)
    except Exception as e:
        print(f"⚠️ Erro ao devolver conexão ao pool: {e}")


def init_db() -> None:
    """Cria a tabela/índices necessários."""
    conn = get_connection()
    if not conn:
        return

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_created
                ON messages(session_id, created_at DESC);
                """
            )
        conn.commit()
        print("✅ Tabelas/índices inicializados")
    except Exception as e:
        conn.rollback()
        print(f"❌ Erro init_db: {e}")
    finally:
        put_connection(conn)


class Database:
    """Operações básicas de mensagem."""

    @staticmethod
    def add_message(session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> Dict:
        conn = get_connection()
        if not conn:
            raise RuntimeError("Sem conexão com o banco")

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO messages (session_id, role, content, timestamp, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING session_id, role, content, timestamp, metadata, created_at
                    """,
                    (
                        session_id,
                        role,
                        content,
                        datetime.utcnow(),
                        json.dumps(metadata or {}, ensure_ascii=False),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
            return {
                "session_id": row[0],
                "role": row[1],
                "content": row[2],
                "timestamp": row[3].isoformat() if row[3] else datetime.utcnow().isoformat(),
                "metadata": row[4] or {},
                "created_at": row[5].isoformat() if row[5] else None,
            }
        except Exception as e:
            conn.rollback()
            raise
        finally:
            put_connection(conn)

    @staticmethod
    def get_messages(session_id: str, limit: Optional[int] = None) -> List[Dict]:
        conn = get_connection()
        if not conn:
            return []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if limit:
                    cur.execute(
                        """
                        SELECT role, content, timestamp, metadata, created_at
                        FROM messages
                        WHERE session_id = %s
                        ORDER BY created_at ASC
                        LIMIT %s
                        """,
                        (session_id, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT role, content, timestamp, metadata, created_at
                        FROM messages
                        WHERE session_id = %s
                        ORDER BY created_at ASC
                        """,
                        (session_id,),
                    )
                rows = cur.fetchall() or []
                return [
                    {
                        "role": r.get("role"),
                        "content": r.get("content"),
                        "timestamp": (r.get("timestamp") or r.get("created_at")).isoformat()
                        if (r.get("timestamp") or r.get("created_at"))
                        else datetime.utcnow().isoformat(),
                        "metadata": r.get("metadata") or {},
                    }
                    for r in rows
                ]
        except Exception as e:
            print(f"❌ Erro get_messages: {e}")
            return []
        finally:
            put_connection(conn)

    @staticmethod
    def list_sessions(limit: int = 100) -> List[str]:
        conn = get_connection()
        if not conn:
            return []

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT session_id, MAX(created_at) as last_seen
                    FROM messages
                    GROUP BY session_id
                    ORDER BY last_seen DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall() or []
                return [r[0] for r in rows]
        except Exception as e:
            print(f"❌ Erro list_sessions: {e}")
            return []
        finally:
            put_connection(conn)

    @staticmethod
    def clear_session(session_id: str) -> None:
        conn = get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"❌ Erro clear_session: {e}")
        finally:
            put_connection(conn)


# Inicializa pool e tabelas ao importar
_init_pool()
if DATABASE_AVAILABLE:
    try:
        init_db()
    except Exception as e:
        print(f"⚠️ init_db falhou: {e}")


# # database.py
# """
# Gerenciamento de mensagens com PostgreSQL
# """

# import os
# import json
# from datetime import datetime
# from typing import List, Dict, Optional

# try:
#     import psycopg2
#     from psycopg2.extras import RealDictCursor
#     from psycopg2.pool import SimpleConnectionPool
#     PSYCOPG2_AVAILABLE = True
# except ImportError:
#     PSYCOPG2_AVAILABLE = False
#     print("⚠️ psycopg2 não instalado")

# # Configuração
# DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
# DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
# DB_NAME = os.getenv("POSTGRES_DB", "atendimento_db")
# DB_USER = os.getenv("POSTGRES_USER", "atendimento_user")
# DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# _pool = None


# def get_pool():
#     """Cria ou retorna pool de conexões"""
#     global _pool
    
#     if not PSYCOPG2_AVAILABLE:
#         return None
    
#     if _pool is None:
#         try:
#             _pool = SimpleConnectionPool(
#                 minconn=1,
#                 maxconn=10,
#                 host=DB_HOST,
#                 port=DB_PORT,
#                 database=DB_NAME,
#                 user=DB_USER,
#                 password=DB_PASSWORD
#             )
#             print(f"✅ PostgreSQL conectado: {DB_HOST}:{DB_PORT}/{DB_NAME}")
#             init_db()
#         except Exception as e:
#             print(f"❌ Erro PostgreSQL: {e}")
#             _pool = None
#     return _pool


# def get_connection():
#     """Obtém conexão do pool"""
#     pool = get_pool()
#     if pool:
#         return pool.getconn()
#     return None


# def release_connection(conn):
#     """Devolve conexão ao pool"""
#     pool = get_pool()
#     if pool and conn:
#         pool.putconn(conn)


# def init_db():
#     """Inicializa tabelas"""
#     conn = get_connection()
#     if not conn:
#         return
    
#     try:
#         cursor = conn.cursor()
        
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS messages (
#                 id SERIAL PRIMARY KEY,
#                 session_id VARCHAR(255) NOT NULL,
#                 role VARCHAR(50) NOT NULL,
#                 content TEXT NOT NULL,
#                 timestamp TIMESTAMP NOT NULL,
#                 metadata JSONB,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#         """)
        
#         cursor.execute("""
#             CREATE INDEX IF NOT EXISTS idx_session_id 
#             ON messages(session_id, created_at DESC)
#         """)
        
#         conn.commit()
#         cursor.close()
#         print("✅ Tabelas PostgreSQL inicializadas")
        
#     except Exception as e:
#         print(f"❌ Erro ao inicializar: {e}")
#         conn.rollback()
#     finally:
#         release_connection(conn)


# class Database:
#     """Classe para gerenciar mensagens no PostgreSQL"""
    
#     @staticmethod
#     def add_message(session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> Dict:
#         """Adiciona mensagem ao banco"""
#         conn = get_connection()
#         if not conn:
#             raise Exception("PostgreSQL não disponível")
        
#         try:
#             cursor = conn.cursor(cursor_factory=RealDictCursor)
            
#             timestamp = datetime.now()
            
#             # Trata metadata
#             if isinstance(metadata, str):
#                 metadata_json = metadata
#             elif isinstance(metadata, dict):
#                 metadata_json = json.dumps(metadata)
#             else:
#                 metadata_json = json.dumps({})
            
#             cursor.execute("""
#                 INSERT INTO messages (session_id, role, content, timestamp, metadata)
#                 VALUES (%s, %s, %s, %s, %s)
#                 RETURNING id, session_id, role, content, timestamp, metadata
#             """, (session_id, role, content, timestamp, metadata_json))
            
#             result = cursor.fetchone()
#             conn.commit()
#             cursor.close()
            
#             return {
#                 "id": int(result["id"]),
#                 "session_id": str(result["session_id"]),
#                 "role": str(result["role"]),
#                 "content": str(result["content"]),
#                 "timestamp": result["timestamp"].isoformat(),
#                 "metadata": json.loads(result["metadata"]) if result["metadata"] else {}
#             }
            
#         except Exception as e:
#             conn.rollback()
#             raise e
#         finally:
#             release_connection(conn)
    
#     @staticmethod
#     def get_messages(session_id: str, limit: Optional[int] = None) -> List[Dict]:
#         """Obtém mensagens de uma sessão"""
#         conn = get_connection()
#         if not conn:
#             return []
        
#         try:
#             cursor = conn.cursor(cursor_factory=RealDictCursor)
            
#             if limit:
#                 cursor.execute("""
#                     SELECT role, content, timestamp, metadata
#                     FROM messages
#                     WHERE session_id = %s
#                     ORDER BY created_at DESC
#                     LIMIT %s
#                 """, (session_id, limit))
#             else:
#                 cursor.execute("""
#                     SELECT role, content, timestamp, metadata
#                     FROM messages
#                     WHERE session_id = %s
#                     ORDER BY created_at ASC
#                 """, (session_id,))
            
#             rows = cursor.fetchall()
#             cursor.close()
            
#             messages = []
#             for row in rows:
#                 messages.append({
#                     "role": row["role"],
#                     "content": row["content"],
#                     "timestamp": row["timestamp"].isoformat(),
#                     "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
#                 })
            
#             if limit:
#                 messages.reverse()
            
#             return messages
            
#         except Exception as e:
#             print(f"❌ Erro ao buscar: {e}")
#             return []
#         finally:
#             release_connection(conn)
    
#     @staticmethod
#     def list_sessions() -> List[str]:
#         """Lista todas as sessões"""
#         conn = get_connection()
#         if not conn:
#             return []
        
#         try:
#             cursor = conn.cursor()
            
#             cursor.execute("""
#                 SELECT DISTINCT session_id
#                 FROM messages
#                 ORDER BY MAX(created_at) DESC
#             """)
            
#             rows = cursor.fetchall()
#             cursor.close()
            
#             return [row[0] for row in rows]
            
#         except Exception as e:
#             print(f"❌ Erro ao listar: {e}")
#             return []
#         finally:
#             release_connection(conn)
    
#     @staticmethod
#     def clear_session(session_id: str):
#         """Limpa mensagens de uma sessão"""
#         conn = get_connection()
#         if not conn:
#             return
        
#         try:
#             cursor = conn.cursor()
#             cursor.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
#             conn.commit()
#             cursor.close()
            
#         except Exception as e:
#             conn.rollback()
#             print(f"❌ Erro ao limpar: {e}")
#         finally:
#             release_connection(conn)


# # Inicializa e testa
# try:
#     if PSYCOPG2_AVAILABLE:
#         get_pool()
#         DATABASE_AVAILABLE = True
#     else:
#         DATABASE_AVAILABLE = False
# except:
#     DATABASE_AVAILABLE = False

# # Testa se os métodos estão acessíveis
# if DATABASE_AVAILABLE:
#     try:
#         # Testa se consegue chamar os métodos
#         assert hasattr(Database, 'get_messages'), "❌ Database.get_messages não encontrado"
#         assert hasattr(Database, 'add_message'), "❌ Database.add_message não encontrado"
#         assert hasattr(Database, 'list_sessions'), "❌ Database.list_sessions não encontrado"
#         print("✅ Todos os métodos da classe Database estão acessíveis")
#     except AssertionError as e:
#         print(f"❌ {e}")
#         DATABASE_AVAILABLE = False