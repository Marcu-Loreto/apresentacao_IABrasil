# test_postgres.py
"""
Teste simples do PostgreSQL
Execute: python test_postgres.py
"""

import os
from database import Database, get_pool

def test_connection():
    """Testa conexão"""
    print("🔍 Testando PostgreSQL...")
    
    pool = get_pool()
    if not pool:
        print("❌ Pool não criado")
        return False
    
    print("✅ Pool OK")
    return True


def test_crud():
    """Testa operações CRUD"""
    print("\n📝 Testando CRUD...")
    
    try:
        # Create
        msg = Database.add_message(
            session_id="test_crud",
            role="user",
            content="Mensagem de teste",
            metadata={"test": True}
        )
        print(f"✅ CREATE: {msg}")
        
        # Read
        messages = Database.get_messages("test_crud")
        print(f"✅ READ: {len(messages)} mensagens")
        
        # List
        sessions = Database.list_sessions()
        print(f"✅ LIST: {len(sessions)} sessões")
        
        # Stats
        stats = Database.get_session_stats("test_crud")
        print(f"✅ STATS: {stats}")
        
        # Clear
        Database.clear_session("test_crud")
        print("✅ DELETE: sessão limpa")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO POSTGRESQL")
    print("=" * 60)
    
    # Mostra variáveis
    print(f"\n📊 Variáveis:")
    print(f"  POSTGRES_HOST: {os.getenv('POSTGRES_HOST')}")
    print(f"  POSTGRES_PORT: {os.getenv('POSTGRES_PORT')}")
    print(f"  POSTGRES_DB: {os.getenv('POSTGRES_DB')}")
    print(f"  POSTGRES_USER: {os.getenv('POSTGRES_USER')}")
    print(f"  POSTGRES_PASSWORD: {'***' if os.getenv('POSTGRES_PASSWORD') else 'NÃO DEFINIDA'}")
    
    # Testes
    if test_connection():
        test_crud()
    
    print("\n" + "=" * 60)