# fix_compatibility.py
"""
Script para corrigir compatibilidade de versões
Execute: python fix_compatibility.py
"""

import re
from pathlib import Path

def fix_app_py():
    """Corrige use_container_width para use_column_width"""
    app_file = Path("app.py")
    
    if not app_file.exists():
        print("❌ app.py não encontrado")
        return
    
    content = app_file.read_text(encoding='utf-8')
    
    # Substitui use_container_width por use_column_width
    new_content = content.replace(
        "use_container_width=True",
        "use_column_width=True"
    )
    
    if content != new_content:
        app_file.write_text(new_content, encoding='utf-8')
        print("✅ app.py corrigido!")
        print(f"   Substituições feitas: {content.count('use_container_width')}")
    else:
        print("ℹ️ Nenhuma alteração necessária")

if __name__ == "__main__":
    fix_app_py()