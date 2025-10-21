# VOXMAP

Autor: Marcu Loreto

Resumo
- VOXMAP é uma aplicação Streamlit que atua como Assistente de Atendimento e Conciliação.
- Usa OpenAI para gerar resumos, propostas de solução e próximos passos a partir de conversas.
- Fornece análises auxiliares (sentimento, wordcloud, grafo) quando as bibliotecas correspondentes estão instaladas.

Prerequisitos
- Python 3.12
- Git (opcional)
- Docker & Docker Compose (opcional, recomendado para VPS)
- Registro DNS apontando para sua VPS (se for usar domínio)

Configuração de variáveis (arquivo .env)
- Crie um arquivo `.env` na raiz com pelo menos:
  OPENAI_API_KEY=seu_openai_key_aqui
  OPENAI_MODEL=gpt-4.1-mini
  OPENAI_TEMPERATURE=0.2
  OPENAI_MAX_TOKENS=400

Instalar dependências e rodar no servidor (modo venv — Linux VPS)
1. Entre na pasta do projeto:
   cd /caminho/para/VOXMAP

2. Criar e ativar virtualenv:
   python -m venv .venv
   source .venv/bin/activate

3. Instalar dependências (requirements.txt deve estar em UTF-8 e com versões fixadas):
   pip install --upgrade pip
   pip install -r requirements.txt

4. Expor porta e rodar Streamlit (escuta em 0.0.0.0 para aceitar conexões externas):
   export $(cat .env | xargs)   # carrega variáveis do .env no shell (opcional)
   python -m streamlit run app/app_01.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

Windows (PowerShell)
1. No diretório do projeto:
   python -m venv .venv
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
   .\.venv\Scripts\Activate.ps1
2. Instalar dependências:
   pip install --upgrade pip
   pip install -r requirements.txt
3. Rodar:
   python -m streamlit run .\app\app_01.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

Rodando via Docker Compose (recomendado para VPS)
- Compose sem proxy (ex.: feeling.yaml)
  docker compose -f feeling.yaml up --build -d

- Compose com nginx + Let's Encrypt (recomendado para domínio público)
  docker compose -f feeling_proxy.yaml up --build -d

Obs: coloque suas credenciais em `.env` (não commite o arquivo). No feeling_proxy.yaml o serviço usa VIRTUAL_HOST e LETSENCRYPT_HOST para geração automática de certificados.

Expor por domínio com Nginx (resumo)
1. Configure DNS A apontando `www.feeling_check.etechats.com.br` para IP da VPS.
2. Opção A (docker-proxy stack): use feeling_proxy.yaml (nginx-proxy + letsencrypt companion).
3. Opção B (host Nginx): configure Nginx para proxy_pass para http://127.0.0.1:8501 e use Certbot:
   sudo certbot --nginx -d www.feeling_check.etechats.com.br

Dicas e Troubleshooting
- Verifique que `requirements.txt` está em UTF-8 e com versões fixadas compatíveis com Python 3.12.
- Se Streamlit não inicia, confira logs:
  docker compose -f feeling.yaml logs -f
  ou, com venv, verifique a saída do terminal ao executar python -m streamlit run...
- Para segurança em produção, rode o container mapeado para 127.0.0.1:8501 e use Nginx/Let's Encrypt para o TLS.

Arquivos importantes
- Dockerfile — imagem baseada em Python 3.12
- feeling.yaml — docker-compose para app
- feeling_proxy.yaml — docker-compose com nginx-proxy + letsencrypt
- requirements.txt — dependências pinadas (UTF-8)
- .env — variáveis de ambiente (NÃO comitar)

Autor
- Marcu Loreto

COMANDOS 

APP
Streamlit run app_01.py ( Roda a plicacao)

API
Uvicorn maim:app --reload 0.0.0.0 --port 8000 (Acionar de dentro do folder app)

# VOXMAP

Autor: Marcu Loreto

Resumo
- VOXMAP é uma aplicação Streamlit que atua como Assistente de Atendimento e Conciliação.
- Usa OpenAI para gerar resumos, propostas de solução e próximos passos a partir de conversas.
- Fornece análises auxiliares (sentimento, wordcloud, grafo) quando as bibliotecas correspondentes estão instaladas.

Prerequisitos
- Python 3.12
- Git (opcional)
- Docker & Docker Compose (opcional, recomendado para VPS)
- Registro DNS apontando para sua VPS (se for usar domínio)

Configuração de variáveis (arquivo .env)
- Crie um arquivo `.env` na raiz com pelo menos:
  OPENAI_API_KEY=seu_openai_key_aqui
  OPENAI_MODEL=gpt-4.1-mini
  OPENAI_TEMPERATURE=0.2
  OPENAI_MAX_TOKENS=400

Instalar dependências e rodar no servidor (modo venv — Linux VPS)
1. Entre na pasta do projeto:
   cd /caminho/para/VOXMAP

2. Criar e ativar virtualenv:
   python -m venv .venv
   source .venv/bin/activate

3. Instalar dependências (requirements.txt deve estar em UTF-8 e com versões fixadas):
   pip install --upgrade pip
   pip install -r requirements.txt

4. Expor porta e rodar Streamlit (escuta em 0.0.0.0 para aceitar conexões externas):
   export $(cat .env | xargs)   # carrega variáveis do .env no shell (opcional)
   python -m streamlit run app/app_01.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

Windows (PowerShell)
1. No diretório do projeto:
   python -m venv .venv
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
   .\.venv\Scripts\Activate.ps1
2. Instalar dependências:
   pip install --upgrade pip
   pip install -r requirements.txt
3. Rodar:
   python -m streamlit run .\app\app_01.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

Rodando via Docker Compose (recomendado para VPS)
- Compose sem proxy (ex.: feeling.yaml)
  docker compose -f feeling.yaml up --build -d

- Compose com nginx + Let's Encrypt (recomendado para domínio público)
  docker compose -f feeling_proxy.yaml up --build -d

Obs: coloque suas credenciais em `.env` (não commite o arquivo). No feeling_proxy.yaml o serviço usa VIRTUAL_HOST e LETSENCRYPT_HOST para geração automática de certificados.

Expor por domínio com Nginx (resumo)
1. Configure DNS A apontando `www.feeling_check.etechats.com.br` para IP da VPS.
2. Opção A (docker-proxy stack): use feeling_proxy.yaml (nginx-proxy + letsencrypt companion).
3. Opção B (host Nginx): configure Nginx para proxy_pass para http://127.0.0.1:8501 e use Certbot:
   sudo certbot --nginx -d www.feeling_check.etechats.com.br

Dicas e Troubleshooting
- Verifique que `requirements.txt` está em UTF-8 e com versões fixadas compatíveis com Python 3.12.
- Se Streamlit não inicia, confira logs:
  docker compose -f feeling.yaml logs -f
  ou, com venv, verifique a saída do terminal ao executar python -m streamlit run...
- Para segurança em produção, rode o container mapeado para 127.0.0.1:8501 e use Nginx/Let's Encrypt para o TLS.

Arquivos importantes
- Dockerfile — imagem baseada em Python 3.12
- voxmap.yaml — docker-compose para app
- voxmap_proxy.yaml — docker-compose com nginx-proxy + letsencrypt
- requirements.txt — dependências pinadas (UTF-8)
- .env — variáveis de ambiente (NÃO comitar)

Autor
- Marcu Loreto

COMANDOS 

APP
Streamlit run app_01.py ( Roda a plicacao)

API
Uvicorn maim:app --reload 0.0.0.0 --port 8000 (Acionar de dentro do folder app)