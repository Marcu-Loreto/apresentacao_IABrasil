# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema (incluindo libpq para PostgreSQL)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia TODOS os arquivos Python
COPY database.py .
COPY shared_state.py .
COPY api.py .

# Expõe porta
EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]