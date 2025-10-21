FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_ENTRY=app/app_01.py

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc curl libfreetype6-dev libpng-dev libgl1 dos2unix ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    if [ -f /tmp/requirements.txt ]; then \
      cp /tmp/requirements.txt /tmp/requirements.fixed && \
      dos2unix /tmp/requirements.fixed || true && \
      sed -i '1s/^\xEF\xBB\xBF//' /tmp/requirements.fixed && \
      pip install --no-cache-dir -r /tmp/requirements.fixed; \
    else \
      pip install --no-cache-dir streamlit python-dotenv openai wordcloud networkx pyvis pillow; \
    fi

COPY . /app
EXPOSE 8501
CMD ["sh","-lc","test -f \"$STREAMLIT_ENTRY\" || { echo \"[ERRO] $STREAMLIT_ENTRY não encontrado em /app\"; ls -la /app; ls -la /app/app || true; exit 1; }; exec streamlit run \"$STREAMLIT_ENTRY\" --server.port ${STREAMLIT_SERVER_PORT:-8501} --server.address 0.0.0.0 --server.headless true"]
