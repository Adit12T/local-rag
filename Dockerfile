# Local RAG: FastAPI app + chat UI. Models run in Ollama (a separate container, or Ollama on your machine).
FROM python:3.12-slim

ARG RERANK=true
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt requirements-rerank.txt ./
RUN pip install -r requirements.txt \
 && if [ "$RERANK" = "true" ]; then pip install -r requirements-rerank.txt; fi

COPY rag ./rag
COPY config.yaml ./config.yaml
COPY sample-docs /docs

# Container defaults. Any config.yaml setting can be overridden with RAG_<SETTING>.
# RAG_HOST=0.0.0.0 is needed inside a container; docker-compose.yml publishes the port on
# 127.0.0.1 only, which is why it also sets RAG_ALLOW_UNAUTHENTICATED_BIND. Set RAG_AUTH_TOKEN
# instead if you publish the port on another address.
ENV RAG_PATHS=/docs \
    RAG_DATA_DIR=/data \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8000 \
    RAG_OLLAMA_URL=http://ollama:11434 \
    HOME=/tmp

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin rag \
 && mkdir -p /data && chown rag /data
USER rag
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"RAG_PORT\",\"8000\")}/', timeout=4)"

ENTRYPOINT ["python", "-m", "rag"]
CMD ["--config", "/app/config.yaml"]
