# SupplementsRx AI — Hugging Face Spaces (Docker SDK) image.
# Serves the FastAPI app on port 7860 as a non-root user (uid 1000).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 user

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

USER user
ENV HOME=/home/user \
    PORT=7860
WORKDIR /home/user/app

# Graph data: graph.json feeds the in-memory graph backend (no Neo4j needed).
COPY --chown=user Knowledge_Graph/graph.json Knowledge_Graph/neo4j_communities.json Knowledge_Graph/

# Server code + retrieval data (supplement_embeddings.npy is not needed at runtime).
COPY --chown=user ["Phase 5 RAG&Frontend/main.py", \
                   "Phase 5 RAG&Frontend/rag_functionality.py", \
                   "Phase 5 RAG&Frontend/phase4_langgraph.py", \
                   "Phase 5 RAG&Frontend/graph_backend.py", \
                   "Phase 5 RAG&Frontend/index.html", \
                   "Phase 5 RAG&Frontend/faiss_index.bin", \
                   "Phase 5 RAG&Frontend/supplement_names.json", \
                   "Phase 5 RAG&Frontend/supplement_metadata.json", \
                   "Phase 5 RAG&Frontend/embedding_config.json", \
                   "server/"]

WORKDIR /home/user/app/server
EXPOSE 7860

# One worker: the rate limiter and per-visitor state live in process memory.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
