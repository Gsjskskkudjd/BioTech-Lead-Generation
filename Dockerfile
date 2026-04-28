# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Base: shared Python environment
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install OS-level deps needed by some Python packages (lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libxml2-dev \
        libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Builder: install Python dependencies
# ---------------------------------------------------------------------------
FROM base AS builder

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Pipeline runner image
# ---------------------------------------------------------------------------
FROM base AS pipeline

COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY pipeline.py .

# Persistent volume for the SQLite database
VOLUME ["/app/data"]

CMD ["python", "pipeline.py"]

# ---------------------------------------------------------------------------
# Streamlit UI image
# ---------------------------------------------------------------------------
FROM base AS streamlit

COPY --from=builder /install /usr/local
COPY biotech_main.py .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "biotech_main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
