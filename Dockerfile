# Multi-stage production build for Bangla Search Engine
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --user .

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application source, rules, and frontend static assets
COPY rules ./rules
COPY src ./src
COPY web ./web

# Ensure data and results directories exist
RUN mkdir -p data/index results

# Expose default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/stats')" || exit 1

# Run Uvicorn server
CMD ["uvicorn", "bsearch.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
