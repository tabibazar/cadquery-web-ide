# CadQuery Web IDE - Production Dockerfile
# This container provides sandboxed code execution for CadQuery

FROM python:3.11-slim-bookworm

# Build arguments
ARG APP_USER=cadquery
ARG APP_UID=1000
ARG APP_GID=1000

# Labels
LABEL maintainer="tabibazar.com"
LABEL description="CadQuery Web IDE - Browser-based CAD modeling"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=production \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKERS=2

# Install system dependencies for CadQuery/OCC
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxrender1 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid ${APP_GID} ${APP_USER} \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --shell /bin/bash --create-home ${APP_USER}

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY index.html .

# Create temp directory for exports with proper permissions
RUN mkdir -p /tmp/cadquery_exports && chown ${APP_USER}:${APP_USER} /tmp/cadquery_exports

# Switch to non-root user
USER ${APP_USER}

# Expose port
EXPOSE ${APP_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${APP_PORT}/health')" || exit 1

# Run with gunicorn for production
CMD ["sh", "-c", "gunicorn main:app --workers ${APP_WORKERS} --worker-class uvicorn.workers.UvicornWorker --bind ${APP_HOST}:${APP_PORT} --timeout 120 --graceful-timeout 30"]
