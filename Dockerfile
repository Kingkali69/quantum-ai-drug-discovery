# Multi-stage Docker build for Quantum-AI Drug Discovery Framework
# Built by KK&GDevOps LLC - Production-Ready Architecture

FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    librdkit-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    jupyterlab \
    pytest \
    black \
    flake8 \
    mypy

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models results logs

# Set permissions
RUN chmod +x main.py

# Expose ports
EXPOSE 8888 5000

# Default command
CMD ["python", "main.py", "--help"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ ./src/
COPY main.py .
COPY requirements.txt .
COPY LICENSE .
COPY README.md .

# Create non-root user
RUN useradd -m -u 1000 quantum && \
    chown -R quantum:quantum /app

USER quantum

# Create necessary directories
RUN mkdir -p data models results logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Production command
CMD ["python", "main.py", "full-pipeline", "--quick"]

# GPU-enabled stage (optional)
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install GPU-specific packages
RUN pip3 install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv

COPY . .

CMD ["python3", "main.py", "full-pipeline"]
