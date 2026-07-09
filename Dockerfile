# Multi-stage build: dependency layer
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies with cache busting
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching (dependency layer won't rebuild on code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage: application layer
FROM python:3.10-slim

WORKDIR /app

# Copy system dependencies from base (if custom packages were installed)
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Install minimal runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY requirements.txt .
COPY main.py .
COPY validators.py .
COPY run.sh .

# Set executable permissions
RUN chmod +x run.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:11434/v1/models || exit 1

# Run the application
CMD ["python", "main.py"]
