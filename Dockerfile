# Lightweight Python base for the application
FROM python:3.10-slim

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Python runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy orchestration scripts and application code
COPY . .

# Grant execution permissions
RUN chmod +x run.sh

# Run the application (assumes Ollama is available at localhost:11434)
CMD ["python", "main.py"]
