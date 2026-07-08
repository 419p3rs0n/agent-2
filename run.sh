#!/bin/bash

set -e

# Wait for Ollama to be ready and pull the model
echo "Pulling Gemma 2 model from Ollama..."
curl -s -X POST http://ollama:11434/api/pull -d '{"name":"gemma2:7b"}' | while read line; do
    echo "$line" | grep -o '"status":"[^"]*"' || true
done

# Execute the routing pipeline
echo "Initializing Routing Pipeline..."
python main.py
