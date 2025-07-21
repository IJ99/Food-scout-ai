#!/bin/bash

# Start Ollama server in the background
ollama serve &

# Wait for Ollama to be ready
sleep 10

# Pull the model (or skip if already pulled)
ollama pull gemma2:2b

# Start the FastAPI app using Render's assigned port
python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
