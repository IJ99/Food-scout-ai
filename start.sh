#!/bin/bash

# Start Ollama server in background
ollama serve &

# Wait for Ollama to start
sleep 10

# Pull the model (using a smaller but good model)
ollama pull gemma2:2b

# Start the FastAPI server
python -m uvicorn main:app --host 0.0.0.0 --port 8000