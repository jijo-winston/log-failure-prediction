#!/usr/bin/env bash
set -e

# Run FastAPI server (reload is helpful during dev)
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload