#!/usr/bin/env bash
# Run backend (FastAPI) and frontend (Vite) concurrently

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Check for concurrently
if ! command -v concurrently &>/dev/null; then
  echo "Installing concurrently..."
  npm install -g concurrently
fi

concurrently \
  --names "api,ui" \
  --prefix-colors "cyan,magenta" \
  "cd '$ROOT' && uv run uvicorn backend.main:app --reload --port 8000" \
  "cd '$ROOT/frontend' && npm run dev"
