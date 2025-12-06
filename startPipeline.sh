#!/bin/bash
set -e

# Farben
GREEN="\e[32m"
YELLOW="\e[33m"
RED="\e[31m"
RESET="\e[0m"

echo -e "${GREEN}--- Starte RunPod Model Runner ---${RESET}"

#export HF_HOME="${HF_HOME:-/workspace/_hf_cache}"
#export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/_hf_cache}"
export HF_HOME=/workspace/_hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/_hf_cache

echo -e "${YELLOW}Verwende Cache unter: $HF_HOME${RESET}"

cd /workspace/app/src

echo -e "${GREEN}Starte main.py ...${RESET}"

# Activate venv if exists
if [ -d "/workspace/venv" ]; then
    source /workspace/venv/bin/activate
fi

# Start uvicorn with production settings
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log \
    --no-use-colors
