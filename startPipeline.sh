#!/bin/bash
set -e

# Farben
GREEN="\e[32m"
YELLOW="\e[33m"
RED="\e[31m"
RESET="\e[0m"

source /workspace/venv/bin/activate

echo -e "${GREEN}--- Starte RunPod Model Runner ---${RESET}"
echo -e "${YELLOW}Verwende Cache unter: $HF_HOME${RESET}"

cd /workspace/runpod-model-runner/src

echo -e "${GREEN}Starte main.py ...${RESET}"
exec python main.py 2>&1 | tee /workspace/runpod-model-runner/runner.log
