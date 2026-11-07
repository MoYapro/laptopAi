#!/usr/bin/env bash
# Stop laptopai. Containers are removed; models, memory (data/) and volumes are kept.
#
#   project-scripts/stop.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# --profile llm: also stop Ollama even if COMPOSE_PROFILES was changed since it started.
docker compose --profile llm down --remove-orphans
printf '\033[1;34m==>\033[0m stopped (data/ and model volumes kept)\n'
