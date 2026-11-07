#!/usr/bin/env bash
# Rebuild the app image and restart the app container with it (Ollama keeps running).
#
#   project-scripts/build.sh              rebuild (cached layers) and restart the app
#   project-scripts/build.sh --no-cache   rebuild from scratch (re-downloads dependencies and face models)
#   project-scripts/build.sh --pull       also pull newer base images (python, ollama)
set -euo pipefail

cd "$(dirname "$0")/.."

build_args=()
pull=false
for arg in "$@"; do
  case "$arg" in
    --no-cache) build_args+=(--no-cache) ;;
    --pull) pull=true; build_args+=(--pull) ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f .env ]] || die "no .env yet; run project-scripts/start.sh first (it configures this machine)"

if $pull && docker compose config --services | grep -qx llm; then
  info "pulling the Ollama image"
  docker compose pull llm
  docker compose up -d llm
fi

info "building the app image"
docker compose build "${build_args[@]}" app

info "restarting the app with the new image"
docker compose up -d app

port=$(docker compose port app 8080 | awk -F: '{ print $NF }')
url="http://localhost:${port:-8080}"
for _ in $(seq 60); do
  status=$(curl -fsS "$url/api/status" 2>/dev/null) && break
  sleep 1
done
[[ -n "${status:-}" ]] || die "app did not respond at $url; see: docker compose logs app"

info "status: $status"
info "ready: $url"
