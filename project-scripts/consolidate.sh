#!/usr/bin/env bash
# Run Housemate's memory review now instead of waiting for an idle period, and print the summary.
# (Backs up the database first; can take a few minutes with the 9B model on CPU.)
set -euo pipefail

cd "$(dirname "$0")/.."

port=$(docker compose port app 8080 2>/dev/null | awk -F: '{ print $NF }')
[[ -n "$port" ]] || { echo "error: the app isn't running; start it with project-scripts/start.sh" >&2; exit 1; }
printf '\033[1;34m==>\033[0m memory review running…\n'
curl -fsS --max-time 1800 -X POST "http://localhost:$port/api/consolidate" | python3 -m json.tool
