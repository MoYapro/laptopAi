#!/usr/bin/env bash
# Send a message to Housemate as a developer tool and print its reply.
# It is labeled as coming from a tool (not a person in the room) and shown on the page as such.
#
#   project-scripts/send.sh --from "Claude Code" "Hi Housemate, quick check: who is in view?"
#   project-scripts/send.sh --from "Tom's script" --camera "What do you see right now?"
#
# Options: --from NAME (sender, default "a developer tool"), --camera, --speak, --timings
set -euo pipefail

cd "$(dirname "$0")/.."

if ! docker compose ps --status running --services 2>/dev/null | grep -qx app; then
  echo "error: the app isn't running; start it with project-scripts/start.sh" >&2
  exit 1
fi
exec docker compose exec -T app python /app/tools/send.py "$@"
