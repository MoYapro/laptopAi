#!/usr/bin/env bash
# Screen helper: lets Housemate look at this computer's screen. It waits until the app asks for a screenshot
# (about once a minute, and whenever the model wants to look), then captures each monitor and sends the images.
# Nothing is captured otherwise; frames are kept in the app's memory only. Ctrl+C (or killing it) stops it, and
# Housemate then knows it can't see the screen. Start it with the desktop session, e.g. in mango's autostart.sh:
#   ~/projects/laptopai/project-scripts/screen.sh >/dev/null 2>&1 &
#
#   project-scripts/screen.sh                 every monitor, each as its own image (mango or sway), else the whole screen
#   project-scripts/screen.sh -o DP-1         only this monitor (see: mmsg get all-monitors / swaymsg -t get_outputs)
#
# Wayland needs grim (wlroots compositors: mango, sway, …) and jq; X11 needs ImageMagick's import.
# SCREEN_URL overrides the app address (default: the app container's port on localhost, else :8080).
set -euo pipefail

cd "$(dirname "$0")/.."

output=""
while getopts "o:h" opt; do
  case "$opt" in
    o) output=$OPTARG ;;
    *) sed -n '2,12p' "$0"; exit 0 ;;
  esac
done

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

if [[ -z "${SCREEN_URL:-}" ]]; then
  port=$(docker compose port app 8080 2>/dev/null | awk -F: '{ print $NF }') || true
  SCREEN_URL="http://localhost:${port:-8080}"
fi

# Monitors as "name x" lines; empty when the compositor can't be asked (then one image of everything).
monitors() {
  if [[ -n "$output" ]]; then
    echo "$output 0"
  elif command -v mmsg >/dev/null && mmsg get all-monitors >/dev/null 2>&1; then
    mmsg get all-monitors | jq -r '.monitors[] | "\(.name) \(.x)"'
  elif command -v swaymsg >/dev/null && swaymsg -t get_outputs >/dev/null 2>&1; then
    swaymsg -t get_outputs -r | jq -r '.[] | select(.active) | "\(.name) \(.rect.x)"'
  fi
}

dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT

# Writes one JPEG per monitor into $dir, named "<monitor>@<x>.jpg".
if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
  command -v grim >/dev/null || die "grim not found (needed on Wayland)"
  capture() {
    local list name x
    list=$(monitors)
    if [[ -z "$list" ]]; then
      grim -t jpeg -q 75 "$dir/screen@0.jpg"
      return
    fi
    while read -r name x; do
      grim -o "$name" -t jpeg -q 75 "$dir/$name@$x.jpg"
    done <<< "$list"
  }
elif [[ -n "${DISPLAY:-}" ]]; then
  command -v import >/dev/null || die "ImageMagick's import not found (needed on X11)"
  capture() { import -silent -window root -quality 75 "jpeg:$dir/screen@0.jpg"; }
else
  die "no graphical session found (WAYLAND_DISPLAY and DISPLAY are empty)"
fi

send() {
  local args=() f
  for f in "$dir"/*.jpg; do
    args+=(-F "shot=@$f;type=image/jpeg")
  done
  curl -fsS -m 20 -o /dev/null "${args[@]}" "$SCREEN_URL/api/screen"
}

info "screen helper for $SCREEN_URL (monitors: $(monitors | awk '{ print $1 }' | xargs echo || true)); Ctrl+C stops"
down=false
while true; do
  if ! answer=$(curl -fsS -m 75 "$SCREEN_URL/api/screen/next?wait=60" 2>/dev/null); then
    $down || warn "can't reach the app at $SCREEN_URL; retrying every 10 s"
    down=true
    sleep 10
    continue
  fi
  $down && info "connected again"
  down=false
  [[ "$answer" == capture ]] || continue
  rm -f "$dir"/*.jpg
  if capture && send; then
    :
  else
    warn "screenshot failed; retrying on the next request"
  fi
done
