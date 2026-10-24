#!/usr/bin/env bash
# Copyright (c) 2026 Eric Cooper.
# Install (or remove) the two health checks that run beside the monitor (see camera_mcp.watchdog):
#   <prefix>.watchdog  camera-watchdog offline, every CAMERA_MCP_WATCHDOG_EVERY seconds (default 300)
#   <prefix>.health    camera-watchdog midday, daily at CAMERA_MCP_HEALTH_TIME (default 12:00)
# Needs the mail settings (and SMS_ALERT_TO for texts) in the credentials file. macOS only.
#
#   deploy/install-watchdog.sh              install both (prefix: CAMERA_MCP_LABEL_PREFIX, default camera-mcp)
#   deploy/install-watchdog.sh --dry-run    print the generated plists
#   deploy/install-watchdog.sh --uninstall
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${CAMERA_MCP_DATA:-$REPO/data}"
TEMPLATE="$REPO/deploy/watchdog.plist.template"
PREFIX="${CAMERA_MCP_LABEL_PREFIX:-camera-mcp}"
PLIST_DIR="${CAMERA_MCP_PLIST_DIR:-$HOME/Library/LaunchAgents}"
BREW_BIN="$( { command -v brew >/dev/null && dirname "$(command -v brew)"; } || echo /opt/homebrew/bin )"
SVC_PATH="${CAMERA_MCP_PATH:-$BREW_BIN:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin}"
EVERY="${CAMERA_MCP_WATCHDOG_EVERY:-300}"
NOON="${CAMERA_MCP_HEALTH_TIME:-12:00}"

render() {  # label check schedule-xml
    sed -e "s|@@LABEL@@|$1|g" -e "s|@@REPO@@|$REPO|g" -e "s|@@DATA@@|$DATA|g" -e "s|@@PATH@@|$SVC_PATH|g" \
        -e "s|@@CHECK@@|$2|g" -e "s|@@SCHEDULE@@|$3|g" "$TEMPLATE"
}
h="${NOON%%:*}"; m="${NOON##*:}"
JOBS=(
    "$PREFIX.watchdog|offline|<key>StartInterval</key><integer>$((10#$EVERY))</integer>"
    "$PREFIX.health|midday|<key>StartCalendarInterval</key><dict><key>Hour</key><integer>$((10#$h))</integer><key>Minute</key><integer>$((10#$m))</integer></dict>"
)

for job in "${JOBS[@]}"; do
    IFS='|' read -r label check schedule <<<"$job"
    plist="$PLIST_DIR/$label.plist"
    case "${1:-}" in
        --dry-run) render "$label" "$check" "$schedule"; continue ;;
        --uninstall)
            launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
            rm -f "$plist"; echo "removed $label"; continue ;;
    esac
    [ "$(uname)" = "Darwin" ] || { echo "macOS only" >&2; exit 1; }
    [ -x "$REPO/.venv/bin/camera-watchdog" ] || { echo "camera-watchdog missing: pip install -e ." >&2; exit 1; }
    mkdir -p "$DATA" "$PLIST_DIR"
    render "$label" "$check" "$schedule" > "$plist"
    plutil -lint "$plist" >/dev/null
    launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$plist"
    echo "installed $label: camera-watchdog $check"
done
[ -n "${1:-}" ] || echo "  logs: $DATA/health.log ; try: $REPO/.venv/bin/camera-watchdog offline --dry-run"
