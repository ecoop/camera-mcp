#!/usr/bin/env bash
# Copyright (c) 2026 Eric Cooper.
# Install (or remove) the camera monitor as a launchd LaunchAgent, so it starts at login
# and restarts if it crashes. Run from a checkout that already has a .venv and cameras.yaml
# (see setup.sh). macOS only.
#
#   deploy/install-service.sh            install and start the service
#   deploy/install-service.sh --dry-run  print the generated plist and exit (no changes)
#   deploy/install-service.sh --uninstall stop and remove the service
set -euo pipefail

LABEL="${CAMERA_MCP_LABEL:-nu.cooper.camera-mcp}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${CAMERA_MCP_DATA:-$REPO/data}"
PLIST="${CAMERA_MCP_PLIST_DIR:-$HOME/Library/LaunchAgents}/$LABEL.plist"
TEMPLATE="$REPO/deploy/camera-monitor.plist.template"
BREW_BIN="$( { command -v brew >/dev/null && dirname "$(command -v brew)"; } || echo /opt/homebrew/bin )"
SVC_PATH="${CAMERA_MCP_PATH:-$BREW_BIN:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin}"

render() {  # template -> stdout, placeholders filled in
    sed -e "s|@@LABEL@@|$LABEL|g" -e "s|@@REPO@@|$REPO|g" -e "s|@@DATA@@|$DATA|g" \
        -e "s|@@PATH@@|$SVC_PATH|g" "$TEMPLATE"
}

case "${1:-install}" in
    --dry-run)
        render
        exit 0 ;;
    --uninstall)
        launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || launchctl unload "$PLIST" 2>/dev/null || true
        rm -f "$PLIST"
        echo "removed $LABEL (data in $DATA is kept)"
        exit 0 ;;
    install) ;;
    *) echo "usage: $0 [--dry-run|--uninstall]" >&2; exit 2 ;;
esac

[ "$(uname)" = "Darwin" ] || { echo "install-service.sh: macOS only (launchd)" >&2; exit 1; }
[ -x "$REPO/.venv/bin/camera-monitor" ] || { echo "no .venv yet: run deploy/setup.sh first" >&2; exit 1; }
[ -f "$REPO/cameras.yaml" ] || { echo "no cameras.yaml yet: copy cameras.example.yaml and edit it" >&2; exit 1; }

mkdir -p "$DATA" "$(dirname "$PLIST")"
render > "$PLIST"
plutil -lint "$PLIST" >/dev/null

# Reload cleanly whether or not it is already loaded.
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl enable "gui/$(id -u)/$LABEL"
echo "installed and started $LABEL"
echo "  logs:   tail -f $DATA/monitor.log"
echo "  status: launchctl print gui/$(id -u)/$LABEL | grep state"
echo "  stop:   deploy/install-service.sh --uninstall"
