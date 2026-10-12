#!/usr/bin/env bash
# Copyright (c) 2026 Eric Cooper.
# Install (or remove) the read-only web dashboard (camera_mcp.web) as a launchd LaunchAgent, so it
# starts at login and restarts if it stops. Run from a checkout with a .venv and cameras.yaml.
# Needs the web extra: pip install '.[web]' (uvicorn). macOS only.
#
#   deploy/install-webapp.sh            install and start (HOST/PORT from CAMERA_MCP_WEB_HOST/PORT)
#   deploy/install-webapp.sh --dry-run  print the generated plist and exit
#   deploy/install-webapp.sh --uninstall  stop and remove
set -euo pipefail

LABEL="${CAMERA_MCP_WEB_LABEL:-nu.cooper.camera-mcp.web}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${CAMERA_MCP_DATA:-$REPO/data}"
PLIST="${CAMERA_MCP_PLIST_DIR:-$HOME/Library/LaunchAgents}/$LABEL.plist"
TEMPLATE="$REPO/deploy/webapp.plist.template"
BREW_BIN="$( { command -v brew >/dev/null && dirname "$(command -v brew)"; } || echo /opt/homebrew/bin )"
SVC_PATH="${CAMERA_MCP_PATH:-$BREW_BIN:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin}"
HOST="${CAMERA_MCP_WEB_HOST:-0.0.0.0}"
PORT="${CAMERA_MCP_WEB_PORT:-8080}"

render() {
    sed -e "s|@@LABEL@@|$LABEL|g" -e "s|@@REPO@@|$REPO|g" -e "s|@@DATA@@|$DATA|g" \
        -e "s|@@PATH@@|$SVC_PATH|g" -e "s|@@HOST@@|$HOST|g" -e "s|@@PORT@@|$PORT|g" "$TEMPLATE"
}

case "${1:-install}" in
    --dry-run) render; exit 0 ;;
    --uninstall)
        launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
        rm -f "$PLIST"; echo "removed $LABEL"; exit 0 ;;
    install) ;;
    *) echo "usage: $0 [--dry-run|--uninstall]" >&2; exit 2 ;;
esac

[ "$(uname)" = "Darwin" ] || { echo "install-webapp.sh: macOS only" >&2; exit 1; }
[ -x "$REPO/.venv/bin/python" ] || { echo "no .venv yet: run deploy/setup.sh first" >&2; exit 1; }
"$REPO/.venv/bin/python" -c "import uvicorn" 2>/dev/null || { echo "web extra missing: pip install '.[web]'" >&2; exit 1; }
[ -f "$REPO/cameras.yaml" ] || { echo "no cameras.yaml yet" >&2; exit 1; }

mkdir -p "$DATA" "$(dirname "$PLIST")"
render > "$PLIST"
plutil -lint "$PLIST" >/dev/null
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl enable "gui/$(id -u)/$LABEL"
echo "installed and started $LABEL on $HOST:$PORT"
echo "  open:   http://<studio-ip>:$PORT  (allow Python through the macOS firewall on first run)"
echo "  logs:   tail -f $DATA/web.log"
echo "  stop:   deploy/install-webapp.sh --uninstall"
