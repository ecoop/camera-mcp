#!/usr/bin/env bash
# Copyright (c) 2026 Eric Cooper.
# Install (or remove) the morning-brief LaunchAgent: runs deploy/morning-brief.sh once each
# morning (local time) and texts the wildlife brief to an iMessage recipient. macOS only.
#
#   deploy/install-brief.sh <imessage-recipient>   install; time from CAMERA_MCP_BRIEF_TIME (default 07:00)
#   deploy/install-brief.sh --dry-run <recipient>  print the generated plist
#   deploy/install-brief.sh --uninstall
set -euo pipefail

LABEL="${CAMERA_MCP_BRIEF_LABEL:-nu.cooper.camera-mcp.brief}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${CAMERA_MCP_DATA:-$REPO/data}"
PLIST="${CAMERA_MCP_PLIST_DIR:-$HOME/Library/LaunchAgents}/$LABEL.plist"
TEMPLATE="$REPO/deploy/morning-brief.plist.template"
BREW_BIN="$( { command -v brew >/dev/null && dirname "$(command -v brew)"; } || echo /opt/homebrew/bin )"
SVC_PATH="${CAMERA_MCP_PATH:-$BREW_BIN:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin}"
TIME="${CAMERA_MCP_BRIEF_TIME:-07:00}"
HOUR="${TIME%%:*}"; MINUTE="${TIME##*:}"

render() {
    sed -e "s|@@LABEL@@|$LABEL|g" -e "s|@@REPO@@|$REPO|g" -e "s|@@DATA@@|$DATA|g" \
        -e "s|@@PATH@@|$SVC_PATH|g" -e "s|@@HOUR@@|$((10#$HOUR))|g" -e "s|@@MINUTE@@|$((10#$MINUTE))|g" \
        -e "s|@@RECIPIENT@@|${RECIPIENT:-}|g" "$TEMPLATE"
}

case "${1:-}" in
    --uninstall)
        launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
        rm -f "$PLIST"; echo "removed $LABEL"; exit 0 ;;
    --dry-run) RECIPIENT="${2:-}"; render; exit 0 ;;
    "") RECIPIENT="" ;;  # no iMessage recipient: the brief uses email if SMTP is set, else writes the file only
    *) RECIPIENT="$1" ;;
esac

[ "$(uname)" = "Darwin" ] || { echo "macOS only" >&2; exit 1; }
[ -x "$REPO/deploy/morning-brief.sh" ] || { echo "morning-brief.sh missing" >&2; exit 1; }
mkdir -p "$DATA" "$(dirname "$PLIST")"
render > "$PLIST"
plutil -lint "$PLIST" >/dev/null
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL: brief to $RECIPIENT daily at $TIME (local)"
echo "  test now:  deploy/morning-brief.sh '$RECIPIENT'"
echo "  logs:      $DATA/brief.log ; briefs in $DATA/briefs/"
