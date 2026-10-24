#!/usr/bin/env bash
# Copyright (c) 2026 Eric Cooper.
# Install (or remove) a LaunchAgent that runs one scheduled agent job weekly (camera-agent <job>
# --email; see camera_mcp.agent), local time. Needs the agent extra and ANTHROPIC_API_KEY and the
# SMTP settings in the credentials file. macOS only.
#
#   deploy/install-agent.sh <job>                  install; CAMERA_MCP_AGENT_DAY (0=Sunday) and
#                                                  CAMERA_MCP_AGENT_TIME (HH:MM), default Sunday 08:00
#   deploy/install-agent.sh --dry-run <job>        print the generated plist
#   deploy/install-agent.sh --uninstall <job>
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${CAMERA_MCP_DATA:-$REPO/data}"
TEMPLATE="$REPO/deploy/agent.plist.template"
BREW_BIN="$( { command -v brew >/dev/null && dirname "$(command -v brew)"; } || echo /opt/homebrew/bin )"
SVC_PATH="${CAMERA_MCP_PATH:-$BREW_BIN:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin}"
DAY="${CAMERA_MCP_AGENT_DAY:-0}"
TIME="${CAMERA_MCP_AGENT_TIME:-08:00}"

mode=install
case "${1:-}" in
    --dry-run) mode=dry; shift ;;
    --uninstall) mode=uninstall; shift ;;
esac
JOB="${1:?usage: install-agent.sh [--dry-run|--uninstall] <job>}"
LABEL="${CAMERA_MCP_LABEL_PREFIX:-camera-mcp}.agent.$JOB"
PLIST="${CAMERA_MCP_PLIST_DIR:-$HOME/Library/LaunchAgents}/$LABEL.plist"

render() {
    local h="${TIME%%:*}" m="${TIME##*:}"
    sed -e "s|@@LABEL@@|$LABEL|g" -e "s|@@REPO@@|$REPO|g" -e "s|@@DATA@@|$DATA|g" -e "s|@@PATH@@|$SVC_PATH|g" \
        -e "s|@@JOB@@|$JOB|g" -e "s|@@WEEKDAY@@|$((10#$DAY))|g" -e "s|@@HOUR@@|$((10#$h))|g" \
        -e "s|@@MINUTE@@|$((10#$m))|g" "$TEMPLATE"
}

case "$mode" in
    dry) render; exit 0 ;;
    uninstall)
        launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
        rm -f "$PLIST"; echo "removed $LABEL"; exit 0 ;;
esac

[ "$(uname)" = "Darwin" ] || { echo "macOS only" >&2; exit 1; }
[ -x "$REPO/.venv/bin/camera-agent" ] || { echo "camera-agent missing: pip install -e '.[agent]'" >&2; exit 1; }
mkdir -p "$DATA" "$(dirname "$PLIST")"
render > "$PLIST"
plutil -lint "$PLIST" >/dev/null
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL: $JOB every weekday $DAY (0=Sunday) at $TIME (local), by email"
echo "  test now:  $REPO/.venv/bin/camera-agent $JOB        (add --email to mail it)"
echo "  logs:      $DATA/agent.log ; reports in $DATA/agent/$JOB/"
