#!/usr/bin/env bash
# Copyright (c) 2026 Eric Cooper.
# Install (or remove) the brief LaunchAgent: runs deploy/morning-brief.sh on a schedule (local
# time) and delivers the wildlife brief (email if SMTP is set, else an iMessage recipient). macOS only.
#
#   deploy/install-brief.sh <imessage-recipient>   install; times from CAMERA_MCP_BRIEF_TIME[S]
#   deploy/install-brief.sh --dry-run <recipient>  print the generated plist
#   deploy/install-brief.sh --uninstall
#
# Schedule: CAMERA_MCP_BRIEF_TIMES is a comma-separated list of local HH:MM (a cadence, e.g.
#   "02:00,06:00,10:00,14:00,18:00,22:00" for every 4 hours); CAMERA_MCP_BRIEF_TIME sets a single
#   time (default 07:00) when TIMES is unset. CAMERA_MCP_BRIEF_SINCE is how far back each brief looks
#   (set it to the cadence, e.g. 4h, so the windows tile without gaps or overlap).
set -euo pipefail

LABEL="${CAMERA_MCP_BRIEF_LABEL:-nu.cooper.camera-mcp.brief}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${CAMERA_MCP_DATA:-$REPO/data}"
PLIST="${CAMERA_MCP_PLIST_DIR:-$HOME/Library/LaunchAgents}/$LABEL.plist"
TEMPLATE="$REPO/deploy/morning-brief.plist.template"
BREW_BIN="$( { command -v brew >/dev/null && dirname "$(command -v brew)"; } || echo /opt/homebrew/bin )"
SVC_PATH="${CAMERA_MCP_PATH:-$BREW_BIN:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin}"
TIMES="${CAMERA_MCP_BRIEF_TIMES:-${CAMERA_MCP_BRIEF_TIME:-07:00}}"
SINCE="${CAMERA_MCP_BRIEF_SINCE:-14h}"  # how far back each brief looks

# The StartCalendarInterval value: one <dict> per HH:MM in TIMES, wrapped in <array> if there's
# more than one (launchd fires the agent at each). XML indentation is cosmetic; plutil validates it.
build_calendar() {
    local dicts="" t h m n=0 oldIFS="$IFS"
    IFS=','
    for t in $TIMES; do
        t="${t// /}"; h="${t%%:*}"; m="${t##*:}"
        dicts+="<dict><key>Hour</key><integer>$((10#$h))</integer><key>Minute</key><integer>$((10#$m))</integer></dict>"
        n=$((n + 1))
    done
    IFS="$oldIFS"
    if [ "$n" -gt 1 ]; then printf '<array>%s</array>' "$dicts"; else printf '%s' "$dicts"; fi
}

render() {
    local cal; cal="$(build_calendar)"
    sed -e "s|@@LABEL@@|$LABEL|g" -e "s|@@REPO@@|$REPO|g" -e "s|@@DATA@@|$DATA|g" \
        -e "s|@@PATH@@|$SVC_PATH|g" -e "s|@@RECIPIENT@@|${RECIPIENT:-}|g" -e "s|@@SINCE@@|$SINCE|g" "$TEMPLATE" \
        | awk -v cal="$cal" '{ if (index($0, "@@CALENDAR@@")) { sub(/@@CALENDAR@@/, cal) } print }'
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
echo "installed $LABEL: brief (last $SINCE) to ${RECIPIENT:-email} at $TIMES (local)"
echo "  test now:  deploy/morning-brief.sh '$RECIPIENT'"
echo "  logs:      $DATA/brief.log ; briefs in $DATA/briefs/"
