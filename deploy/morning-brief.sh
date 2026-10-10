#!/usr/bin/env bash
# Copyright (c) 2026 Eric Cooper.
# Generate the wildlife brief and deliver it: always write a dated file (reliable), and if an
# iMessage recipient is given, text it (then a montage of the night's notable visitors). Run by
# the morning-brief LaunchAgent, or by hand.
#
#   deploy/morning-brief.sh [imessage-recipient]
# The recipient is a phone number (+15551234567) or an iMessage email. Omit to only write the file.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${CAMERA_MCP_DATA:-$REPO/data}"
RECIPIENT="${1:-${CAMERA_MCP_IMESSAGE:-}}"
cd "$REPO"
mkdir -p "$DATA/briefs"
since="${CAMERA_MCP_BRIEF_SINCE:-14h}"
py="$REPO/.venv/bin/python"

# Prefer email when SMTP is configured (the brief writes the file + montage and sends it itself).
if "$py" -c "import sys; from camera_mcp.mailer import email_settings; sys.exit(0 if email_settings() else 1)" 2>/dev/null; then
    "$py" -W ignore -m camera_mcp.brief --since "$since" --email >>"$DATA/brief.log" 2>&1
    rc=$?
    echo "$(date '+%F %T') email rc=$rc" >> "$DATA/brief.log"
    exit $rc
fi

# Otherwise: write the dated file, and optionally iMessage it.
montage="$DATA/briefs/$(date +%F).jpg"
rm -f "$montage"
brief="$("$py" -W ignore -m camera_mcp.brief \
    --since "$since" --montage "$montage" 2>>"$DATA/brief.log")"
if [ -z "$brief" ]; then
    echo "$(date '+%F %T') brief was empty (see brief.log)" >> "$DATA/brief.log"
    exit 1
fi

printf '%s\n' "$brief" > "$DATA/briefs/$(date +%F).txt"

[ -z "$RECIPIENT" ] && exit 0

# Send the message, then the montage image if there is one. The text is passed as an argument
# (not interpolated into the script) so its newlines and emoji are safe. The first send triggers
# a one-time "control Messages" Automation prompt that must be approved.
osascript - "$RECIPIENT" "$brief" <<'APPLESCRIPT'
on run {recipient, body}
    tell application "Messages"
        send body to participant recipient of (1st account whose service type = iMessage)
    end tell
end run
APPLESCRIPT
rc=$?

if [ "$rc" -eq 0 ] && [ -f "$montage" ]; then
    osascript - "$RECIPIENT" "$montage" <<'APPLESCRIPT' || true
on run {recipient, imagePath}
    tell application "Messages"
        send (POSIX file imagePath) to participant recipient of (1st account whose service type = iMessage)
    end tell
end run
APPLESCRIPT
fi
echo "$(date '+%F %T') imessage rc=$rc montage=$([ -f "$montage" ] && echo yes || echo no)" >> "$DATA/brief.log"
exit $rc
