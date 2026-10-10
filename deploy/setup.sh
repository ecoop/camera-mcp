#!/usr/bin/env bash
# Copyright (c) 2026 Eric Cooper.
# Set up camera-mcp on this machine: check the tools, build the venv with SpeciesNet, and
# run the probe and the benchmark. Safe to re-run. Run it from a checkout of the repo.
#
#   deploy/setup.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
note() { printf '\n== %s\n' "$1"; }

note "tools"
missing=()
for tool in ffmpeg ffprobe uv; do
    if command -v "$tool" >/dev/null; then echo "  $tool: $(command -v "$tool")"; else missing+=("$tool"); fi
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "  missing: ${missing[*]}"
    echo "  install with: brew install ffmpeg uv"
    exit 1
fi

note "virtualenv with SpeciesNet (camera-mcp[detect])"
# SpeciesNet needs Python < 3.14; pin it, or uv picks a newer one and silently drops SpeciesNet
# (its dependency is gated on python_version < '3.14').
uv venv -p "${CAMERA_MCP_PYTHON:-3.13}" .venv
uv pip install --python .venv/bin/python -e '.[detect]'
.venv/bin/python -c "from camera_mcp.detect import available; import sys; sys.exit(0 if available() else 1)" \
    && echo "  SpeciesNet: installed" || { echo "  SpeciesNet failed to import"; exit 1; }

note "camera login (macOS Keychain item 'camera-mcp')"
if security find-generic-password -s camera-mcp >/dev/null 2>&1; then
    echo "  found"
else
    echo "  NOT set. Store it (you will be prompted for the password):"
    echo "    security add-generic-password -U -s camera-mcp -a <camera user> -w"
fi

note "camera config"
if [ -f cameras.yaml ]; then
    echo "  cameras.yaml present"
else
    echo "  NO cameras.yaml. Copy and edit it:  cp cameras.example.yaml cameras.yaml"
fi

if [ -f cameras.yaml ] && security find-generic-password -s camera-mcp >/dev/null 2>&1; then
    note "probe (reachability and stream resolutions)"
    .venv/bin/python -m camera_mcp.probe || true

    note "benchmark (how many cameras this machine can keep up with)"
    .venv/bin/python -W ignore -m camera_mcp.bench --camera "$(.venv/bin/python - <<'PY'
from camera_mcp.config import load_cameras
print(next(iter(load_cameras("cameras.yaml", require_login=False))))
PY
)" || true
fi

note "next"
echo "  install the always-on service:  deploy/install-service.sh"
