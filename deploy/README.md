# Running the monitor on the Mac Studio

_Last updated: 2026-10-24_

The monitor identifies animals with SpeciesNet, which needs a GPU to keep up. On Apple Silicon
(the Mac Studio M2) PyTorch uses the GPU (MPS); on the Intel MacBook it is CPU-only and far too
slow (see the benchmark below). So the monitor's long-term home is the Studio.

Nothing here runs the monitor on a schedule until you run `install-service.sh`.

## Set up (on the Studio)

```bash
brew install ffmpeg uv                                   # if not already present
git clone https://github.com/ecoop/camera-mcp ~/projects/camera-mcp
cd ~/projects/camera-mcp
cp cameras.example.yaml cameras.yaml                     # then edit: your camera IPs and site
security add-generic-password -U -s camera-mcp -a <camera user> -w   # prompts for the password
deploy/setup.sh                                          # venv (Python 3.13) + SpeciesNet, probe, benchmark
```

`setup.sh` is safe to re-run. It checks the tools, builds `.venv` with `camera-mcp[detect]`,
confirms the Keychain login and `cameras.yaml`, and runs the probe and the benchmark.

## Check the speed first

```bash
.venv/bin/python -m camera_mcp.bench --camera backyard --camera east_deck
```

It reports the device (expect `mps` on the Studio) and how many busy cameras the machine can keep
up with. Baseline on the 2020 Intel MacBook Pro (CPU): ~7 s/frame whole, ~20 s tiled — about 0.14
cameras. The Studio should be far higher; the number decides whether tiled wide-view cameras keep
up or still want motion-guided crops.

## Run it always-on

```bash
deploy/install-service.sh            # install + start the LaunchAgent; starts again at login
deploy/install-service.sh --dry-run  # just print the generated plist
deploy/install-service.sh --uninstall
```

This installs a launchd **LaunchAgent** (`~/Library/LaunchAgents/camera-mcp.plist`). It
runs in your login session, which is required: the camera password lives in the **login
Keychain**, and only a logged-in session can read it. So:

- The Studio must be **logged in** (enable automatic login if it runs headless).
- It must **not sleep**. The service wraps the monitor in `caffeinate -s`, and also set
  System Settings > Energy > "Prevent automatic sleeping when the display is off".

Logs go to `data/monitor.log`. Status: `launchctl print gui/$(id -u)/camera-mcp`.

## Use the tools from the MacBook

Point Claude Code's MCP server at the Studio over SSH (one time, needs key-based SSH). The plugin
does this for you: install it and give the Studio as its monitor host (see "Use from Claude Code"
in the main README). By hand:

```bash
claude mcp remove cameras    # if the local one was added earlier
claude mcp add cameras -- ssh <studio> ~/projects/camera-mcp/.venv/bin/camera-mcp \
    --config ~/projects/camera-mcp/cameras.yaml --data-dir ~/projects/camera-mcp/data
```

`cameras.yaml` is git-ignored, so it does not come from the clone — copy or recreate it on the
Studio. Put the Studio on the same LAN as the cameras, and reserve its address in the router so
the SSH line does not break if it changes.

History tools work over SSH, but `snapshot` needs the camera login, and the Keychain is locked in
an SSH session. For remote `snapshot`, put the login in a file the SSH-launched server can read
(readable only by you):

```bash
mkdir -p ~/.config/camera-mcp
umask 077; { echo "CAMERA_USER=<camera user>"; printf "CAMERA_PASS="; read -rs p; echo "$p"; } \
    > ~/.config/camera-mcp/credentials; unset p
```

The monitor keeps using the Keychain (it runs in the desktop session); only the SSH server falls
back to the file.

## Morning brief by email

The brief can be emailed (montage inline) instead of iMessaged. Add the mail settings to the same
credentials file (real values stay on the host, never in git):

```
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_FROM=camera@example.com
SMTP_TO=you@example.com
SMTP_USER=camera@example.com
SMTP_PASS=...
```

Then install the 7am agent (no iMessage recipient needed — it uses email when SMTP is set):

```bash
CAMERA_MCP_BRIEF_TIME=07:00 deploy/install-brief.sh
```

Test it now: `deploy/morning-brief.sh` (writes `data/briefs/<date>.txt` and emails it). Port 587
uses STARTTLS, 465 uses SSL.

## Weekly digest by email

A scheduled agent (see "Scheduled agents" in the main README) needs the agent extra, the API key and
the mail settings above:

```bash
.venv/bin/pip install -e '.[agent]'
.venv/bin/camera-agent weekly-digest            # try it once; add --email to mail it
deploy/install-agent.sh weekly-digest           # every Sunday 08:00 (CAMERA_MCP_AGENT_DAY / _TIME)
```

Logs go to `data/agent.log`. `deploy/install-agent.sh --uninstall weekly-digest` removes it.

## Health checks: offline alerts and the midday report

Two checks run beside the monitor (not inside it, so they still report if the monitor stops), with
the mail settings above and `SMS_ALERT_TO` for texts:

- **offline**, every 5 minutes: a text when a camera has sent no video for 20 minutes inside its
  watch hours (paused hours don't count), and one more when it's back. If every camera that should
  be watched is silent, it says to check the monitor instead.
- **midday**, at noon: an email on whether the detector kept up over the last 4 hours.

```bash
.venv/bin/camera-watchdog offline --dry-run     # what it would send right now (sends nothing)
deploy/install-watchdog.sh                      # install both (CAMERA_MCP_LABEL_PREFIX for the names)
```

Logs go to `data/health.log`; `deploy/install-watchdog.sh --uninstall` removes both.
