#!/bin/bash
# Install the Yarnotron supervisor as a macOS LaunchAgent so it starts at login
# and restarts if it dies. Run on the Mac:  bash scripts/install_supervisor_launchd.sh
# Remove with:                              bash scripts/install_supervisor_launchd.sh --uninstall
set -euo pipefail

LABEL="com.yarnotron.supervisor"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
STATE="$HOME/.yarnotron-supervisor"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DOMAIN="gui/$(id -u)"

if [[ "${1:-}" == "--uninstall" ]]; then
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed $LABEL. Your config and logs in $STATE were left alone."
  exit 0
fi

mkdir -p "$STATE/logs" "$HOME/Library/LaunchAgents"
if [[ ! -f "$STATE/supervisor.toml" ]]; then
  cp "$REPO/supervisor.example.toml" "$STATE/supervisor.toml"
  echo "Created $STATE/supervisor.toml from the example. Edit it (paths, token), then re-run this script."
  exit 0
fi

PYTHON="$(command -v python3)"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/caffeinate</string><string>-i</string>
    <string>$PYTHON</string>
    <string>$REPO/scripts/yarnotron_supervisor.py</string>
    <string>--config</string><string>$STATE/supervisor.toml</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$STATE/logs/supervisor.log</string>
  <key>StandardErrorPath</key><string>$STATE/logs/supervisor.log</string>
</dict></plist>
PLIST

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"
launchctl kickstart -k "$DOMAIN/$LABEL"
sleep 2
if curl -fsS "http://127.0.0.1:8990/v1/health" >/dev/null; then
  echo "Supervisor is running (port 8990). Logs: $STATE/logs/supervisor.log"
else
  echo "Installed, but port 8990 is not answering yet. Check $STATE/logs/supervisor.log" >&2
  exit 1
fi
