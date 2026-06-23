#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
ROOT="${SCRIPT_DIR:h}"
RUNTIME="$HOME/Library/Application Support/poly5m-paper-trader"
mkdir -p "$RUNTIME" "$RUNTIME/outputs/paper_trader" "$RUNTIME/work/pycache" "$HOME/Library/LaunchAgents"

cp -R "$ROOT/src" "$RUNTIME/"
cp -R "$ROOT/scripts" "$RUNTIME/"
cp -R "$ROOT/config" "$RUNTIME/"
cp "$ROOT/pyproject.toml" "$RUNTIME/"
chmod +x "$RUNTIME/scripts/"*.sh "$RUNTIME/scripts/"*.py

cat > "$HOME/Library/LaunchAgents/com.poly5m.paperbot.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.poly5m.paperbot</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>POLY5M_ROOT</key>
    <string>$RUNTIME</string>
    <key>PYTHONPATH</key>
    <string>$RUNTIME/src</string>
    <key>PYTHONPYCACHEPREFIX</key>
    <string>$RUNTIME/work/pycache</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>$RUNTIME/scripts/run_paper_bot.py</string>
    <string>--config</string>
    <string>$RUNTIME/config/paper_btc_5m.json</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$RUNTIME/outputs/paper_trader/launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>$RUNTIME/outputs/paper_trader/launchd.err.log</string>
</dict>
</plist>
EOF

cat > "$HOME/Library/LaunchAgents/com.poly5m.maintenance.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.poly5m.maintenance</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>POLY5M_ROOT</key>
    <string>$RUNTIME</string>
    <key>PYTHONPATH</key>
    <string>$RUNTIME/src</string>
    <key>PYTHONPYCACHEPREFIX</key>
    <string>$RUNTIME/work/pycache</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>$RUNTIME/scripts/run_daily_maintenance.py</string>
    <string>--output-dir</string>
    <string>$RUNTIME/outputs/paper_trader</string>
    <string>--report-dir</string>
    <string>$RUNTIME/outputs/paper_trader/reports</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>9</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>$RUNTIME/outputs/paper_trader/maintenance.out.log</string>
  <key>StandardErrorPath</key>
  <string>$RUNTIME/outputs/paper_trader/maintenance.err.log</string>
</dict>
</plist>
EOF

launchctl unload "$HOME/Library/LaunchAgents/com.poly5m.paperbot.plist" 2>/dev/null || true
launchctl unload "$HOME/Library/LaunchAgents/com.poly5m.maintenance.plist" 2>/dev/null || true
launchctl load "$HOME/Library/LaunchAgents/com.poly5m.paperbot.plist"
launchctl load "$HOME/Library/LaunchAgents/com.poly5m.maintenance.plist"

echo "Installed and started:"
echo "  com.poly5m.paperbot"
echo "  com.poly5m.maintenance"
echo
echo "Runtime directory:"
echo "  $RUNTIME"
echo
echo "Check status:"
echo "  launchctl list | grep com.poly5m"
echo
echo "Tail logs:"
echo "  tail -f \"$RUNTIME/outputs/paper_trader/launchd.err.log\""
