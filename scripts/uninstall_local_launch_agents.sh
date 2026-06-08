#!/usr/bin/env zsh
set -euo pipefail

launchctl unload "$HOME/Library/LaunchAgents/com.ethan.poly5m.paperbot.plist" 2>/dev/null || true
launchctl unload "$HOME/Library/LaunchAgents/com.ethan.poly5m.maintenance.plist" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/com.ethan.poly5m.paperbot.plist"
rm -f "$HOME/Library/LaunchAgents/com.ethan.poly5m.maintenance.plist"

echo "Uninstalled local Polymarket paper-trader LaunchAgents."
