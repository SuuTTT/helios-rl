#!/usr/bin/env bash
# Launch the Flask-based TD-MPC-Glass dashboard.
#
# Usage:
#   bash scripts/launch_web_dashboard.sh
#   (or for background mode:)
#   nohup setsid bash scripts/launch_web_dashboard.sh > /tmp/web_dashboard.log 2>&1 &
#
# Default URL: http://localhost:5055
set -u
REPO=${REPO:-/root/helios-rl}
cd "$REPO"
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export DASHBOARD_PORT=${DASHBOARD_PORT:-5055}
export DASHBOARD_HOST=${DASHBOARD_HOST:-0.0.0.0}
exec /root/venv/bin/python3 -u scripts/web_dashboard.py
