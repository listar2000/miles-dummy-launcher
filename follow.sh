#!/bin/sh
# Feed the viewer's log tail from the app's start until the app stops. `modal app logs` with no --tail follows.
A=$1; R=$2; cd "$(dirname "$0")"
while python -m modal app list --json 2>/dev/null | grep -A3 "$A" | grep -q ephemeral; do
  python -m modal app logs "$A" 2>&1 | sed -u -E 's/\x1b\[[0-9;]*m//g; s/(token[=: ]+)[A-Za-z0-9_-]+/\1***/I' >> "logs/$R-follow.log"
  sleep 5
done
