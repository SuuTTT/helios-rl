#!/usr/bin/env bash
# Archive the TD-MPC-Glass control plane and optionally upload to S3.
#
# Usage:
#   bash scripts/backup_control_plane.sh
#   S3_URI=s3://my-bucket/tdmpc-glass/control bash scripts/backup_control_plane.sh
#
# This intentionally backs up control data, not huge training checkpoints.

set -euo pipefail

REPO=${REPO:-/root/helios-rl}
STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-$REPO/exp/tdmpc_glass/control_backups}
mkdir -p "$OUT_DIR"

ARCHIVE="$OUT_DIR/tdmpc_glass_control_${STAMP}.tgz"
TMP_ARCHIVE="$OUT_DIR/.tdmpc_glass_control_${STAMP}.tgz.tmp"
LATEST="$OUT_DIR/tdmpc_glass_control_latest.tgz"
FILE_LIST="$OUT_DIR/.tdmpc_glass_control_${STAMP}.files"

cd "$REPO"

{
  find scripts/queues docs/tdmpc-glass exp/tdmpc_glass/logs \
    -type f 2>/dev/null
  find exp/tdmpc_glass/remote_mirror \
    -type f \( -name '*.csv' -o -name '*.json' -o -name '*.log' \) 2>/dev/null
} | sort -u > "$FILE_LIST"

tar -czf "$TMP_ARCHIVE" -T "$FILE_LIST"
mv "$TMP_ARCHIVE" "$ARCHIVE"
rm -f "$FILE_LIST"

ln -sf "$(basename "$ARCHIVE")" "$LATEST"

echo "wrote $ARCHIVE"
echo "latest $LATEST"

if [[ -n "${S3_URI:-}" ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "S3_URI is set but aws CLI is not installed" >&2
    exit 2
  fi
  aws s3 cp "$ARCHIVE" "$S3_URI/"
  aws s3 cp "$ARCHIVE" "$S3_URI/tdmpc_glass_control_latest.tgz"
  echo "uploaded to $S3_URI"
fi
