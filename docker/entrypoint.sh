#!/usr/bin/env bash
set -euo pipefail

cd /workspace
source /workspace/.venv/bin/activate

exec "$@"
