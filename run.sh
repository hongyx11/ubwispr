#!/usr/bin/env bash
# Launch the ubwispr dictation daemon.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
exec uv run python -m ubwispr "$@"
