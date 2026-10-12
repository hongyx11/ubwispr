#!/usr/bin/env bash
# Launch the ubwispr dictation daemon.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# Discover Conda without depending on interactive-shell initialization.
task_conda_exe="${UBWISPR_CONDA_EXE:-${CONDA_EXE:-}}"
if [ -z "$task_conda_exe" ]; then
    if [ -x "$HOME/miniconda3/bin/conda" ]; then
        task_conda_exe="$HOME/miniconda3/bin/conda"
    elif command -v conda >/dev/null 2>&1; then
        task_conda_exe="$(command -v conda)"
    else
        printf '%s\n' 'Miniconda was not found. Set UBWISPR_CONDA_EXE to its bin/conda executable.' >&2
        exit 127
    fi
fi
task_conda_root="$("$task_conda_exe" info --base)"
source "$task_conda_root/etc/profile.d/conda.sh"
task_conda_env="${UBWISPR_CONDA_ENV:-ubwispr}"
if ! conda activate "$task_conda_env"; then
    printf '%s\n' 'Create the app environment first: conda env create -f environment.yml' >&2
    exit 1
fi
exec python -m ubwispr "$@"
