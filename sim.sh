#!/usr/bin/env bash
# Run real-robot code in the simulator: ./sim.sh [scene options] script.py [script args]
# e.g. ./sim.sh --puzzle C --seed 7 src/main.py --calib src/calibration/sim.npz
# See ./sim.sh --help for the scene options.
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "No venv found, run ./setup_venv.sh first." >&2
    exit 1
fi
unset PYTHONPATH
exec "$VENV_DIR/bin/python" -m crs_sim.launch "$@"
