#!/usr/bin/env bash
# Start an interactive Python console with a simulated robot ready as `robot`.
# Extra arguments are passed to python, e.g. ./start.sh examples/demo.py
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "No venv found, run ./setup_venv.sh first." >&2
    exit 1
fi
unset PYTHONPATH

if [ $# -gt 0 ]; then
    exec "$VENV_DIR/bin/python" "$@"
fi
exec "$VENV_DIR/bin/python" -i -c '
import numpy as np
from crs_sim import CRS97
np.set_printoptions(precision=3, suppress=True)
robot = CRS97()
robot.initialize(home=False)
print("`robot` is a simulated CRS97, e.g. robot.move_to_q(robot.q_home + [0.3, 0, 0, 0, 0, 0])")
'
