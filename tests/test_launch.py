import ast
import subprocess
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]

# Real-robot script: only ctu_crs, moves the hoop down puzzle A at x = 0.45
# (table top at z = 0.026, see test_contact.MAZE_POSE) and optionally into
# the wire
SCRIPT = '''
import sys
import numpy as np
from ctu_crs import CRS97

DESCENT = np.array([
    [0.200, -0.280, -1.976, 0.000, -0.886, 0.724],
    [0.200, -0.404, -2.008, 0.000, -0.729, 0.724],
    [0.200, -0.548, -2.013, 0.000, -0.580, 0.724],
    [0.200, -0.706, -1.991, 0.000, -0.445, 0.724],
])
robot = CRS97()
assert CRS97() is robot
robot.initialize()
for q in DESCENT:
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
if "--bump" in sys.argv:
    robot.move_to_q(DESCENT[-1] + [0.08, 0, 0, 0, 0, 0])
    robot.wait_for_motion_stop()
print("image", robot.grab_image().shape)
sys.exit(3)
'''


def launch(*args, cwd=ROOT):
    return subprocess.run([sys.executable, "-m", "crs_sim.launch", *map(str, args)],
                          capture_output=True, text=True, cwd=cwd, timeout=300)


@pytest.fixture(scope="module")
def puzzle_files():
    pytest.importorskip("trimesh")
    from crs_sim.puzzles import download_puzzles
    try:
        download_puzzles(names=["A", "base"])
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"puzzle CAD files not available: {e}")


@pytest.mark.parametrize("bump", [False, True])
def test_real_robot_script_runs_in_scene(tmp_path, puzzle_files, bump):
    script = tmp_path / "solution.py"
    script.write_text(SCRIPT)
    result = launch("--headless", "--speedup", "inf", "--puzzle", "A",
                    "--x", 0.45, "--table-z", 0.026, script,
                    *(["--bump"] if bump else []))
    out = result.stdout
    assert result.returncode == 3, result.stderr
    assert "image (2050, 2448, 3)" in out
    assert "puzzle A at x=0.450" in out
    assert f"{int(bump)} contact(s)" in out
    assert "hoop threaded down to" in out


def test_script_errors_still_give_verdict(tmp_path):
    script = tmp_path / "broken.py"
    script.write_text("from ctu_crs import CRS97\nCRS97()\nraise RuntimeError('boom')\n")
    result = launch("--headless", "--puzzle", "none", script)
    assert result.returncode != 0
    assert "Verdict" in result.stdout and "boom" in result.stderr


def test_unsimulated_ctu_crs_names_fail_clearly(tmp_path):
    script = tmp_path / "crs93.py"
    script.write_text("import ctu_crs\nctu_crs.CRS93()\n")
    result = launch("--headless", "--puzzle", "none", script)
    assert "only CRS97 is" in result.stderr


def test_solution_does_not_import_simulator():
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom)
                     else [])
            assert not any(n.split(".")[0] == "crs_sim" for n in names), path


def test_src_main_runs_in_simulator(puzzle_files):
    result = launch("--headless", "--speedup", "inf", "src/main.py")
    assert result.returncode == 0, result.stderr
    assert "Verdict" in result.stdout
