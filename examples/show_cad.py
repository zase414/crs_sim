"""
Show CAD files (STEP/STP, STL, OBJ, ...) in the robot workspace.

    ./start.sh examples/show_cad.py                     # all ROB1 puzzles
    ./start.sh examples/show_cad.py part.step --x 0.4 --yaw 1.57
    ./start.sh examples/show_cad.py bracket.stl --scale 0.001   # STL in mm

STEP files are converted to meters automatically, other formats carry no
units, use --scale. Needs the 'step' extra (installed by setup_venv.sh).
"""
import argparse

import numpy as np

from crs_sim import CRS97
from crs_sim.puzzles import show_puzzle
from crs_sim.transforms import pose

COLORS = [0xffa500, 0x2e86de, 0x27ae60, 0xc0392b, 0x8e44ad]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("files", nargs="*", help="CAD files, ROB1 puzzles if empty")
    parser.add_argument("--x", type=float, default=0.45, help="position [m]")
    parser.add_argument("--y", type=float, default=0.0, help="position [m]")
    parser.add_argument("--z", type=float, default=0.0, help="position [m]")
    parser.add_argument("--yaw", type=float, default=0.0, help="rotation [rad]")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="vertex scale, 0.001 for mm (not needed for STEP)")
    args = parser.parse_args()

    robot = CRS97()
    robot.initialize(home=False)

    if args.files:
        for i, f in enumerate(args.files):
            # several files are placed side by side along y
            y = args.y + 0.15 * (i - (len(args.files) - 1) / 2)
            robot.viz.show_mesh(f"cad/{i}", f, pose([args.x, y, args.z], args.yaw),
                                color=COLORS[i % len(COLORS)], scale=args.scale)
            print(f"Showing {f}")
    else:
        # The five puzzles on an arc in front of the robot
        for i, name in enumerate("ABCDE"):
            angle = np.deg2rad(-60 + 30 * i)
            xy = 0.5 * np.array([np.cos(angle), np.sin(angle)])
            show_puzzle(robot.viz, name, pose([*xy, args.z], angle), path=True,
                        color=COLORS[i])
            print(f"Puzzle {name} at {np.round(xy, 3)}")

    input("Press enter to exit.")


if __name__ == "__main__":
    main()
