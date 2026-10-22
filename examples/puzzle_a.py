"""
Puzzle A with fixed, hand-picked joint configurations: show the puzzle in the
workspace and move the hoop down the wire and back.

    ./start.sh examples/puzzle_a.py

This is NOT a solver: the joint values below were picked for this one
placement of the puzzle. It shows how to put a puzzle into the scene (drawn,
seen by the camera and checked for contacts), how to get the hoop pose from
the joint configuration (fk @ HOOP_IN_FLANGE) and how to execute motions with
the ctu_crs interface (move_to_q / wait_for_motion_stop), which work the same
way on the real robot.
"""
import numpy as np

from crs_sim import CRS97
from crs_sim.puzzles import BASE_PLATE_THICKNESS, HOOP_IN_FLANGE, Puzzle, maze_path
from crs_sim.transforms import densify, pose, translation

# Puzzle A placed 0.45 m in front of the robot, base plate top at z = 0.031 m
MAZE_POSE = pose([0.45, 0.0, 0.031], yaw=0.0)

# Hand-picked joint configurations [rad], from above the wire down to the base
KEYFRAMES = np.array([
    [0.200, -0.105, -1.835, 0.000, -1.202, 0.724],  # above the wire
    [0.200, -0.280, -1.976, 0.000, -0.886, 0.724],  # hoop 10 cm above the tip
    [0.200, -0.404, -2.008, 0.000, -0.729, 0.724],  # hoop at the wire tip
    [0.200, -0.548, -2.013, 0.000, -0.580, 0.724],
    [0.200, -0.706, -1.991, 0.000, -0.445, 0.724],
    [0.200, -0.873, -1.941, 0.000, -0.327, 0.724],
    [0.200, -1.037, -1.868, 0.000, -0.237, 0.724],  # hoop 2 cm above the base
])


def hoop_center(robot, q) -> np.ndarray:
    """Hoop center position in the robot base frame."""
    return (robot.fk(q) @ HOOP_IN_FLANGE)[:3, 3]


def main():
    robot = CRS97(speedup=2.0)
    robot.initialize()

    # Scene: table under the base plate, the puzzle and its wire center line
    table_top = MAZE_POSE[2, 3] - BASE_PLATE_THICKNESS
    robot.viz.show_box("table", [1.2, 1.2, 0.02],
                       translation([0.3, 0, table_top - 0.01]), color=0x8b6b4a)
    robot.add_object(Puzzle("D", MAZE_POSE))
    robot.viz.show_points("path", (MAZE_POSE @ densify(maze_path("A"), 0.01))[:, :3, 3],
                          color=0xff0000, radius=0.003)

    wire_axis = MAZE_POSE[:2, 3]
    for q in list(KEYFRAMES) + list(KEYFRAMES[::-1]):
        assert robot.in_limits(q)
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        c = hoop_center(robot, robot.get_q())
        print(f"hoop z = {c[2]:.3f} m, "
              f"{1000 * np.linalg.norm(c[:2] - wire_axis):.1f} mm from the wire axis")

    robot.soft_home()
    print(f"{len(robot.contacts)} contact(s) with the puzzle")


if __name__ == "__main__":
    main()
