"""
The puzzle task as a virtual environment: any of the puzzles A - E placed on
the table (randomly or where you say), seen by the overhead camera, with
every touch of the robot detected. Optionally replays a joint trajectory and
judges it like the physical attempt.

    ./start.sh examples/puzzle_env.py --puzzle C --seed 7 --image frame.png
    ./start.sh examples/puzzle_env.py --puzzle A --x 0.45 --y 0 --table-z 0.026 \\
        --trajectory my_solution.txt

The trajectory file has one joint configuration [rad] per line (6 numbers,
or an Nx6 .npy array), executed with move_to_q / wait_for_motion_stop. The
verdict lists every contact and how far down the wire the hoop got (the
wire passing through the hoop opening, measured above the base plate top).

Use the same setup in your own code:

    robot = CRS97()
    puzzle = Puzzle("C", random_pose(7))
    robot.add_object(puzzle)          # viewer + camera + contact checking
    image = robot.grab_image()        # find the ArUco markers 1 and 2 here
    ...                               # your solution
    print(robot.contacts)
"""
import argparse

import numpy as np

from crs_sim import CRS97
from crs_sim.camera import write_png
from crs_sim.puzzles import BASE_PLATE_THICKNESS, HOOP_IN_FLANGE, Puzzle, random_pose
from crs_sim.transforms import pose, translation


def load_trajectory(path: str) -> np.ndarray:
    q = np.load(path) if path.endswith(".npy") else np.loadtxt(path, ndmin=2)
    assert q.ndim == 2 and q.shape[1] == 6, "expected N x 6 joint values"
    return q


def lowest_threaded_height(robot, puzzle, trajectory, step=0.002):
    """
    Lowest hoop center height above the base plate top at which the wire went
    through the hoop, along the executed joint path; None if it never did.
    """
    best = None
    for qa, qb in zip(trajectory[:-1], trajectory[1:]):
        n = max(1, int(np.ceil(np.abs(qb - qa).sum() / step)))
        for s in np.linspace(0, 1, n + 1):
            hoop = robot.fk(qa + s * (qb - qa)) @ HOOP_IN_FLANGE
            if puzzle.wire_through_hoop(hoop):
                h = (np.linalg.inv(puzzle.pose) @ hoop)[2, 3]
                best = h if best is None else min(best, h)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--puzzle", default="A", choices=list("ABCDE"))
    parser.add_argument("--seed", type=int, help="random placement (default: --x/--y/--yaw)")
    parser.add_argument("--x", type=float, default=0.45, help="puzzle position [m]")
    parser.add_argument("--y", type=float, default=0.0, help="puzzle position [m]")
    parser.add_argument("--yaw", type=float, default=0.0, help="puzzle rotation [rad]")
    parser.add_argument("--table-z", type=float, default=0.0,
                        help="table top height in the robot base frame [m]")
    parser.add_argument("--image", help="save a camera frame (PNG) here")
    parser.add_argument("--trajectory", help="joint configurations to execute and judge")
    parser.add_argument("--speedup", type=float, default=1.0, help="motion time scale")
    parser.add_argument("--headless", action="store_true", help="no viewer")
    args = parser.parse_args()

    robot = CRS97(visualize=not args.headless, speedup=args.speedup)
    if args.seed is not None:
        maze_pose = random_pose(args.seed, table_z=args.table_z)
    else:
        maze_pose = pose([args.x, args.y, args.table_z + BASE_PLATE_THICKNESS],
                         args.yaw)
    puzzle = Puzzle(args.puzzle, maze_pose)
    if robot.viz is not None:
        robot.viz.show_box("table", [1.2, 1.2, 0.02],
                           translation([0.3, 0, args.table_z - 0.01]),
                           color=0x8b6b4a)
    robot.add_object(puzzle)

    print(f"Puzzle {args.puzzle}, maze frame in the robot base frame:\n"
          f"{np.round(puzzle.pose, 4)}")
    for i, corners in puzzle.marker_corners().items():
        print(f"ArUco {i} corners [m]: {np.round(corners, 4).tolist()}")

    robot.initialize()
    if args.image:
        write_png(args.image, robot.grab_image())
        print(f"Camera frame saved to {args.image}")

    if args.trajectory:
        trajectory = load_trajectory(args.trajectory)
        executed = [robot.get_q()]
        for i, q in enumerate(trajectory):
            if not robot.in_limits(q):
                print(f"Configuration {i} is out of the joint limits, stopping.")
                break
            robot.move_to_q(q)
            robot.wait_for_motion_stop()
            executed.append(robot.get_q())
        height = lowest_threaded_height(robot, puzzle, np.array(executed))

        print("\nVerdict:")
        for event in robot.contacts:
            print(f"  {event}")
        print(f"  {len(robot.contacts)} contact(s)")
        if height is None:
            print("  the hoop never got onto the wire")
        else:
            print(f"  hoop threaded down to {1000 * height:.0f} mm above the base plate")

    if robot.viz is not None:
        input("Press enter to exit.")


if __name__ == "__main__":
    main()
