"""
Cartesian motion: trace a horizontal circle with the flange pointing down,
solving IK for every point and picking the solution closest to the previous one.

    ./start.sh examples/draw_circle.py --radius 0.1 --height 0.2
"""
import argparse

import numpy as np

from crs_sim import CRS97
from crs_sim.transforms import ry, translation


def closest_ik(robot, target, q_ref):
    sols = [q for q in robot.ik(target) if robot.in_limits(q)]
    if not sols:
        return None
    return min(sols, key=lambda q: np.abs(q - q_ref).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--center", type=float, nargs=2, default=[0.45, 0.0],
                        help="circle center x y [m]")
    parser.add_argument("--radius", type=float, default=0.1, help="[m]")
    parser.add_argument("--height", type=float, default=0.2, help="flange z [m]")
    parser.add_argument("--points", type=int, default=48)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="relative speed 0..1 (set_speed_relative)")
    args = parser.parse_args()

    robot = CRS97()
    robot.initialize()
    robot.set_speed_relative(args.speed)

    angles = np.linspace(0, 2 * np.pi, args.points + 1)
    centers = np.c_[args.center[0] + args.radius * np.cos(angles),
                    args.center[1] + args.radius * np.sin(angles),
                    np.full_like(angles, args.height)]
    robot.viz.show_points("circle", centers, color=0xff0000, radius=0.003)

    trail = []
    q = robot.get_q()
    for p in centers:
        # flange z axis pointing down
        target = translation(p) @ ry(np.pi)
        q = closest_ik(robot, target, q)
        if q is None:
            print(f"Point {np.round(p, 3)} is not reachable")
            return
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        trail.append(robot.fk(robot.get_q())[:3, 3])
        robot.viz.show_points("trail", trail, color=0x00a0ff, radius=0.004)

    error = np.max(np.linalg.norm(np.array(trail) - centers, axis=1))
    print(f"Max position error: {1000 * error:.3f} mm")
    robot.soft_home()


if __name__ == "__main__":
    main()
