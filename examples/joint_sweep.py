"""
Move every joint from home towards both of its limits, one by one, and print
the joint limits, home configuration and simulated joint speeds.

    ./start.sh examples/joint_sweep.py --fraction 0.8
"""
import argparse

import numpy as np

from crs_sim import CRS97


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--fraction", type=float, default=0.6,
                        help="how far towards the limit to go (0..1)")
    parser.add_argument("--speedup", type=float, default=2.0)
    args = parser.parse_args()

    robot = CRS97(speedup=args.speedup)
    robot.initialize()

    print("joint   q_min [deg]  q_home [deg]  q_max [deg]")
    for i, (lo, home, hi) in enumerate(zip(robot.q_min, robot.q_home, robot.q_max)):
        print(f"{i:5d}   {np.rad2deg(lo):11.1f}  {np.rad2deg(home):12.1f}  "
              f"{np.rad2deg(hi):11.1f}")

    for i in range(6):
        for limit in (robot.q_min[i], robot.q_max[i]):
            q = robot.q_home.copy()
            q[i] += args.fraction * (limit - q[i])
            print(f"joint {i} -> {np.rad2deg(q[i]):7.1f} deg")
            robot.move_to_q(q)
            robot.wait_for_motion_stop()
        robot.soft_home()


if __name__ == "__main__":
    main()
