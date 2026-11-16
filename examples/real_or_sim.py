"""
The same control code for the simulator and the real robot. Only the import
differs; everything below uses the ctu_crs interface.

    ./start.sh examples/real_or_sim.py          # simulator
    python examples/real_or_sim.py --real       # real CRS97 (needs ctu_crs)

On the real robot this performs a hard home and small joint motions around
the home configuration, make sure the workspace is clear.
"""
import argparse
import time

import numpy as np


def run(robot):
    robot.initialize()                      # hard home + soft home
    print("home q:", np.round(robot.get_q(), 3))

    robot.set_speed_relative(0.5)
    for dq in ([0.3, 0, 0, 0, 0, 0], [0, 0.15, -0.2, 0, 0.3, 0],
               [-0.3, 0, 0, 0.4, 0, -0.5]):
        q = robot.q_home + np.array(dq)
        assert robot.in_limits(q)
        robot.move_to_q(q)                  # non-blocking
        while robot.in_motion():
            time.sleep(0.1)                 # do something useful here
        robot.wait_for_motion_stop()
        print("at", np.round(robot.get_q(), 3),
              "flange", np.round(robot.fk(robot.get_q())[:3, 3], 3))

    robot.soft_home()
    robot.release()
    robot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--real", action="store_true", help="use the real robot")
    args = parser.parse_args()

    if args.real:
        from ctu_crs import CRS97
    else:
        from crs_sim import CRS97
    run(CRS97())


if __name__ == "__main__":
    main()
