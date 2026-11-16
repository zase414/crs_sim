"""
Minimal crs_sim demo: home the robot, draw a target, reach it with IK.
Swap the import for ``from ctu_crs import CRS97`` (and drop the viz calls)
to run the same motion on the real robot.
"""
import time

import numpy as np

from crs_sim import CRS97

robot = CRS97(speedup=2.0)
robot.initialize()

# Target: flange 0.45 m in front of the robot, pointing down
target = np.eye(4)
target[:3, :3] = np.diag([-1.0, 1.0, -1.0])
target[:3, 3] = [0.45, 0.1, 0.25]
robot.viz.show_frame("target", target)

sols = [q for q in robot.ik(target) if robot.in_limits(q)]
if not sols:
    raise SystemExit("Target not reachable")
q = min(sols, key=lambda s: np.linalg.norm(s - robot.get_q()))

robot.move_to_q(q)
while robot.in_motion():
    robot.viz.show_frame("flange", robot.fk(robot.get_q()), scale=0.05)
    time.sleep(0.05)
robot.wait_for_motion_stop()
print("reached q =", np.round(robot.get_q(), 3))
print("position error [m] =", np.linalg.norm(robot.fk(robot.get_q())[:3, 3] - target[:3, 3]))

input("Press enter to go home and exit.")
robot.soft_home()
