import numpy as np
import time
import cv2
from ctu_mitsubishi import Rv6s

#moving along xy
def move_rel(robot, offset_pose):
    qi = robot.get_q()
    curr_pose = robot.fk(qi)

    target_pose = curr_pose @ offset_pose

    move_abs(robot, target_pose)

def move_abs(robot, target_pose):
    qi = robot.get_q()

    # IK + only solutions within joint limits
    qs = [q for q in robot.ik(target_pose) if robot.in_limits(q)]
    if not qs:
        print("Skipping move: unreachable, target", target_pose[:3, 3].round(3))
        return False

    q = min(qs, key=lambda q: np.linalg.norm(q - qi))

    robot.move_to_q(q)
    robot.wait_for_motion_stop()
