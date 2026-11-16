from robot.clanker import Clanker

import numpy as np
import cv2
from utils.image_utils import *
from utils.make_key_vectors import *
import time

# RUN IN SIMULATION: python -m crs_sim.launch [scene options] src/main.py [main.py args]

MAX_PHOTO_DISABILITY_COUNT = 3

def main():
    try:
        robot = Clanker()
        robot.initialize()
        time.sleep(1)
        robot.soft_home()
        robot.wait_for_motion_stop()
        time.sleep(1)
        q0 = robot.get_q()
        time.sleep(1)
        H = np.load("src/utils/homography.npy")
        puzzle = np.load("src/utils/key_vectors/puzzle_c_se3_interp.npy")

        img = cv2.imread("image_empty.jpg")
        height, width = img.shape[:2]  # 2050, 2448

        block_size = 20
        in_limits = np.zeros((height, width), dtype=bool)
        base_pose = robot.fk(q0)

        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                # Střed bloku; souřadnice obrazu jsou [x=sloupec, y=řádek]
                x = min(col + block_size // 2, width - 1)
                y = min(row + block_size // 2, height - 1)

                px = np.array([x, y, 1])
                robot_pos = H @ px
                robot_pos = robot_pos / robot_pos[2]

                move_pose = base_pose.copy()
                move_pose[0, 3] = robot_pos[0]
                move_pose[1, 3] = robot_pos[1]
                move_pose[2, 3] = 0.535 + 0.04

                reachable = robot.pose_in_limits(move_pose)
                in_limits[row:min(row + block_size, height),
                        col:min(col + block_size, width)] = reachable

        overlay = np.zeros_like(img, dtype=np.uint8)
        overlay[in_limits] = [0, 255, 0]
        overlay[~in_limits] = [0, 0, 255]

        alpha = 0.35
        result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        cv2.imwrite("overlay.jpg", result)



    finally:
        time.sleep(1)
        robot.soft_home()
        robot.wait_for_motion_stop()
        robot.stop_robot()
        robot.close_connection()
        print("EXIT SUCCESSFUL")


def is_in_reach(robot, puzzle_pose, base_pos, rot_center, rot_center_inv):
    q = robot.get_q()

    iterations = 10
    # Check for furthest point of puzzle
    for i in range(iterations):
            # angle the table turned since base_pos was taken
            d_ang = i/iterations * 2*np.pi
            cos_ang = np.cos(d_ang)
            sin_ang = np.sin(d_ang)

            current_base_offset = np.array(
                [[cos_ang, -sin_ang, 0, 0],
                [sin_ang,   cos_ang, 0, 0],
                [0,         0,       1, 0],
                [0,         0,       0, 1]]
            )
            # rotate base_pos about the turntable center (world frame)
            current_base_pos = rot_center @ current_base_offset @ rot_center_inv @ base_pos

            next_robot_pose = current_base_pos @ puzzle_pose
            move_pose = robot.fk(q)
            move_pose[0,3] = next_robot_pose[0,3]
            move_pose[1,3] = next_robot_pose[1,3]
            move_pose[2,3] = next_robot_pose[2,3] - 0.01

            if not robot.pose_in_limits(move_pose):
                return False

    return True



if __name__ == "__main__":
    main()
