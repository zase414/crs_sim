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

        robot.move_to_q(q0 + [0.5*np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot.wait_for_motion_stop()
        print(robot.fk(robot.get_q()))

        images = []
        valid_questionmark = False

        for i in range(MAX_PHOTO_DISABILITY_COUNT):
            rot_time_start = time.time()

            image_taken_times = []
            photo_sleep = 1
            for i in range(10):
                images.append(robot.grab_image())
                image_taken_times.append(time.time())
                time.sleep(photo_sleep)

            nigger = base_rotation_params(images, H)
            if nigger is not None:
                (rot_c_x,rot_c_y),R, init_ang, first_id, rot_speed = nigger
                valid_questionmark = True
                break

        # Did not found
        if not valid_questionmark:
            return

        #meassured period of rotation: 28.2, however, only Bulgarian constant works - 27
        rot_speed = np.sign(rot_speed) * ((2*np.pi)/27) if np.abs(rot_speed) > 0.05 else 0
        print("rotation period is", (2*np.pi)/rot_speed )
        base_pos = localize_base(images[first_id], H)
        base_pos[2,3] = 0.11
        print(base_pos)
    
        robot.soft_home()
        robot.wait_for_motion_stop()

        # turntable rotation center
        rot_center = np.eye(4)
        rot_center[0,3] = rot_c_x
        rot_center[1,3] = rot_c_y
        rot_center_inv = np.linalg.inv(rot_center)

        if not is_in_reach(robot, puzzle[0],base_pos, rot_center, rot_center_inv):
            print("position is not reachable fuck YOU")
            return


        # time when images[first_id] (base_pos) was taken
        base_pos_time = image_taken_times[first_id]

        print(puzzle.shape)
        start_high_pose = puzzle[0].copy()
        start_high_pose[2, 3] += 0.02
        puzzle = np.insert(puzzle, 0, start_high_pose, axis=0)
        print(puzzle.shape)


        # needle head on string
        for i, point in enumerate(puzzle):

            q = robot.get_q()
            # angle the table turned since base_pos was taken
            d_ang = rot_speed * (time.time() - base_pos_time)
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

            next_robot_pose = current_base_pos @ point
            move_pose = robot.fk(q)
            move_pose[0,3] = next_robot_pose[0,3]
            move_pose[1,3] = next_robot_pose[1,3]
            move_pose[2,3] = next_robot_pose[2,3] - 0.01

            robot.move_abs(move_pose)

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
