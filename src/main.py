from robot.clanker import Clanker
import numpy as np
import cv2
from utils.image_utils import *
from utils.make_key_vectors import *
import time

# RUN IN SIMULATION: python -m crs_sim.launch [scene options] src/main.py [main.py args]

def main():
    try:
        robot = Clanker()
        robot.initialize()
        robot.soft_home()
        robot.wait_for_motion_stop()
        q0 = robot.get_q()

        H = np.load("src/utils/homography.npy")
        puzzle = np.load("src/utils/key_vectors/puzzle_c_se3_interp.npy")

        robot.move_to_q(q0 + [0.5*np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot.wait_for_motion_stop()
        images = []

        rot_time_start = time.time()

        image_taken_times = []
        photo_sleep = 2
        for i in range(6):
            images.append(robot.grab_image())
            image_taken_times.append(time.time())
            time.sleep(photo_sleep)

        robot.soft_home()
        robot.wait_for_motion_stop()

        (rot_c_x,rot_c_y),R, init_ang, first_id, rot_speed = base_rotation_params(images, H)
        rot_speed /= photo_sleep
        base_pos = localize_base(images[first_id], H)
        base_pos[2,3] = 0.11
        print(base_pos)
    

        # turntable rotation center
        rot_center = np.eye(4)
        rot_center[0,3] = rot_c_x
        rot_center[1,3] = rot_c_y
        rot_center_inv = np.linalg.inv(rot_center)

        # time when images[first_id] (base_pos) was taken
        base_pos_time = image_taken_times[first_id]

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

if __name__ == "__main__":
    main()
