from robot.clanker import Clanker
import numpy as np
from utils.image_utils import *
from utils.make_key_vectors import *
import time

# RUN IN SIMULATION: python -m crs_sim.launch [scene options] src/main.py [main.py args]

def main():
    try:
        robot = Clanker()
        robot.initialize()
        robot.soft_home()
        q0 = robot.get_q()

        H = np.load("src/utils/homography.npy")
        puzzle = np.load("src/utils/key_vectors/puzzle_c_se3.npy")

        robot.move_to_q(q0 + [0.5*np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot.wait_for_motion_stop()
        images = []

        rot_time_start = time.time()

        photo_sleep = 2
        for i in range(6):
            images.append(robot.grab_image())
            time.sleep(photo_sleep)

        base_pos = localize_base(images[0], H)
        print(base_pos)
        (rot_c_x,rot_c_y),R, ang, first_id = base_rotation_params(images, H)
        rot_period = 28.2
        ang_speed = (2*np.pi)/rot_period

        # calculating initial angle
        ini_ang = ang - (ang_speed * photo_sleep) * first_id

        # needle head on string
        for point in puzzle:
            q = robot.get_q()
            current_angle = init_ang + ang_speed * (time.time() - rot_time_start)
            cos_ang = np.cos(current_angle)
            sin_ang = np.sin(current_angle)

            current_base_offset = np.array(
                [[cos_ang, sin_ang, 0, 0],
                [-sin_ang, cos_ang, 0, R],
                [0,         0,      1, 0],
                [0,         0,      0, 1]]
            )
            current_base_pos = base_pos @ current_base_offset

            next_robot_pose = current_base_pos @ point
            move_pose = robot.fk(q)
            move_pose[0,3] = next_robot_pose[0,3]
            move_pose[1,3] = next_robot_pose[1,3]
            move_pose[2,3] = next_robot_pose[2,3]

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
