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
        puzzle = np.load("src/utils/key_vectors/puzzle_c_se3.npy")

        robot.move_to_q(q0 + [0.5*np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot.wait_for_motion_stop()
        images = []

        rot_time_start = time.time()

        photo_sleep = 2
        for i in range(6):
            images.append(robot.grab_image())
            time.sleep(photo_sleep)

        print(base_pos)
        (rot_c_x,rot_c_y),R, init_ang, first_id = base_rotation_params(images, H)
        rot_period = 28.2
        ang_speed = (2*np.pi)/rot_period
        base_pos = localize_base(images[first_id], H)
        base_pos[2,3] = 0.11

        # visualize initial_rotation
        vis_img = images[0]
        inv_H = np.linagl.inv(H)
        start_line = inv_H @ np.linalg([rot_c_x, rot_c_y, 1])
        start_x = start_line[0] / start_line[2]
        start_y = start_line[1] / start_line[2]
        end_x = start_x + R * np.cos(init_ang)
        end_y = start_y + R * np.sin(init_ang)
        cv2.line(vis_img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 3)
        vis_img = cv2.resize(vis_img, (1280, 720))
        cv2.imshow("niggers, hate them", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
