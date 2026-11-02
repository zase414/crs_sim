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
            ste = time.time()
            images.append(robot.grab_image())
            ste1 = time.time() - ste
            time.sleep(photo_sleep - ste1)

        robot.soft_home()
        robot.wait_for_motion_stop()

        (rot_c_x,rot_c_y),R, init_ang, first_id = base_rotation_params(images, H)
        rot_period = 28.2
        ang_speed = (2*np.pi)/rot_period
        base_pos = localize_base(images[first_id], H)
        base_pos[2,3] = 0.11
        print(base_pos)

        # # visualize initial_rotation
        # vis_img = images[first_id]
        # inv_H = np.linalg.inv(H)
        # start_line = inv_H @ np.array([rot_c_x, rot_c_y, 1])
        # start_x = int(start_line[0] / start_line[2])
        # start_y = int(start_line[1] / start_line[2])
        # end_x = int(start_x + R * np.cos(init_ang))
        # end_y = int(start_y + R * np.sin(init_ang))
        # cv2.line(vis_img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 3)
        # vis_img = cv2.resize(vis_img, (1280, 720))
        # cv2.imshow("niggers, hate them", vis_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # turntable rotation center
        rot_center = np.eye(4)
        rot_center[0,3] = rot_c_x
        rot_center[1,3] = rot_c_y
        rot_center_inv = np.linalg.inv(rot_center)

        # time when images[first_id] (base_pos) was taken
        base_pos_time = rot_time_start + photo_sleep * first_id

        # hoop center in the flange frame (fk gives the flange)
        hoop_in_flange = np.eye(4)
        hoop_in_flange[0,3] = -0.135
        hoop_in_flange[2,3] = 0.012
        flange_in_hoop = np.linalg.inv(hoop_in_flange)

        # stupid interpolation: 10 steps between key points, position linear,
        # orientation turned about one axis (rotation vector times s)
        approach = puzzle[0].copy()
        approach[2,3] += 0.05   # 5 cm above the free end of the wire
        path = [approach]
        for a, b in zip(puzzle[:-1], puzzle[1:]):
            rot = cv2.Rodrigues(a[:3,:3].T @ b[:3,:3])[0]
            for s in np.linspace(0, 1, 10, endpoint=False):
                p = a.copy()
                p[:3,:3] = a[:3,:3] @ cv2.Rodrigues(s * rot)[0]
                p[:3,3] = (1 - s) * a[:3,3] + s * b[:3,3]
                path.append(p)
        path.append(puzzle[-1])

        # how long one move takes, measured on the previous move
        step_time = 0.0

        # needle head on string
        for point in path:
            # angle the table turned since base_pos was taken, when this move ends
            d_ang = ang_speed * (time.time() + step_time - base_pos_time)
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

            move_pose = current_base_pos @ point @ flange_in_hoop

            move_start = time.time()
            robot.move_abs(move_pose)
            # the first move from home is long, do not use it as a guess
            step_time = min(time.time() - move_start, 0.5)

    finally:
        time.sleep(1)
        robot.soft_home()
        robot.wait_for_motion_stop()
        robot.stop_robot()
        robot.close_connection()
        print("EXIT SUCCESSFUL")

if __name__ == "__main__":
    main()
