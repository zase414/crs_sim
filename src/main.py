from robot.clanker import Clanker
import numpy as np
from utils.image_utils import *
import time

# RUN IN SIMULATION: python -m crs_sim.launch [scene options] src/main.py [main.py args]

def main():
    try:
        robot = Clanker()
        robot.initialize()
        robot.soft_home()
        q0 = robot.get_q()
        robot.move_to_q(q0 + [0.5*np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])

        H = np.load("src/utils/homography.npy")
        images = []
        for i in range(14):
            images.append(robot.grab_image())
            time.sleep(2)

        pos = localize_base(images[0], H)
        print(pos)
        (rot_c_x,rot_c_y),R = base_rotation_params(images, H)

        final_pose = robot.fk(q0)
        final_pose[0, 3] = rot_c_x #0.3
        final_pose[1, 3] = rot_c_y
        final_pose[2, 3] = 0.11

        robot.move_abs(final_pose)
    finally:
        time.sleep(2)
        robot.soft_home()
        robot.wait_for_motion_stop()
        robot.stop_robot()
        robot.close_connection()

if __name__ == "__main__":
    main()
