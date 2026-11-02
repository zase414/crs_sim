from robot.clanker import Clanker
import numpy as np
from utils.image_utils import *
import time

# RUN IN SIMULATION: python -m crs_sim.launch [scene options] src/main.py [main.py args]

def main():
    try:
        c = Clanker()
        c.initialize()
        c.soft_home()
        c.wait_for_motion_stop()
        q0 = c.get_q()

        H = np.load("src/utils/homography.npy")

        c.move_to_q(q0 + [0.5*np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
        c.wait_for_motion_stop()
        images = []
        for i in range(6):
            images.append(c.grab_image())
            time.sleep(2)

        pos = localize_base(images[0], H)
        print(pos)
        (rot_c_x,rot_c_y),R = base_rotation_params(images, H)

        final_pose = c.fk(q0)
        final_pose[0, 3] = rot_c_x #0.3
        final_pose[1, 3] = rot_c_y
        final_pose[2, 3] = 0.11

        c.move_abs(final_pose)
    finally:
        time.sleep(2)
        c.soft_home()
        c.wait_for_motion_stop()
        c.stop_robot()
        c.close_connection()

if __name__ == "__main__":
    main()
