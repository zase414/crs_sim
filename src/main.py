from ctu_mitsubishi import Rv6s
import numpy as np
from src.utils.image_utils import *
from src.utils.movement import move_rel, move_abs

# RUN IN SIMULATION: python -m crs_sim.launch [scene options] src/main.py [main.py args]

def main():
    robot = Rv6s()
    robot.initialize()
    robot.soft_home()
    robot.wait_for_motion_stop()
    q0 = robot.get_q()

    H = np.load("src/utils/homography.npy")

    robot.move_to_q(q0 + [0.5*np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
    robot.wait_for_motion_stop()

    img = robot.grab_image()   

    detect_aruco(img)

    pos = localize_base(img, H)
    print(pos)
    final_pose = robot.fk(q0)
    final_pose[0, 3] = pos[0,3]
    final_pose[1, 3] = pos[1,3]
    final_pose[2, 3] = 0.045

    move_abs(robot, final_pose)

if __name__ == "__main__":
    main()
