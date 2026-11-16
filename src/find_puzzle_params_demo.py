import numpy as np
from ctu_mitsubishi import Rv6s
from src.utils.image_utils import localize_base
from src.utils.movement import move_rel, move_abs


def main():
    H = np.load("src/utils/homography.npy")
    robot = Rv6s()
    try:
        robot.initialize()
        robot.soft_home()
        q0 = robot.get_q()
        pose = robot.fk(q0)


        for i in range(20):
            img = robot.grab_image()
            pos = localize_base(img, H)
            

        # if pos != -1:
        #     final_pose = robot.fk(q0)
        #     final_pose[0, 3] = pos[0]
        #     final_pose[1, 3] = pos[1]
        #     final_pose[2, 3] = 0.045

        #     move_abs(robot, final_pose)
    
    finally:
        robot.stop_robot()
        robot.close_connection()
        
if __name__ == "__main__":
    main()

