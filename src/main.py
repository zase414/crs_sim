from ctu_mitsubishi import Rv6s
import numpy as np
import utils.image_utils
# RUN IN SIMULATION: python -m crs_sim.launch [scene options] src/main.py [main.py args]

def main():
    robot = Rv6s()
    robot.initialize()
    robot.soft_home()
    robot.wait_for_motion_stop()

    # chci udelat nejakou kalibraci kamery
    q = robot.get_q()
    robot.move_to_q(q + [0.8*np.pi, 0.0, 0.0, 1.0, 0.0, 0.0])

    img = robot.grab_image()   
    corners, ids = detect_aruco(img)

    pos = localize_base(img, )
    print(pos)

if __name__ == "__main__":
    main()
