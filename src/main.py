from ctu_crs import CRS97
import numpy as np
from aruco_detection import detect_aruco
# RUN IN SIMULATION: python -m crs_sim.launch [scene options] src/main.py [main.py args]

def main():
    robot = CRS97()
    robot.initialize(home = False)

    # chci udelat nejakou kalibraci kamery
    q = robot.get_q()
    robot.move_to_q()

    img = robot.grab_image()   
    T_aruco = detect_aruco(img)

if __name__ == "__main__":
    main()
