import numpy as np
import cv2
from ctu_mitsubishi import Rv6s

def main():
  robot = Rv6s() # Connect without a real robot for offline kinematics
  robot.initialize()
  robot.soft_home()
  q0 = robot.get_q()
  pose = robot.fk(q0)

  print("Current q:", np.rad2deg(q0).round())
  print("Current pose:", pose.round(3))

  #starting pose
  offset_pose = np.eye(4)
  offset_pose[0, 3] = 0.1
  offset_pose[1, 3] = 0.3
  offset_pose[2, 3] = 0.3

  move(offset_pose, robot)
  print("Moving to q:", np.rad2deg(q).round())
  for i in range(10):
    robot.move_to_q(q + [0.05,0,0,0,0,0])

  robot.stop_robot()
  robot.close_connection()

def move(offset_pose, robot):
    qi = robot.get_q()
    curr_pose = robot.fk(qi)

    raget_pose = curr_pose @ offset_pose
    qs = robot.ik(raget_pose)

    q = min(qs, key=lambda q: np.linalg.norm(q-qi))
    robot.move_to_q(q)

if __name__ == "__main__":
  main()