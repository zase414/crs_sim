import numpy as np
import math
from ctu_mitsubishi import Rv6s


def main():
  robot = Rv6s()
  try:
    robot.initialize()
    robot.soft_home()

    q0 = robot.get_q()
    pose = robot.fk(q0)

    print("Current q:", np.rad2deg(q0).round())
    print("Current pose:", pose.round(3))

    offset_pose = np.eye(4)
    offset_pose[:3, 3] = [-0.2, 0.0, 0.3]

    q = move(offset_pose, robot)
    delta = np.array([(math.pi), 0, 0, 0, 0, 0])
    robot.move_to_q(q)


    print("Moving to q:", np.rad2deg(q).round())

    steps = 50
    delta = np.array([(2*math.pi)/steps, 0, 0, 0, 0, 0])
    for _ in range(steps):
      q = q + delta
      robot.move_to_q(q)

  finally:
    robot.stop_robot()
    robot.close_connection()


def move(offset_pose, robot):
  qi = robot.get_q()
  curr_pose = robot.fk(qi)

  target_pose = curr_pose @ offset_pose
  solutions = robot.ik(target_pose)

  if not solutions:
    raise RuntimeError("Inverse kinematics returned no solutions")

  q = min(solutions, key=lambda candidate: np.linalg.norm(candidate - qi))
  robot.move_to_q(q)
  return q


if __name__ == "__main__":
  main()
