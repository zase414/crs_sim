import numpy as np
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
    offset_pose[:3, 3] = [0.1, 0.3, 0.3]

    q = move(offset_pose, robot)
    print("Moving to q:", np.rad2deg(q).round())

    delta = np.array([0.05, 0, 0, 0, 0, 0])
    for _ in range(10):
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
