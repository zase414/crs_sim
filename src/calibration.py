import numpy as np
import time
import cv2
from ctu_mitsubishi import Rv6s
from src.utils.image_utils import find_hoop_homography, localize_base
from src.utils.movement import move_rel, move_abs


robot = Rv6s() # Connect without a real robot for offline kinematics
robot.initialize()

robot.soft_home()
q0 = robot.get_q()

print("Current q:", np.rad2deg(q0).round())

pose = robot.fk(q0)
print("Current pose:", pose.round(3))

#starting pose
offset_pose = np.eye(4)
offset_pose[0, 3] = 0.1
offset_pose[1, 3] = 0.3
offset_pose[2, 3] = 0.6


target_pose = pose @ offset_pose
print("Target pose:", target_pose.round(3))
qs = robot.ik(target_pose)

q = min(qs, key=lambda q: np.linalg.norm(q - q0))
print("Moving to q:", np.rad2deg(q).round())
robot.move_to_q(q) # This would move the real robot
robot.wait_for_motion_stop()

images = []
effector_poses = []
dir = -1
for i in range(4):
    images.append(robot.grab_image())
    effector_poses.append(robot.fk(robot.get_q())) # this is nigga
    for j in range(5):
        offset_pose = np.eye(4)
        offset_pose[1,3] = dir*0.1

        move_rel(robot, offset_pose)

        time.sleep(0.5)
        images.append(robot.grab_image())
        effector_poses.append(robot.fk(robot.get_q())) # this is nigga

    if i !=3:
        offset_pose = np.eye(4)
        offset_pose[0,3] = -0.1
        move_rel(robot, offset_pose)
    dir *=-1


#np.save("images_hoop.npy", images)
#np.save("effector_poses.npy", effector_poses)
M = find_hoop_homography(images, effector_poses)

# save homography
np.save("homography.npy", M)

robot.move_to_q([np.pi*0.8, 0, 0, 0, 0, 0])
robot.wait_for_motion_stop()

img = robot.grab_image()
pose = localize_base(img, M)
print(pose)

final_pose = robot.fk(q0)
final_pose[0, 3] = pose[0]
final_pose[1, 3] = pose[1]
final_pose[2, 3] = 0.045

move_abs(robot, final_pose)

robot.stop_robot()
robot.close_connection()
