import numpy as np
import time
import cv2
from ctu_mitsubishi import Rv6s

def find_hoop_homography(images, hoop_positions):
    images = np.asarray(images)
    assert images.shape[0] == len(hoop_positions)

    # todo HW03: Detect circle in each image
    circle_centers = []

    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.medianBlur(gray_img, 3)
        rows = gray_img.shape[0]

        circle = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, rows / 8, param1 = 100, param2 = 100, minRadius = 50, maxRadius = 200)[0]
        circle_centers.append([circle[0][0], circle[0][1]])

    # todo HW03: Find homography using cv2.findHomography. Use the hoop positions and circle centers.
    hoops_2D = []
    for hoop_pos in hoop_positions:
        hoops_2D.append(hoop_pos[:2, 3])
    #print(hoops_2D)

    M, mask = cv2.findHomography(np.array(circle_centers), np.array(hoops_2D), cv2.RANSAC, 5.0)
    print(M)

    return M

robot = Rv6s() # Connect without a real robot for offline kinematics

robot.soft_home()
q0 = robot.get_q()

print("Current q:", np.rad2deg(q0).round())

pose = robot.fk(q0)
print("Current pose:", pose.round(3))

#starting pose
offset_pose = np.eye(4)
offset_pose[0, 3] = 0.1
offset_pose[1, 3] = 0.3
offset_pose[2, 3] = 0.3


target_pose = pose @ offset_pose
print("Target pose:", target_pose.round(3))
qs = robot.ik(target_pose)

q = min(qs, key=lambda q: np.linalg.norm(q - q0))
print("Moving to q:", np.rad2deg(q).round())
robot.move_to_q(q) # This would move the real robot

#moving along xy
def move(offset_pose):
    qi = robot.get_q()
    curr_pose = robot.fk(qi)

    target_pose = curr_pose @ offset_pose
    qs = robot.ik(target_pose)

    q = min(qs, key=lambda q: np.linalg.norm(q-qi))
    robot.move_to_q(q)

images = []
effector_poses = []
dir = -1
for i in range(5):
    for j in range(5):
        offset_pose = np.eye(4)
        offset_pose[1,3] = dir*0.05

        move(offset_pose)

        time.sleep(0.5)
        images.append(robot.grab_image())
        effector_poses.append(robot.fk(robot.get_q())) # this is nigga

    offset_pose = np.eye(4)
    offset_pose[1,3] = dir*0.05

    move(offset_pose)
    dir *=-1

M = find_hoop_homography(images, effector_poses)
print(M)

robot.stop_robot()
robot.close_connection()
