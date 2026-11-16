from robot.clanker import Clanker

import numpy as np
import cv2
from utils.image_utils import *
from utils.make_key_vectors import *
import time
import matplotlib.pyplot as plt

# RUN IN SIMULATION: python -m crs_sim.launch [scene options] src/main.py [main.py args]

MAX_PHOTO_DISABILITY_COUNT = 3

def main():
    try:
        robot = Clanker()
        robot.initialize()
        time.sleep(1)
        robot.soft_home()
        robot.wait_for_motion_stop()
        time.sleep(1)
        q0 = robot.get_q()
        time.sleep(1)
        H = np.load("src/utils/homography.npy")
        puzzle = np.load("src/utils/key_vectors/puzzle_c_se3_interp.npy")

        img = cv2.imread("image_empty.jpg")
        height, width = img.shape[:2]  # 2050, 2448

        block_size = 20
        in_limits_low = np.zeros((height, width), dtype=bool)
        in_limits_high = np.zeros((height, width), dtype=bool)
        base_pose = robot.fk(q0)

        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                # Střed bloku; souřadnice obrazu jsou [x=sloupec, y=řádek]
                x = min(col + block_size // 2, width - 1)
                y = min(row + block_size // 2, height - 1)

                px = np.array([x, y, 1])
                robot_pos = H @ px
                robot_pos = robot_pos / robot_pos[2]

                move_pose = base_pose.copy()
                move_pose[0, 3] = robot_pos[0]
                move_pose[1, 3] = robot_pos[1]
                move_pose[2, 3] = 0.08 + 0.04

                reachable = robot.pose_in_limits(move_pose)
                in_limits_low[row:min(row + block_size, height),
                        col:min(col + block_size, width)] = reachable

        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                # Střed bloku; souřadnice obrazu jsou [x=sloupec, y=řádek]
                x = min(col + block_size // 2, width - 1)
                y = min(row + block_size // 2, height - 1)

                px = np.array([x, y, 1])
                robot_pos = H @ px
                robot_pos = robot_pos / robot_pos[2]

                move_pose = base_pose.copy()
                move_pose[0, 3] = robot_pos[0]
                move_pose[1, 3] = robot_pos[1]
                move_pose[2, 3] = 0.08 + 0.24

                reachable = robot.pose_in_limits(move_pose)
                in_limits_high[row:min(row + block_size, height),
                        col:min(col + block_size, width)] = reachable

        overlay = np.zeros_like(img, dtype=np.uint8)
        overlay[in_limits_low & in_limits_high] = [0, 255, 0]
        overlay[~(in_limits_low & in_limits_high)] = [0, 0, 255]

        alpha = 0.35
        result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        
        result = cv2.resize(result, (1024, 720))
        cv2.imwrite("overlay_final.jpg", result)

        # img = cv2.imread("image_empty.jpg")
        # height, width = img.shape[:2]  # 2050, 2448

        # block_size = 40
        # z_min, z_max, z_step = 0.08 + 0.04, 0.645, 0.025
        # nz = int(round((z_max - z_min) / z_step))   # počet vrstev
        # show_unreachable = True                      # červené kostky, kam to nedosáhne

        # base_pose = robot.fk(q0)

        # # hrany voxelů v obraze (poslední blok oříznutý na okraj) a ve výšce
        # col_edges = np.append(np.arange(0, width, block_size), width).astype(float)
        # row_edges = np.append(np.arange(0, height, block_size), height).astype(float)
        # z_edges = np.linspace(z_min, z_max, nz + 1)

        # col_c = (col_edges[:-1] + col_edges[1:]) / 2
        # row_c = (row_edges[:-1] + row_edges[1:]) / 2
        # z_c = (z_edges[:-1] + z_edges[1:]) / 2

        # def to_robot(c, r):
        #     """pixely [x=sloupec, y=řádek] -> souřadnice robota přes H"""
        #     p = H @ np.stack([c.ravel(), r.ravel(), np.ones(c.size)])
        #     p = p[:2] / p[2]
        #     return p[0].reshape(c.shape), p[1].reshape(c.shape)

        # # --- dosažitelnost ve středech voxelů, index [sloupec, řádek, z] ---
        # C, R = np.meshgrid(col_c, row_c, indexing="ij")
        # cx, cy = to_robot(C, R)

        # reach = np.zeros((*C.shape, nz), dtype=bool)
        # for k, z in enumerate(z_c):
        #     for i in range(C.shape[0]):
        #         for j in range(C.shape[1]):
        #             move_pose = base_pose.copy()
        #             move_pose[0, 3] = cx[i, j]
        #             move_pose[1, 3] = cy[i, j]
        #             move_pose[2, 3] = z
        #             reach[i, j, k] = robot.pose_in_limits(move_pose)
        #     print(f"z = {z:.3f} m: {reach[..., k].sum()}/{C.size}")

        # np.save("reachability_3d.npy", reach)

        # # --- rohy voxelů v souřadnicích robota (křivočará mřížka, H může otáčet/zkreslovat) ---
        # CE, RE = np.meshgrid(col_edges, row_edges, indexing="ij")
        # ex, ey = to_robot(CE, RE)
        # XE = np.broadcast_to(ex[..., None], (*ex.shape, nz + 1))
        # YE = np.broadcast_to(ey[..., None], (*ey.shape, nz + 1))
        # ZE = np.broadcast_to(z_edges[None, None, :], XE.shape)

        # # --- průhledné barvy ---
        # green = (0.0, 0.8, 0.0, 0.25)
        # red = (1.0, 0.0, 0.0, 0.12)
        # colors = np.empty((*reach.shape, 4))
        # colors[reach] = green
        # colors[~reach] = red
        # filled = np.ones_like(reach) if show_unreachable else reach

        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(projection="3d")
        # ax.voxels(XE, YE, ZE, filled, facecolors=colors, edgecolor=(0, 0, 0, 0.15), linewidth=0.3)
        # ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        # ax.set_box_aspect((np.ptp(XE), np.ptp(YE), np.ptp(ZE)))
        # ax.set_title("3D prostor dosažitelnosti (zelená = dosažitelné, červená = mimo limity)")
        # plt.show()



    finally:
        time.sleep(1)
        robot.soft_home()
        robot.wait_for_motion_stop()
        robot.stop_robot()
        robot.close_connection()
        print("EXIT SUCCESSFUL")


def is_in_reach(robot, puzzle_pose, base_pos, rot_center, rot_center_inv):
    q = robot.get_q()

    iterations = 10
    # Check for furthest point of puzzle
    for i in range(iterations):
            # angle the table turned since base_pos was taken
            d_ang = i/iterations * 2*np.pi
            cos_ang = np.cos(d_ang)
            sin_ang = np.sin(d_ang)

            current_base_offset = np.array(
                [[cos_ang, -sin_ang, 0, 0],
                [sin_ang,   cos_ang, 0, 0],
                [0,         0,       1, 0],
                [0,         0,       0, 1]]
            )
            # rotate base_pos about the turntable center (world frame)
            current_base_pos = rot_center @ current_base_offset @ rot_center_inv @ base_pos

            next_robot_pose = current_base_pos @ puzzle_pose
            move_pose = robot.fk(q)
            move_pose[0,3] = next_robot_pose[0,3]
            move_pose[1,3] = next_robot_pose[1,3]
            move_pose[2,3] = next_robot_pose[2,3] - 0.01

            if not robot.pose_in_limits(move_pose):
                return False

    return True



if __name__ == "__main__":
    main()
