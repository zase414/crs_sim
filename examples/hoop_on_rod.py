"""
Slide the hoop onto a moving rod using the (lagged) overhead camera.

    ./start.sh examples/hoop_on_rod.py                 # lag compensated
    ./start.sh examples/hoop_on_rod.py --no-predict    # use the stale measurement
    ./start.sh examples/hoop_on_rod.py --lag 0.3 --save last.png

The rod oscillates in front of the robot. Every cycle the robot grabs a camera
image, finds the rod's top face (bright red pixels), back-projects it onto the
plane of the known rod top height and records the measurement with the frame's
exposure time. A line fitted through the last measurements predicts where the
rod will be when the robot gets there (now + a lead for the robot motion), the
hoop is moved above that point and lowered over the rod once it follows it
well. A new motion is commanded only when the previous one has finished
(the arm is slow and every move_to_q starts from rest, re-commanding it every
frame would keep it from getting anywhere).

The printed errors compare against the ground truth (rod.top()): "measured" is
how far the stale measurement is from where the rod is now, "aim" the same for
the prediction. Collisions are not simulated, the hoop only counts as hitting
the rod if it is below the rod top and farther from its axis than the
clearance.

This is a minimal example, not a robust solution: no outlier rejection, no
smooth trajectory, a straight line prediction of a sinusoidal motion.
"""
import argparse
import time

import numpy as np

from crs_sim import CRS97
from crs_sim.camera import write_png
from crs_sim.puzzles import HOOP_IN_FLANGE
from crs_sim.rod import MovingRod, oscillating
from crs_sim.transforms import rx, rz, translation

HOOP_INNER_RADIUS = 0.030  # [m]
ROD_RADIUS = 0.010
ROD_LENGTH = 0.20
ROD_CENTER = translation([0.45, 0.0, 0.0])  # rod bottom, middle of its motion


def hoop_target(robot, position, yaw, q_now):
    """
    Joint configuration with the hoop center at position and its axis
    vertical, closest to q_now. yaw turns the hoop about its axis.
    """
    hoop = translation(position) @ rz(yaw) @ rx(np.pi)
    flange = hoop @ np.linalg.inv(HOOP_IN_FLANGE)
    sols = [q for q in robot.ik(flange) if robot.in_limits(q)]
    if not sols:
        return None
    return min(sols, key=lambda q: np.abs(q - q_now).max())


def rod_top_pixels(image):
    """
    Pixel coordinates [u, v] of the rod's top face: bright red (the face
    looks at the camera, the darker rod sides are left out). BGR image.
    """
    b, g, r = (image[:, :, k].astype(int) for k in range(3))
    v, u = np.nonzero((r > 150) & (g < 100) & (b < 100))
    return np.column_stack([u, v])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lag", type=float, default=0.15, help="camera lag [s]")
    parser.add_argument("--no-predict", action="store_true",
                        help="aim at the last measurement, ignore the lag")
    parser.add_argument("--lead", type=float, default=0.4,
                        help="how far ahead of now to aim [s]")
    parser.add_argument("--period", type=float, default=10.0,
                        help="period of the rod motion [s]")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="give up after this many seconds")
    parser.add_argument("--save", help="save the last camera image as PNG")
    parser.add_argument("--headless", action="store_true", help="no viewer")
    args = parser.parse_args()

    robot = CRS97(visualize=not args.headless)
    robot.initialize(home=False)
    robot.set_speed_relative(1.0)
    robot.camera.lag = args.lag
    rod = MovingRod(oscillating(ROD_CENTER, [0.0, 0.08, 0.0], args.period),
                    radius=ROD_RADIUS, length=ROD_LENGTH)
    robot.add_object(rod)
    top_z = ROD_CENTER[2, 3] + ROD_LENGTH

    # start above the middle of the rod motion. The hoop sticks out sideways
    # from the wrist: the wrist joints stay away from their limits and the
    # wrist does not hide the rod from the camera. The yaw is kept constant,
    # so that the hoop does not swing around.
    yaw = -np.pi / 2
    q = hoop_target(robot, [0.45, 0.0, top_z + 0.08], yaw, robot.get_q())
    robot.move_to_q(q)
    robot.wait_for_motion_stop()

    measurements = []  # (exposure time, xy)
    height = 0.08      # hoop height above the rod top
    image = None
    clearance = HOOP_INNER_RADIUS - ROD_RADIUS
    hit = False
    deadline = time.monotonic() + args.timeout
    while height > -0.12 and time.monotonic() < deadline:
        image = robot.grab_image()
        t_exp = robot.camera.last_exposure_time
        pixels = rod_top_pixels(image)
        if len(pixels) == 0:
            print("rod not visible")
            continue
        xy = robot.camera.backproject(pixels.mean(axis=0), top_z)[0, :2]
        measurements = (measurements + [(t_exp, xy)])[-5:]
        if robot.in_motion():
            continue

        now = time.monotonic()
        if args.no_predict or len(measurements) < 3:
            aim = xy
        else:
            t = np.array([m[0] for m in measurements])
            p = np.array([m[1] for m in measurements])
            coef = np.polyfit(t - t[-1], p, 1)  # line through the measurements
            aim = coef[0] * (now + args.lead - t[-1]) + coef[1]

        # how far the hoop center is from the rod axis right now (ground truth)
        hoop = (robot.fk(robot.get_q()) @ HOOP_IN_FLANGE)[:3, 3]
        rod_xy = rod.top()[:2]
        error = np.linalg.norm(hoop[:2] - rod_xy)
        hit |= hoop[2] < top_z and error > clearance

        # descend only while the hoop follows the rod well
        if len(measurements) >= 3 and error < 0.5 * clearance:
            height -= 0.01
        q = hoop_target(robot, [*aim, top_z + height], yaw, robot.get_q())
        if q is None:
            print("target out of reach")
            break
        robot.move_to_q(q)

        print(f"image age {1000 * (now - t_exp):3.0f} ms | rod error: measured "
              f"{1000 * np.linalg.norm(xy - rod_xy):4.1f} mm, aim "
              f"{1000 * np.linalg.norm(aim - rod_xy):4.1f} mm | hoop "
              f"{1000 * error:5.1f} mm from the rod axis, "
              f"{100 * (hoop[2] - top_z):+5.1f} cm above its top")

    robot.wait_for_motion_stop()
    hoop = (robot.fk(robot.get_q()) @ HOOP_IN_FLANGE)[:3, 3]
    error = np.linalg.norm(hoop[:2] - rod.top()[:2])
    if hit or error > clearance:
        print("The hoop hit the rod.")
    else:
        print(f"The hoop is on the rod ({1000 * error:.1f} mm off center).")
    if args.save and image is not None:
        write_png(args.save, image)
    time.sleep(2.0)


if __name__ == "__main__":
    main()
