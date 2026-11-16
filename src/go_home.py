from ctu_mitsubishi import Rv6s
import cv2

def main():
    try:
        robot = Rv6s()
        robot.initialize()
        robot.soft_home()
        img = robot.grab_image()
        smaller = cv2.resize(img, (1280, 720))
        cv2.imshow("center of base", smaller)
        cv2.waitKey()
        cv2.destroyAllWindows()
    finally:
        robot.wait_for_motion_stop()
        robot.stop_robot()
        robot.close_connection()
        print("EXIT SUCCESSFUL")


if __name__ == "__main__":
    main()
