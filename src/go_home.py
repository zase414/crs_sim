from robot.clanker import Clanker

def main():
    try:
        robot = Clanker()
        robot.initialize()
        robot.soft_home()
    finally:
        robot.wait_for_motion_stop()
        robot.stop_robot()
        robot.close_connection()
        print("EXIT SUCCESSFUL")


if __name__ == "__main__":
    main()
