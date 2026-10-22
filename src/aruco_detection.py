import numpy as np
import cv2

def detect_aruco(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dictionary)

    corners, ids, rejected = detector.detectMarkers(gray)
    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    smaller = cv2.resize(img, (1280, 720))
    cv2.imshow("Aruco", smaller)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

