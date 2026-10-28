def find_hoop_homography(images, hoop_positions):
    images = np.asarray(images)
    assert images.shape[0] == len(hoop_positions)

    # todo HW03: Detect circle in each image
    circle_centers = []

    for img in images:
        circle = find_hoop(img)
        circle_centers.append([circle[0][0], circle[0][1]])

    # todo HW03: Find homography using cv2.findHomography. Use the hoop positions and circle centers.
    hoops_2D = []
    for hoop_pos in hoop_positions:
        hoops_2D.append(hoop_pos[:2, 3])
    #print(hoops_2D)

    M, mask = cv2.findHomography(np.array(circle_centers), np.array(hoops_2D), cv2.RANSAC, 5.0)

    return M

def find_hoop(img):
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_img = cv2.medianBlur(gray_img, 3)
  rows = gray_img.shape[0]

  circle = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, rows / 8, param1 = 100, param2 = 100, minRadius = 50, maxRadius = 200)[0]
  return circle

def detect_aruco(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dictionary)

    corners, ids, rejected = detector.detectMarkers(gray)
    #cv2.aruco.drawDetectedMarkers(img, corners, ids)
    #smaller = cv2.resize(img, (1280, 720))
    #cv2.imshow("Aruco", smaller)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print(corners)
    return corners, ids

def find_cam_intrinsics(imgs):
    squares_x, squares_y = 5, 7
    square_len = 0.035
    marker_len = 0.025

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, dictionary)

    all_charuco_corners = []
    all_charuco_ids = []

    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_aruco(img)

        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)  
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) >= 4:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)      

    if len(all_charuco_corners) > 0:
        image_size = imgs[0][::-1]

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, board, image_size, None, None
        )

        return camera_matrix
    else:
        # ARRRR, no charucos detected
        return -1