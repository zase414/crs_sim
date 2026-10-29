import cv2
import numpy as np

def find_hoop_homography(images, hoop_positions):
    images = np.asarray(images)
    assert images.shape[0] == len(hoop_positions)

    # todo HW03: Detect circle in each image
    circle_centers = []
    hoops_2D = []

    for i, (img, hoop_pos) in enumerate(zip(images, hoop_positions)):
        circle = find_hoop(img)
        if circle is None:
            print(f"find_hoop_homography: no circle in image {i}, skipping")
            continue
        circle_centers.append([circle[0][0], circle[0][1]])
        # todo HW03: Find homography using cv2.findHomography. Use the hoop positions and circle centers.
        hoops_2D.append(hoop_pos[:2, 3])
    #print(hoops_2D)

    if len(circle_centers) < 4:
        raise RuntimeError(f"Need at least 4 detected circles for homography, got {len(circle_centers)}")

    M, mask = cv2.findHomography(np.array(circle_centers), np.array(hoops_2D), cv2.RANSAC, 5.0)

    return M

def find_hoop(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img, 3)
    rows = gray_img.shape[0]

    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, rows / 8, param1 = 100, param2 = 40, minRadius = 40, maxRadius = 90)
    if circles is None:
        return None
    return circles[0]

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
    #print(corners)
    return corners, ids

def find_cam_intrinsics(imgs):
    squares_x, squares_y = 7, 5
    square_len = 0.035
    marker_len = 0.025

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, dictionary)
    charuco_detector = cv2.aruco.CharucoDetector(board)

    all_object_points = []
    all_image_points = []

    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #cv2.imshow("nig", gray)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) >= 4:
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
            
            if obj_points is not None and len(obj_points) >= 4:
                all_object_points.append(obj_points)
                all_image_points.append(img_points)

    if len(all_object_points) > 0:
        image_size = imgs[0].shape[:2][::-1]

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            all_object_points, all_image_points, image_size, None, None
        )

        return camera_matrix
    else:
        # ARRRR, no charucos detected
        return -1

def find_aruco_homography(path, marker_world_corners):
    # collect image files from a folder, or use the single file given
    if os.path.isdir(path):
        files = sorted(
            f for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            for f in glob.glob(os.path.join(path, ext))
        )
    else:
        files = [path]

    img_pts, world_pts = [], []

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue

        corners, ids = detect_aruco(img)
        if ids is None:
            continue

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if int(marker_id) not in marker_world_corners:
                continue
            img_pts.extend(marker_corners.reshape(4, 2))
            world_pts.extend(np.asarray(marker_world_corners[int(marker_id)], dtype=np.float32))

    if len(img_pts) < 4:
        return None  # not enough correspondences

    M, mask = cv2.findHomography(
        np.array(img_pts, dtype=np.float32),
        np.array(world_pts, dtype=np.float32),
        cv2.RANSAC,
        5.0,
    )
    return M

def localize_base(image, find_cam_intrinsics, homography_M):
    # Take the hoop points and transform them to image space with with M^-1
    M_inv = np.linalg.inv(homography_M)

    corners, ids = detect_aruco(image)

    if len(corners != 8):
        # No base found
        return -1

    aruco1_center = (corners[0][0] + corners[0][1] + corners[0][2] + corners[0][3]) / 4
    aruco2_center = (corners[1][0] + corners[1][1] + corners[1][2] + corners[1][3]) / 4

    center_of_base_cam = (aruco1_center + aruco2_center) / 2

    base_center_robot = homography_M @ center_of_base_cam
    return base_center_robot
    