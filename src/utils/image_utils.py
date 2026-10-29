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

    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, rows / 8, param1 = 100, param2 = 40, minRadius = 40, maxRadius = 200)
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


def localize_base(image, homography_M):

    corners, ids = detect_aruco(image)

    if len(corners) != 2:
        # No base found
        return None

    corners = np.array(corners)

    # get center
    c_o_aruco_mass = np.mean(corners, axis = 2).squeeze()
    #print(c_o_aruco_mass)

    center_of_base_cam = np.mean(c_o_aruco_mass, axis = 0).squeeze()
    #print(center_of_base_cam)
    #cv2.circle(image, (int(center_of_base_cam[0]), int(center_of_base_cam[1])), 10, (255, 0, 0), -1)
    #smaller = cv2.resize(image, (1280, 720))
    #cv2.imshow("center of base", smaller)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    homogen_center = np.append(center_of_base_cam, 1.0)

    base_center_robot = homography_M @ homogen_center
    X_real = base_center_robot[0] / base_center_robot[2]
    Y_real = base_center_robot[1] / base_center_robot[2]


    # get rotation
    tag = corners[0][0]
    edges_px = np.array([
    [tag[0][0], tag[0][1], 1.0],
    [tag[1][0], tag[1][1], 1.0]
    ])

    print(edges_px)

    robot_edges = homography_M @ edges_px.T
    print(robot_edges)

    robot_left_upper = robot_edges[:2,0] / robot_edges[2, 0]
    robot_right_upper = robot_edges[:2,1] / robot_edges[2, 1]

    dx = robot_left_upper[0] - robot_right_upper[0]
    dy = robot_left_upper[1] - robot_right_upper[1]

    ang = np.arctan2(dy, dx)
    print(ang)

    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    transform = np.eye(4)
    transform[0,3] = X_real
    transform[1,3] = Y_real
    transform[0,0] = cos_a
    transform[0,1] = sin_a
    transform[1,0] = -sin_a
    transform[1,1] = cos_a

    return transform
    

def find_circle_from_points(points):
    points = np.asarray(points)
    x = points[:, 0]
    y = points[:, 1]
    
    #Ax -b least square solution
    A = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
    B = x**2 + y**2
    
    c, residues, rank, s = np.linalg.lstsq(A, B, rcond=None)
    
    #result solution
    x_c = c[0]
    y_c = c[1]
    
    #R calculation
    C = c[2]
    R = np.sqrt(C + x_c**2 + y_c**2)
    
    return (x_c, y_c), R

def base_rotation_params(images, homography_M): #returns (x,y) ,r

    if len(images) < 3:
        print("not enough images to find parameters")
        return None
    
    poses = []
    idxs = []
    for idx, image in enumerate(images):
        pos = localize_base(image, homography_M)
        
        if pos is None:
            continue

        poses.append(pos)
        idxs.append(idx)

    if len(poses) < 3:
        print("not enough stand centers")
        return None
    
    pts = np.stack(poses)[:, :2, 3]
    return find_circle_from_points(pts)




    
        




