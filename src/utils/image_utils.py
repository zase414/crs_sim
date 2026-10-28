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