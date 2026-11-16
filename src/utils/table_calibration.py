import cv2
import numpy
import glob
import numpy as np
from image_utils import find_cam_intrinsics

def main():
    images = [cv2.imread(file) for file in glob.glob("src/utils/photos/*.bmp")]
    K = np.array(find_cam_intrinsics(images))
    print(K)
    np.save("src/utils/cam_K.npy", K)

if __name__ == "__main__":
    main()