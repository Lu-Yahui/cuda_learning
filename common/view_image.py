import cv2
import numpy as np
import sys

if __name__ == "__main__":
    channel = 1
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        image = np.fromfile(image_file, dtype=np.uint8)
        if len(sys.argv) == 3:
            channel = int(sys.argv[2])
        image = image.reshape((1080, 1443, channel))
        cv2.imshow("image", image)
        cv2.waitKey(0)
